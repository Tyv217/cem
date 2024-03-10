import argparse
import copy
import joblib
import json
import logging
import numpy as np
import os
import sys
import torch
import yaml
import PIL
import random
import pytorch_lightning as pl


from datetime import datetime
from pathlib import Path
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset

import cem.data.celeba_loader as celeba_data_module
import cem.data.mnist_add as mnist_add_data_module
import cem.data.mnist as mnist_data_module
from cem.models.acflow import ACFlow, ac_transform_dataloader

################################################################################
## MAIN FUNCTION
################################################################################

# Helper class to apply transformations to a dataset

def main(
    data_module,
    results_dir,
    experiment_config,
    num_workers=8,
    accelerator="auto",
    devices="auto"
):
    seed_everything(42)
    # parameters for data, model, and training
    experiment_config = copy.deepcopy(experiment_config)
    if 'shared_params' not in experiment_config:
        experiment_config['shared_params'] = {}
    # Move all global things into the shared params
    for key, vals in experiment_config.items():
        if key not in ['runs', 'shared_params']:
            experiment_config['shared_params'][key] = vals
    experiment_config['shared_params']['num_workers'] = num_workers
    logging.debug(
        f"Processing dataset..."
    )
    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
        data_module.generate_data(
            config=experiment_config['shared_params'],
            seed=42,
            output_dataset_vars=True,
            root_dir=experiment_config['shared_params'].get('root_dir', None),
        )
    logging.debug(
        f"Applying transformations..."
    )
    train_dl = ac_transform_dataloader(train_dl, n_tasks, experiment_config["batch_size"])
    # For now, we assume that all concepts have the same
    # aquisition cost
    experiment_config["shared_params"]["n_concepts"] = \
        experiment_config["shared_params"].get(
            "n_concepts",
            n_concepts,
        )
    experiment_config["shared_params"]["n_tasks"] = \
        experiment_config["shared_params"].get(
            "n_tasks",
            n_tasks,
        )

    
    logging.info(
        f"\tNumber of output classes: {n_tasks}"
    )
    logging.info(
        f"\tNumber of training concepts: {n_concepts}"
    )
    val_dl = ac_transform_dataloader(val_dl, n_tasks, experiment_config["batch_size"])
    test_dl = ac_transform_dataloader(test_dl, n_tasks, experiment_config["batch_size"])

    sample = next(iter(train_dl.dataset))

    logging.debug(
        f"Sample: {sample}"
    )

    task_class_weights = None

    if experiment_config['shared_params'].get('use_task_class_weights', False):
        logging.info(
            f"Computing task class weights in the training dataset with "
            f"size {len(train_dl)}..."
        )
        
        attribute_count = np.zeros((max(n_tasks, 2),))
        samples_seen = 0
        for i, data in enumerate(train_dl):
            y = data['y']
            if len(y.shape) > 1:
                y = y.squeeze(dim = 0)
            if n_tasks > 1:
                y = torch.nn.functional.one_hot(
                    y,
                    num_classes=n_tasks,
                ).clone().cpu().numpy()
            else:
                y = torch.cat(
                    [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                    dim=-1,
                ).clone().cpu().numpy()
            attribute_count += np.sum(y, axis=0)
            samples_seen += y.shape[0]
        print("Class distribution is:", attribute_count / samples_seen)
        if n_tasks > 1:
            task_class_weights = samples_seen / attribute_count - 1
        else:
            task_class_weights = np.array(
                [attribute_count[0]/attribute_count[1]]
            )


    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )
    loglevel = os.environ['LOGLEVEL']
    logging.info(f'Setting log level to: "{loglevel}"')

    results = {}
    for split in range(
        experiment_config['shared_params'].get("start_split", 0),
        experiment_config['shared_params']["trials"],
    ):
        results[f'{split}'] = {}
        now = datetime.now()
        print(
            f"[TRIAL "
            f"{split + 1}/{experiment_config['shared_params']['trials']} "
            f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
        )
        # And then over all runs in a given trial
        
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=experiment_config['shared_params'].get('max_epochs', 500),
            logger=False,
            enable_checkpointing = False
        )

        model = ACFlow(
            n_concepts = n_concepts, 
            n_tasks = n_tasks,
            layer_cfg = experiment_config['shared_params']['layer_cfg'], 
            affine_hids = experiment_config['shared_params']['affine_hids'], 
            linear_rank = experiment_config['shared_params']['linear_rank'],
            linear_hids = experiment_config['shared_params']['linear_hids'], 
            transformations = experiment_config['shared_params']['transformations'], 
            optimizer = experiment_config['shared_params']['optimizer'], 
            learning_rate = experiment_config['shared_params']['learning_rate'], 
            weight_decay = experiment_config['shared_params']['decay_rate'], 
            momentum = experiment_config['shared_params'].get('momentum', 0.9), 
            prior_units = experiment_config['shared_params']['prior_units'], 
            prior_layers = experiment_config['shared_params']['prior_layers'], 
            prior_hids = experiment_config['shared_params']['prior_hids'], 
            n_components = experiment_config['shared_params']['n_components'], 
            lambda_xent = 1, 
            lambda_nll = 1
        )

        trainer.fit(model, train_dl, val_dl)
        model.freeze()

        sample = next(iter(test_dl))
        image_size = sample[0].shape[-1]

        print(
            f"Testing inpaint on image size {image_size}x{image_size}..."
        )

        it = iter(test_dl)

        for i in range(10):
            data = next(it)

            inpaint_iters = int(np.random.rand() * 10) + 1

            print(data.keys())

            b = torch.ones_like(data[0])

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            start_point = (random.randint(0, 6), random.randint(0, 6))
            path = [start_point]
            while len(path) < inpaint_iters:
                current_point = path[-1]
                possible_moves = []
                for d in directions:
                    next_point = (current_point[0] + d[0], current_point[1] + d[1])
                    if (0 <= next_point[0] < image_size) and (0 <= next_point[1] < image_size) and (next_point not in path):
                        possible_moves.append(next_point)
                if not possible_moves:
                    break
                path.append(random.choice(possible_moves))

            inpaint_iters = len(path)

            pred = data[0].clone()

            for p in path:
                b[p[0]][p[1]] = 0
                pred[p[0]][p[1]] = 0

            def array_to_image(tensor):
                image_size = tensor.shape[-1]
                image_size = int(np.sqrt(image_size))
                image_size = (image_size, image_size)
                tensor = tensor*255
                tensor = np.array(tensor, dtype=np.uint8)
                tensor = np.reshape(tensor, image_size)
                if np.ndim(tensor)>3:
                    assert tensor.shape[0] == 1
                    tensor = tensor[0]
                return PIL.Image.fromarray(tensor, mode='L')
            counter = 0
            original = array_to_image(data[0].clone().cpu().numpy())
            original.save(f"{results_dir}/original_{i}.png")
            for p in path:
                pred_with = pred.clone()
                pred_without = pred.clone()
                pred_with[p[0]][p[1]] = 1
                pred_without[p[0]][p[1]] = 0
                m = b.clone()
                m[p[0]][p[1]] = 1
                batch_with = {}
                batch_with['x'] = pred_with
                batch_with['b'] = b
                batch_with['m'] = m
                batch_with['y'] = None
                
                logpu_with, logpo_with = model.predict_step(batch_with, counter)
                loglikel_with = torch.mean(torch.logsumexp(logpu_with + logpo_with, dim = 1) - torch.logsumexp(logpo_with, dim = 1))
                batch_without = {}
                batch_without['x'] = pred_with
                batch_without['b'] = b
                batch_without['m'] = m
                batch_without['y'] = None
                logpu_without, logpo_without = model.predict_step(batch_without, counter)
                loglikel_without = torch.mean(torch.logsumexp(logpu_without + logpo_without, dim = 1) - torch.logsumexp(logpo_without, dim = 1))

                pred[p[0]][p[1]] = 1 if loglikel_with > loglikel_without else 0

                inpainted = np.where(m.clone().cpu().numpy() == 1, pred.clone().cpu().numpy(), 0.5)
                inpainted = array_to_image(inpainted)
                inpainted.save(f"results/inpainted_{i}_{counter}.png")
                
                b[p[0]][p[1]] = 1

    return results


################################################################################
## Arg Parser
################################################################################


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Runs AC Flow experiments in a given dataset.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help=(
            "YAML file with the configuration for the experiment. If not "
            "provided, then we will use the default configuration for the "
            "dataset."
        ),
        metavar="config.yaml",
    )
    parser.add_argument(
        '--dataset',
        choices=[
            'cub',
            'celeba',
            'mnist_add'
        ],
        help=(
            "Dataset to run experiments for. Must be a supported dataset with "
            "a loader."
        ),
        metavar="ds_name",
        default=None,
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        help=(
            'number of workers used for data feeders. Do not use more workers '
            'than cores in the machine.'
        ),
        metavar='N',
        type=int,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="starts debug mode in our program.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="forces CPU training.",
    )
    return parser


################################################################################
## Main Entry Point
################################################################################

if __name__ == '__main__':
    # Build our arg parser first
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    if args.config:
        with open(args.config, "r") as f:
            loaded_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        loaded_config = {}

    if args.dataset is not None:
        loaded_config["dataset"] = args.dataset
    if loaded_config.get("dataset", None) is None:
        raise ValueError(
            "A dataset must be provided either as part of the "
            "configuration file or as a command line argument."
        )
    # if loaded_config["dataset"] == "cub":
    #     data_module = cub_data_module
    if loaded_config["dataset"] == "celeba":
        data_module = celeba_data_module
    elif loaded_config["dataset"] == "mnist_add":
        data_module = mnist_add_data_module
        num_operands = loaded_config.get('num_operands', 32)
    elif loaded_config["dataset"] == "mnist":
        data_module = mnist_data_module
    else:
        raise ValueError(f"Unsupported dataset {loaded_config['dataset']}!")

    main(
        data_module=data_module,
        results_dir=(
            loaded_config['results_dir']
        ),
        accelerator=(
            "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
            else "cpu"
        ),
        experiment_config=loaded_config
    )
