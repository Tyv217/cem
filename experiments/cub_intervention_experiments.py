import argparse
import copy
import joblib
import numpy as np
import os

import logging
import torch
from pathlib import Path
from pytorch_lightning import seed_everything

from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
import cem.train.training as training
import cem.train.utils as utils
from intervention_utils import (
    intervene_in_cbm, CUB_CONCEPT_GROUP_MAP, random_int_policy
)

################################################################################
## GLOBAL CUB VARIABLES
################################################################################

# IMPORANT NOTE: THIS DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE BEING ABLE
#                TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download it can be found
#                in the original CBM paper's repository
#                found here: https://github.com/yewsiang/ConceptBottleneck
CUB_DIR = 'cem/data/CUB200/'
BASE_DIR = os.path.join(CUB_DIR, 'class_attr_data_10')


################################################################################
## MAIN FUNCTION
################################################################################


def main(
    rerun=False,
    result_dir='results/cub/',
    project_name='',
    num_workers=8,
    global_params=None,
    test_uncertain=False,
    include_uncertain_train=False,
    gpu=torch.cuda.is_available(),
):
    seed_everything(42)
    # parameters for data, model, and training
    og_config = dict(
        cv=5,
        max_epochs=300,
        patience=15,
        batch_size=128,
        num_workers=num_workers,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=5,
        normalize_loss=False,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=True,
        pretrain_model=True,
        c_extractor_arch="resnet34",
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        sampling_percent=1,

        momentum=0.9,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.0,
        embeding_activation=None,
        concat_prob=False,
    )
    gpu = 1 if gpu else 0
    utils.extend_with_global_params(og_config, global_params or [])

    train_data_path = os.path.join(BASE_DIR, 'train.pkl')
    if og_config['weight_loss']:
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = og_config.get("sampling_percent", 1)
    n_concepts, n_tasks = 112, 200
    if sampling_percent != 1:
        # Do the subsampling
        new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
        selected_concepts_file = os.path.join(
            result_dir,
            f"selected_concepts_sampling_{sampling_percent}.npy",
        )
        if (not rerun) and os.path.exists(selected_concepts_file):
            selected_concepts = np.load(selected_concepts_file)
        else:
            selected_concepts = sorted(
                np.random.permutation(n_concepts)[:new_n_concepts]
            )
            np.save(selected_concepts_file, selected_concepts)
        print("\t\tSelected concepts:", selected_concepts)
        def subsample_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        if og_config['weight_loss']:
            imbalance = np.array(imbalance)[selected_concepts]

        train_dl = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        val_dl = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )

        train_dl_uncertain = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        val_dl_uncertain = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )


        test_dl = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )
        test_dl_uncertain = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
            concept_transform=subsample_transform,
        )

        # And set the right number of concepts to be used
        n_concepts = new_n_concepts
    else:
        train_dl = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )
        val_dl = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )

        train_dl_uncertain = load_data(
            pkl_paths=[train_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )
        val_dl_uncertain = load_data(
            pkl_paths=[val_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )

        test_dl = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=False,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )
        test_dl_uncertain = load_data(
            pkl_paths=[test_data_path],
            use_attr=True,
            no_img=False,
            batch_size=og_config['batch_size'],
            uncertain_label=True,
            n_class_attr=2,
            image_dir='images',
            resampling=False,
            root_dir=CUB_DIR,
            num_workers=og_config['num_workers'],
        )

    sample = next(iter(train_dl))
    n_concepts, n_tasks = sample[2].shape[-1], 200
    print("Training sample shape is:", sample[0].shape)
    print("Training label shape is:", sample[1].shape)
    print("Training concept shape is:", sample[2].shape)

    os.makedirs(result_dir, exist_ok=True)
    old_results = {}
    if os.path.exists(os.path.join(result_dir, f'results.joblib')):
        old_results = joblib.load(
            os.path.join(result_dir, f'results.joblib')
        )

    results = {}
    for train_uncertain in set([False, include_uncertain_train]):
        used_train_dl = train_dl_uncertain if train_uncertain else train_dl
        used_val_dl = val_dl_uncertain if train_uncertain else val_dl
        for split in range(og_config["cv"]):
            print(f'Experiment {split+1}/{og_config["cv"]}')
            results[f'{split}'] = {}

            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptEmbeddingModel"
            config["extra_name"] = "Uncertain" if train_uncertain else ""
            config["shared_prob_gen"] = True
            config["sigmoidal_prob"] = True
            config["sigmoidal_embedding"] = False
            config['training_intervention_prob'] = 0.25
            config['concat_prob'] = False
            config['emb_size'] = config['emb_size']
            config["embeding_activation"] = "leakyrelu"
            mixed_emb_shared_prob_model,  mixed_emb_shared_prob_test_results = \
                training.train_model(
                    gpu=gpu if gpu else 0,
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    config=config,
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
                    test_dl=test_dl,
                    split=split,
                    result_dir=result_dir,
                    rerun=rerun,
                    project_name=project_name,
                    seed=split,
                    imbalance=imbalance,
                )
            training.update_statistics(
                results[f'{split}'],
                config,
                mixed_emb_shared_prob_model,
                mixed_emb_shared_prob_test_results,
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )
            results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
                intervene_in_cbm(
                    concept_selection_policy=random_int_policy,
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    imbalance=imbalance,
                    split=split,
                    adversarial_intervention=False,
                    rerun=rerun,
                    old_results=old_results.get(str(split), {}).get(
                        f'test_acc_y_ints_{full_run_name}'
                    ),
                )
            if test_uncertain:
                results[f'{split}'][f'test_acc_y_uncert_ints_{full_run_name}'] = \
                    intervene_in_cbm(
                        concept_selection_policy=random_int_policy,
                        concept_group_map=CUB_CONCEPT_GROUP_MAP,
                        intervened_groups=list(
                            range(
                                0,
                                len(CUB_CONCEPT_GROUP_MAP) + 1,
                                config.get('intervention_freq', 4),
                            )
                        ),
                        gpu=gpu if gpu else None,
                        config=config,
                        test_dl=test_dl_uncertain,
                        train_dl=used_train_dl,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        result_dir=result_dir,
                        imbalance=imbalance,
                        adversarial_intervention=False,
                        rerun=rerun,
                        old_results=old_results.get(str(split), {}).get(
                            f'test_acc_y_uncert_ints_{full_run_name}'
                        ),
                    )
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

            # train vanilla CBM models with both logits and sigmoidal
            # bottleneck activations
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["bool"] = False
            config["extra_dims"] = (config['emb_size'] - 1) * n_concepts
            config["extra_name"] = (
                "Uncertain_Logit" if train_uncertain else f"Logit"
            )
            config["bottleneck_nonlinear"] = "leakyrelu"
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = False
            extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
                training.train_model(
                    gpu=gpu if gpu else None,
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    config=config,
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
                    test_dl=test_dl,
                    split=split,
                    result_dir=result_dir,
                    rerun=rerun,
                    project_name=project_name,
                    seed=split,
                    imbalance=imbalance,
                )
            training.update_statistics(
                results[f'{split}'],
                config,
                extra_fuzzy_logit_model,
                extra_fuzzy_logit_test_results,
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )
            # No uncertain interventions here as it is unclear how to do that
            # when the bottleneck's activations are unconstrained
            results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
                intervene_in_cbm(
                    concept_selection_policy=random_int_policy,
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    split=split,
                    imbalance=imbalance,
                    adversarial_intervention=False,
                    rerun=rerun,
                    old_results=old_results.get(int(split), {}).get(
                        f'test_acc_y_ints_{full_run_name}'
                    ),
                )
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))


            # CBM with sigmoidal bottleneck
            config = copy.deepcopy(og_config)
            config["architecture"] = "ConceptBottleneckModel"
            config["extra_name"] = (
                "Uncertain_Sigmoid" if train_uncertain else f"Sigmoid"
            )
            config["bool"] = False
            config["extra_dims"] = 0
            config["sigmoidal_extra_capacity"] = False
            config["sigmoidal_prob"] = True
            extra_fuzzy_logit_model, extra_fuzzy_logit_test_results = \
                training.train_model(
                    gpu=gpu if gpu else None,
                    n_concepts=n_concepts,
                    n_tasks=n_tasks,
                    config=config,
                    train_dl=used_train_dl,
                    val_dl=used_val_dl,
                    test_dl=test_dl,
                    split=split,
                    result_dir=result_dir,
                    rerun=rerun,
                    project_name=project_name,
                    seed=split,
                    imbalance=imbalance,
                )
            training.update_statistics(
                results[f'{split}'],
                config,
                extra_fuzzy_logit_model,
                extra_fuzzy_logit_test_results,
            )
            full_run_name = (
                f"{config['architecture']}{config.get('extra_name', '')}"
            )
            results[f'{split}'][f'test_acc_y_ints_{full_run_name}'] = \
                intervene_in_cbm(
                    concept_selection_policy=random_int_policy,
                    concept_group_map=CUB_CONCEPT_GROUP_MAP,
                    intervened_groups=list(
                        range(
                            0,
                            len(CUB_CONCEPT_GROUP_MAP) + 1,
                            config.get('intervention_freq', 4),
                        )
                    ),
                    gpu=gpu if gpu else None,
                    config=config,
                    test_dl=test_dl,
                    train_dl=used_train_dl,
                    n_tasks=n_tasks,
                    n_concepts=n_concepts,
                    result_dir=result_dir,
                    split=split,
                    imbalance=imbalance,
                    adversarial_intervention=False,
                    rerun=rerun,
                    old_results=old_results.get(int(split), {}).get(
                        f'test_acc_y_ints_{full_run_name}'
                    ),
                )
            if test_uncertain:
                results[f'{split}'][f'test_acc_y_uncert_ints_{full_run_name}'] = \
                    intervene_in_cbm(
                        concept_selection_policy=random_int_policy,
                        concept_group_map=CUB_CONCEPT_GROUP_MAP,
                        intervened_groups=list(
                            range(
                                0,
                                len(CUB_CONCEPT_GROUP_MAP) + 1,
                                config.get('intervention_freq', 4),
                            )
                        ),
                        gpu=gpu if gpu else None,
                        config=config,
                        test_dl=test_dl_uncertain,
                        train_dl=used_train_dl,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        result_dir=result_dir,
                        split=split,
                        imbalance=imbalance,
                        adversarial_intervention=False,
                        rerun=rerun,
                        old_results=old_results.get(int(split), {}).get(
                            f'test_acc_y_uncert_ints_{full_run_name}'
                        ),
                    )

            # save results
            joblib.dump(results, os.path.join(result_dir, f'results.joblib'))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Runs CEM intervention experiments in CUB dataset.'
        ),
    )
    parser.add_argument(
        '--project_name',
        default='',
        help=(
            "Project name used for Weights & Biases monitoring. If not "
            "provided, then we will not log in W&B."
        ),
        metavar="name",

    )

    parser.add_argument(
        '--output_dir',
        '-o',
        default='results/cub_intervention_experiments/',
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use ./results/cub/."
        ),
        metavar="path",

    )

    parser.add_argument(
        '--rerun',
        '-r',
        default=False,
        action="store_true",
        help=(
            "If set, then we will force a rerun of the entire experiment even "
            "if valid results are found in the provided output directory. "
            "Note that this may overwrite and previous results, so use "
            "with care."
        ),

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
        "--test_uncertain",
        action="store_true",
        default=False,
        help="test uncertain concept labels during interventions.",
    )
    parser.add_argument(
        "--include_uncertain_train",
        action="store_true",
        default=False,
        help="includes uncertainty in concept labels at training time.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        default=False,
        help="forces CPU training.",
    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    args = parser.parse_args()
    if args.project_name:
        # Lazy import to avoid importing unless necessary
        import wandb
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(
        rerun=args.rerun,
        result_dir=args.output_dir,
        project_name=args.project_name,
        num_workers=args.num_workers,
        global_params=args.param,
        test_uncertain=args.test_uncertain,
        include_uncertain_train=args.include_uncertain_train,
        gpu=(not args.force_cpu) and (torch.cuda.is_available()),
    )