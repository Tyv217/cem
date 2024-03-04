import logging
import numpy as np
import os
import pytorch_lightning
import random
import sklearn.model_selection
import torch
import torchvision

from torchvision import transforms

from pytorch_lightning import seed_everything

def load_mnist(
    cache_dir="mnist",
    seed=42,
    train_dataset_size=30000,
    test_dataset_size=10000,
    num_operands=10,
    selected_digits=list(range(10)),
    uncertain_width=0,
    renormalize=True,
    val_percent=0.2,
    batch_size=512,
    test_only=False,
    num_workers=-1,
    sample_concepts=None,
    as_channels=True,
    img_format='channels_first',
    output_channels=1,
    threshold=False,
    mixing=True,
    even_concepts=False,
    even_labels=False,
    threshold_labels=None,
    concept_transform=None,
    noise_level=0.0,
    test_noise_level=None,
):
    test_noise_level = (
        test_noise_level if (test_noise_level is not None) else noise_level
    )
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    pytorch_lightning.utilities.seed.seed_everything(seed)

    concept_groups = []
    for operand_digits in selected_digits:
        concept_groups.append(list(range(
            len(concept_groups),
            len(concept_groups) + len(operand_digits),
        )))

    

    transformations = transforms.Compose([
        transforms.Resize((7,7)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    ds_test = torchvision.datasets.MNIST(
        cache_dir,
        train=False,
        download=True,
        transform = transformations
    )

    ds_test.transform = transformations

    # Put all the images into a single np array for easy
    # manipulation
    x_test = []
    y_test = []

    for x, y in ds_test:
        x_test.append(
            np.expand_dims(x, axis=0)
        )
        y_test.append(
            np.expand_dims(y, axis=0)
        )
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    logging.debug(
        f"Dataset sizes:"
        f"\tx: {x_test.shape}"
        f"\ty: {y_test.shape}"
    )

    raise ValueError("For debug")

    x_test = torch.FloatTensor(x_test)
    
    y_test = torch.LongTensor(y_test)
    
    test_data = torch.utils.data.TensorDataset(x_test, y_test)

    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        transform = transformations
    )
    if test_only:
        return None, None, test_dl

    # Now time to do the same for the train/val datasets!
    ds_train = torchvision.datasets.MNIST(
        cache_dir,
        train=True,
        download=True,
        transform=transformations,
    )


    ds_train.transform = transformations

    x_train = []
    y_train = []

    for x, y in ds_train:
        x_train.append(
            np.expand_dims(x, axis=0)
        )
        y_train.append(
            np.expand_dims(y, axis=0)
        )

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)


    if val_percent:
        x_train, x_val, y_train, y_val = \
            sklearn.model_selection.train_test_split(
                x_train,
                y_train,
                test_size=val_percent,
            )
        
        x_val = torch.FloatTensor(x_val)
        if even_labels or (threshold_labels is not None):
            y_val = torch.FloatTensor(y_val)
        else:
            y_val = torch.LongTensor(y_val)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        val_dl = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        val_dl = None

    x_train = torch.FloatTensor(x_train)
    if even_labels or (threshold_labels is not None):
        y_train = torch.FloatTensor(y_train)
    else:
        y_train = torch.LongTensor(y_train)
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_dl, val_dl, test_dl


def generate_data(
        config,
        root_dir="mnist",
        seed=42,
        output_dataset_vars=False,
        rerun=False,
    ):
    selected_digits = config.get("selected_digits", list(range(2)))
    num_operands = config.get("num_operands", 32)
    if not isinstance(selected_digits[0], list):
        selected_digits = [selected_digits[:] for _ in range(num_operands)]
    elif len(selected_digits) != num_operands:
        raise ValueError(
            "If selected_digits is a list of lists, it must have the same "
            "length as num_operands"
        )
    even_concepts = config.get("even_concepts", False)
    even_labels = config.get("even_labels", False)
    threshold_labels = config.get("threshold_labels", None)

    if even_concepts:
        num_concepts = num_operands
        concept_group_map = {
            i: [i] for i in range(num_operands)
        }
    else:
        num_concepts = 0
        concept_group_map = {}
        n_tasks = 1 # Zero is always included as a possible sum
        for operand_idx, used_operand_digits in enumerate(selected_digits):
            num_curr_concepts = len(used_operand_digits) if len(used_operand_digits) > 2 else 1
            concept_group_map[operand_idx] = list(range(num_concepts, num_concepts + num_curr_concepts))
            num_concepts += num_curr_concepts
            n_tasks += np.max(used_operand_digits)

    n_tasks = 10

    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                root_dir,
                f"selected_groups_sampling_{sampling_percent}_operands_{num_operands}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
            new_n_concepts = len(selected_concepts)
        else:
            new_n_concepts = int(np.ceil(num_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                root_dir,
                f"selected_concepts_sampling_{sampling_percent}_operands_{num_operands}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(num_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        def concept_transform(sample):
            return sample[:, selected_concepts]
        num_concepts = new_n_concepts
        concept_group_map = new_concept_group
        logging.debug(
            f"\t\tUpdated concept group map "
            f"(with {len(concept_group_map)} groups):"
        )
        for k, v in concept_group_map.items():
            logging.debug(f"\t\t\t{k} -> {v}")
    else:
        concept_transform = None
    train_dl, val_dl, test_dl = load_mnist(
        cache_dir=root_dir,
        seed=seed,
        train_dataset_size=config.get("train_dataset_size", 30000),
        test_dataset_size=config.get("test_dataset_size", 10000),
        num_operands=num_operands,
        selected_digits=selected_digits,
        uncertain_width=config.get("uncertain_width", 0),
        renormalize=config.get("renormalize", True),
        val_percent=config.get("val_percent", 0.2),
        batch_size=config.get("batch_size", 512),
        test_only=config.get("test_only", False),
        num_workers=config.get("num_workers", -1),
        sample_concepts=config.get("sample_concepts", None),
        as_channels=config.get("as_channels", True),
        img_format=config.get("img_format", 'channels_first'),
        output_channels=config.get("output_channels", 1),
        threshold=config.get("threshold", True),
        mixing=config.get("mixing", True),
        even_labels=even_labels,
        threshold_labels=threshold_labels,
        even_concepts=even_concepts,
        concept_transform=concept_transform,
        noise_level=config.get("noise_level", 0),
        test_noise_level=config.get(
            "test_noise_level",
            config.get("noise_level", 0),
        ),
    )

    x_sample = next(iter(train_dl))[0]
    n_concepts = x_sample.shape[1] * (1 if len(x_sample.shape) < 2 else x_sample.shape[2])

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, None
    return (
        train_dl,
        val_dl,
        test_dl,
        None,
        (n_concepts, n_tasks, concept_group_map)
    )