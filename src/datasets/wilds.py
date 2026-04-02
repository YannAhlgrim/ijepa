import torch
from logging import getLogger
from wilds import get_dataset

logger = getLogger()


def make_iwildcam(
    transform,
    batch_size,
    collator=None,
    split="extra_unlabeled",
    num_workers=8,
    world_size=1,
    rank=0,
    root_path="./wilds_data",
    download=True,
    pin_mem=True,
    drop_last=True,
):
    """
    Adapted loader for WILDS-iWildCam
    """

    shuffle = True if split == "extra_unlabeled" or split == "train" else False
    unlabeled = True if split == "extra_unlabeled" else False

    full_dataset = get_dataset(
        dataset="iwildcam", download=download, root_dir=root_path, unlabeled=unlabeled
    )

    dataset = full_dataset.get_subset(split, transform=transform)
    logger.info(f"iWildCam {split} dataset created with {len(dataset)} samples")

    if unlabeled:
        dataset = WildsToTorchWrapperUnlabeled(dataset)
    else:
        dataset = WildsToTorchWrapper(dataset)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    logger.info(f"iWildCam {split} data loader created")

    return dataset, data_loader, dist_sampler


class WildsToTorchWrapperUnlabeled(torch.utils.data.Dataset):
    """
    WILDS __getitem__ returns (image, metadata).
    """

    def __init__(self, wilds_subset):
        self.dataset = wilds_subset

    def __getitem__(self, i):
        x, _ = self.dataset[i]  # remove metadata
        return x

    def __len__(self):
        return len(self.dataset)


class WildsToTorchWrapper(torch.utils.data.Dataset):
    """
    WILDS __getitem__ returns (image, target, metadata).
    """

    def __init__(self, wilds_subset):
        self.dataset = wilds_subset

    def __getitem__(self, i):
        x, y, _ = self.dataset[i]  # remove metadata
        return x, y

    def __len__(self):
        return len(self.dataset)
