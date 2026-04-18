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
    unlabeled = True if split == "extra_unlabeled" else False
    shuffle = True if split == "extra_unlabeled" or split == "train" else False

    full_dataset = get_dataset(
        dataset="iwildcam", download=download, root_dir=root_path, unlabeled=unlabeled
    )

    dataset = full_dataset.get_subset(split, transform=transform)

    # Use the unified wrapper that mimics the ImageNet structure
    dataset = WildsToTorchWrapper(dataset, is_unlabeled=unlabeled)

    logger.info(f"iWildCam {split} dataset created with {len(dataset)} samples")

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

    return dataset, data_loader, dist_sampler


class WildsToTorchWrapper(torch.utils.data.Dataset):
    """
    Mimics the ImageNet wrapper by always returning (image, target).
    """

    def __init__(self, wilds_subset, is_unlabeled=False):
        self.dataset = wilds_subset
        self.is_unlabeled = is_unlabeled

    def __getitem__(self, i):
        if self.is_unlabeled:
            # metadata is at index 1 for unlabeled WILDS
            x, _ = self.dataset[i]
            target = -1  # Dummy target to match (img, target) signature
        else:
            # image, target, metadata
            x, y, _ = self.dataset[i]
            target = y

        return x, target

    def __len__(self):
        return len(self.dataset)
