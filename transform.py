"""Utilities of data transforms."""

from collections.abc import Callable
from typing import Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tvtrnsfrm


class TransformWraper(Dataset):
    """Dataset wrapper of data transform."""

    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, ind: int) -> Any:
        return self.transform(self.dataset[ind])


NRMLMEAN = [0.485, 0.456, 0.406]
NRMLSTD = [0.229, 0.224, 0.225]


def default_transform() -> tvtrnsfrm._transform.Transform:
    return tvtrnsfrm.Compose(
        [
            tvtrnsfrm.CenterCrop(size=240),
            tvtrnsfrm.Resize(size=224),
            tvtrnsfrm.ToDtype(torch.float32, scale=True),
            tvtrnsfrm.Normalize(mean=NRMLMEAN, std=NRMLSTD),
        ]
    )


def default_augment_transform() -> tvtrnsfrm._transform.Transform:
    return tvtrnsfrm.Compose(
        [
            tvtrnsfrm.CenterCrop(size=(240, int(1.2 * 240))),
            tvtrnsfrm.RandomResizedCrop(size=224, scale=(0.6, 1), ratio=(1, 1)),
            tvtrnsfrm.ColorJitter(
                brightness=(0.7, 1.3),
                contrast=(0.7, 1.3),
                saturation=(0.5, 1.5),
                hue=(-0.05, 0.05),
            ),
            tvtrnsfrm.ToDtype(torch.float32, scale=True),
            tvtrnsfrm.Normalize(mean=NRMLMEAN, std=NRMLSTD),
        ]
    )


def default_pretrain_augment_transform() -> tvtrnsfrm._transform.Transform:
    return tvtrnsfrm.Compose(
        [
            tvtrnsfrm.CenterCrop(size=(240, int(1.2 * 240))),
            tvtrnsfrm.RandomResizedCrop(size=224, scale=(0.6, 1), ratio=(1, 1)),
            tvtrnsfrm.ToDtype(torch.float32, scale=True),
            tvtrnsfrm.Normalize(mean=NRMLMEAN, std=NRMLSTD),
        ]
    )


def logits_to_pixels(logits: torch.Tensor) -> torch.Tensor:
    return (
        ((logits * torch.tensor(NRMLSTD) + torch.tensor(NRMLMEAN)) * 255)
        .clip(0, 255)
        .to(torch.uint8)
    )


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
