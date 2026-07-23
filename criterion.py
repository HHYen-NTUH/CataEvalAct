"""Utilities of loss functions."""

from collections.abc import Sequence, Callable
from typing import Any
import torch
from torch.utils.data import Dataset


def _label_as_last_element(item: Sequence) -> int:
    return item[-1]


def balanced_cross_entropy_loss(
    dataset: Dataset,
    num_classes: int,
    label_getter: Callable[[Any], int] = _label_as_last_element,
    label_smoothing: float = 0,
) -> torch.nn.CrossEntropyLoss:
    """Return cross-entropy loss with balanced class weights."""
    labels = torch.tensor([label_getter(dataset[ind]) for ind in range(len(dataset))])
    counts = torch.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


class FocalCrossEntropyLoss(torch.nn.Module):
    """Focal loss of cross entropy."""

    def __init__(
        self,
        gamma: float,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.weight: torch.Tensor | None
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = (
            torch.exp(-cross_entropy)
            if self.weight is None
            else torch.exp(-cross_entropy / self.weight[targets])
        )
        focal_loss = ((1 - probs) ** self.gamma * cross_entropy).mean()
        return focal_loss


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
