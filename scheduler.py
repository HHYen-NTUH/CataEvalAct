"""Utilities of learning rate schedulers."""

import torch


def constant_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)


def warmup_consine_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_epochs: int,
    warmuplr: float,
    lr: float,
    minlr: float,
) -> torch.optim.lr_scheduler.SequentialLR:
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmuplr / lr, total_iters=num_warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - num_warmup_epochs, eta_min=minlr
            ),
        ],
        milestones=[num_warmup_epochs],
    )


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
