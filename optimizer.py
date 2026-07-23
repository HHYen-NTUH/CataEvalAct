"""Utilities of optimizers."""

from collections.abc import Callable
import torch


def adamw_factory(
    model: torch.nn.Module,
    paramgroups_factory: Callable[[torch.nn.Module], list[dict]] | None = None,
    **kwargs,
) -> torch.optim.AdamW:
    params = (
        paramgroups_factory(model)
        if paramgroups_factory is not None
        else model.parameters()
    )
    return torch.optim.AdamW(params, **kwargs)


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
