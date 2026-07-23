"""Utilities of masking."""

from collections.abc import Callable
import torch
from transformers import VideoMAEForPreTraining


def _random_sampling(num_ppln: int, num_smpls: int) -> torch.Tensor:
    inds = torch.randperm(num_ppln)
    slct = torch.zeros(num_ppln).bool()
    slct[inds[:num_smpls]] = True
    return slct


def random_patch_masks(
    image_size: int, patch_size: int, mask_ratio: float
) -> torch.Tensor:
    return _random_sampling(
        num_ppln=(image_size // patch_size) ** 2,
        num_smpls=int((image_size // patch_size) ** 2 * mask_ratio),
    ).reshape(image_size // patch_size, image_size // patch_size)


def whole_temporal_tube_cube_masks(
    patch_masks: torch.Tensor, num_tublets: int
) -> torch.Tensor:
    return patch_masks.expand(num_tublets, -1, -1).flatten()


def videomae_mask_factory(
    mask_ratio: float = 0.9,
) -> Callable[[VideoMAEForPreTraining], torch.Tensor]:
    def factory(model: VideoMAEForPreTraining) -> torch.Tensor:
        return whole_temporal_tube_cube_masks(
            random_patch_masks(
                model.config.image_size, model.config.patch_size, mask_ratio
            ),
            model.config.num_frames // model.config.tubelet_size,
        )

    return factory


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
