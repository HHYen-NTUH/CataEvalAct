"""Utilities of VideoMAE model."""

from collections.abc import Callable
from typing import Literal
import torch
import lightning as ltn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    VideoMAEForPreTraining,
)
from mask import videomae_mask_factory
from optimizer import adamw_factory
from scheduler import constant_scheduler
from metric import classification_metrics
from transform import logits_to_pixels


INPUT_LABEL_BATCH = tuple[torch.Tensor, torch.Tensor]


def layerwise_lr_decay_paramgroups(
    model: VideoMAEForVideoClassification, baselr: float, decay: float
) -> list[dict]:
    num_layers = len(model.videomae.encoder.layer)
    paramgrps = [
        {
            "params": model.videomae.embeddings.parameters(),
            "lr": baselr * decay ** (num_layers + 1),
        },
        *[
            {
                "params": model.videomae.encoder.layer[depth].parameters(),
                "lr": baselr * decay ** (num_layers - depth),
            }
            for depth in range(num_layers)
        ],
        {
            "params": model.classifier.parameters(),
            "lr": baselr,
        },
    ]
    return paramgrps


class VideoMaeClsfrModule(ltn.LightningModule):
    """Lightning module of a VideoMAE model for downstream classification."""

    def __init__(
        self,
        num_classes: int,
        use_pretrained_checkpoint: bool = False,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        optimizer_factory: Callable[
            [torch.nn.Module], torch.optim.Optimizer
        ] = adamw_factory,
        scheduler_factory: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ] = constant_scheduler,
        mode: Literal["multiclass", "multilabel"] = "multiclass",
    ) -> None:
        super().__init__()
        self.save_hyperparameters("num_classes", "mode")  # NOTE:
        _path = "MCG-NJU/videomae-base"
        # _path = "MCG-NJU/videomae-base-ssv2"
        _problem_type = "multi_label_classification" if mode == "multilabel" else None
        self.videomae = (
            VideoMAEForVideoClassification.from_pretrained(
                _path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                problem_type=_problem_type,
            )
            if use_pretrained_checkpoint
            else VideoMAEForVideoClassification(
                VideoMAEConfig.from_pretrained(
                    _path, num_labels=num_classes, problem_type=_problem_type
                )
            )
        ).train()
        self.criterion = criterion
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        metrics = classification_metrics(num_classes, mode=mode)
        self.trainmetrics = metrics.clone(postfix="/train")
        self.valmetrics = metrics.clone(postfix="/val")
        self.mode = mode
        self.predict_logits = False  # default to predict probabilities

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_factory(self.videomae)
        scheduler = self.scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        match self.mode:
            case "multilabel":
                probs = torch.nn.functional.sigmoid(logits)
            case _:  # multi-class
                probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def training_step(self, batch: INPUT_LABEL_BATCH, batch_idx: int) -> STEP_OUTPUT:
        inputs, labels = batch
        outputs = self.videomae(pixel_values=inputs, labels=labels)
        loss = self.criterion(outputs.logits, labels)
        probs = self._probs_from_logits(outputs.logits)
        self.trainmetrics(probs, labels.int())
        self.log("Loss/train", loss, prog_bar=True)
        self.log_dict(self.trainmetrics)
        return loss

    def validation_step(self, batch: INPUT_LABEL_BATCH, batch_idx: int) -> STEP_OUTPUT:
        inputs, labels = batch
        outputs = self.videomae(pixel_values=inputs, labels=labels)
        loss = self.criterion(outputs.logits, labels)
        probs = self._probs_from_logits(outputs.logits)
        self.valmetrics(probs, labels.int())
        self.log("Loss/val", loss, prog_bar=True)
        self.log_dict(self.valmetrics)

    def predict_step(self, batch: INPUT_LABEL_BATCH, batch_idx: int) -> torch.Tensor:
        inputs, _ = batch
        outputs = self.videomae(pixel_values=inputs)
        if self.predict_logits:
            return outputs.logits
        else:
            probs = self._probs_from_logits(outputs.logits)
            return probs


def videomae_unpatchify_logits(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    model: VideoMAEForPreTraining,
) -> torch.Tensor:
    # Revert the forward method of VideoMAEForPreTraining model
    bsz = inputs.size(0)
    num_tubes = model.config.num_frames // model.config.tubelet_size
    tsz = model.config.tubelet_size
    num_ptchs = model.config.image_size // model.config.patch_size
    psz = model.config.patch_size
    num_chns = model.config.num_channels
    if model.config.norm_pix_loss:
        labels = inputs.view(
            bsz, num_tubes, tsz, num_chns, num_ptchs, psz, num_ptchs, psz
        )
        labels = labels.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        labels = labels.view(
            bsz, num_tubes * num_ptchs * num_ptchs, tsz * psz * psz, num_chns
        )
        _mean = labels.mean(dim=-2, keepdim=True)
        _std = labels.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        logits = logits.view(bsz, -1, tsz * psz * psz, num_chns)
        logits = logits * _std[mask] + _mean[mask]
        logits = logits.view(bsz, -1, tsz * psz * psz * num_chns)
    _logits = torch.zeros(
        bsz, num_tubes * num_ptchs * num_ptchs, tsz * psz * psz * num_chns
    )
    _logits[mask] = logits
    _logits = _logits.view(
        bsz, num_tubes, num_ptchs, num_ptchs, tsz, psz, psz, num_chns
    )
    _logits = _logits.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    _logits = _logits.view(
        bsz, num_tubes * tsz, num_chns, num_ptchs * psz, num_ptchs * psz
    )
    return _logits.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]


def videomae_unpatchify_masks(
    masks: torch.Tensor, inputs: torch.Tensor, model: VideoMAEForPreTraining
) -> torch.Tensor:
    num_ptchs = model.config.image_size // model.config.patch_size
    num_tubes = model.config.num_frames // model.config.tubelet_size
    batch_size, num_frames, num_chns, height, width = inputs.shape
    masks = masks.view(batch_size, 1, num_tubes, num_ptchs, num_ptchs)
    masks = torch.nn.functional.interpolate(
        masks.float(), size=(num_frames, height, width)
    )
    masks = masks.permute(0, 2, 3, 4, 1)
    masks = masks.expand(-1, -1, -1, -1, num_chns)
    return masks


class VideoMaePrtrnModule(ltn.LightningModule):
    """Lightning module of a VideoMAE model for pre-training."""

    def __init__(
        self,
        use_pretrained_checkpoint: bool = False,
        mask_factory: Callable[
            [VideoMAEForPreTraining], torch.Tensor
        ] = videomae_mask_factory(),
        optimizer_factory: Callable[
            [torch.nn.Module], torch.optim.Optimizer
        ] = adamw_factory,
        scheduler_factory: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ] = constant_scheduler,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters()  # NOTE:
        _path = "MCG-NJU/videomae-base"
        # _path = "MCG-NJU/videomae-base-ssv2"
        self.videomae = (
            VideoMAEForPreTraining.from_pretrained(_path)
            if use_pretrained_checkpoint
            else VideoMAEForPreTraining(VideoMAEConfig.from_pretrained(_path))
        ).train()
        self.mask_factory = mask_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_factory(self.videomae)
        scheduler = self.scheduler_factory(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch: INPUT_LABEL_BATCH, batch_idx: int) -> STEP_OUTPUT:
        inputs, _ = batch
        masks = torch.stack(
            [self.mask_factory(self.videomae) for _ in range(inputs.size(0))]
        ).to(inputs.device)
        outputs = self.videomae(pixel_values=inputs, bool_masked_pos=masks)
        loss = outputs.loss
        self.log("Loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: INPUT_LABEL_BATCH, batch_idx: int) -> STEP_OUTPUT:
        inputs, _ = batch
        masks = torch.stack(
            [self.mask_factory(self.videomae) for _ in range(inputs.size(0))]
        ).to(inputs.device)
        outputs = self.videomae(pixel_values=inputs, bool_masked_pos=masks)
        loss = outputs.loss
        self.log("Loss/val", loss, prog_bar=True)

    def predict_step(
        self, batch: INPUT_LABEL_BATCH, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, _ = batch
        masks = torch.stack(
            [self.mask_factory(self.videomae) for _ in range(inputs.size(0))]
        ).to(inputs.device)
        outputs = self.videomae(pixel_values=inputs, bool_masked_pos=masks)
        reconstructed = logits_to_pixels(outputs.logits)
        return reconstructed, masks


def load_pretrained_into_classifier(
    prtrn: VideoMaePrtrnModule, clsfr: VideoMaeClsfrModule
) -> None:
    """Load the model of a pre-training module into a classifier module."""
    clsfr.videomae.videomae.load_state_dict(
        prtrn.videomae.videomae.state_dict(), strict=True
    )


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
