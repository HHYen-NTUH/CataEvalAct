"""Pre-train a VideoMAE model."""

import argparse
import pathlib
import functools
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from mask import videomae_mask_factory
from parser import (
    frame_dataset_parser,
    query_parser,
    clip_dataset_parser,
    split_parser,
    compute_parser,
    train_parser,
)
from schema import (
    read_frame_metadata,
    read_segment_metadata,
    read_splitting,
)
from query import (
    query_by_phase,
    query_by_performance,
    query_chaining,
    query_by_split_group,
    SplitGroup,
)
from annotation import (
    Action,
    action_groups_from_arguments,
    Phase,
    phases_from_texts,
    ActionScore,
    scores_from_texts,
)
from dataset import constant_rate_clip_action_dataset
from transform import (
    TransformWraper,
    default_transform,
    default_pretrain_augment_transform,
)
from videomae import VideoMaePrtrnModule
from optimizer import adamw_factory
from scheduler import warmup_consine_decay_scheduler


def main(
    framedir: pathlib.Path,
    framemetapath: pathlib.Path,
    sgmtmetapath: pathlib.Path,
    smplfreq: float | None,
    phases: Phase,
    action_groups: list[Action],
    scores: ActionScore,
    exclude_idle: bool,
    clipsize: int,
    cliplength: float,
    clipstep: float,
    clipfocalrange: tuple[float, float] | None,
    exclude_mixed_action_clips: bool,
    splitpath: pathlib.Path,
    batchsize: int,
    num_workers: int,
    device: str,
    device_indices: list[int] | None,
    num_epochs: int,
    num_accml_batches: int,
    mlflow_server_uri: str,
    mlflow_expname: str,
    mlflow_runname: str,
    mlflow_rundesc: str,
    ckptdir: pathlib.Path,
    selected_ckptname: str,
) -> None:
    """Save MLflow logging and checkpoints of a VideoMAE model."""
    # Prepare dataset
    sgmtmeta = read_segment_metadata(sgmtmetapath)
    framemeta = read_frame_metadata(framemetapath)
    split = read_splitting(splitpath)
    sgmtqueries = [
        functools.partial(query_by_phase, phases=phases),
        functools.partial(
            query_by_performance,
            scores=scores,
            include_idle=not exclude_idle,
        ),
    ]
    sgmtquery = query_chaining(*sgmtqueries)
    trainset = constant_rate_clip_action_dataset(
        framedir,
        query_by_split_group(framemeta, split, SplitGroup.TRAIN),
        query_by_split_group(sgmtmeta, split, SplitGroup.TRAIN),
        action_groups=action_groups,
        clipsize=clipsize,
        clipframediff=cliplength / clipsize,
        clipstepdiff=clipstep,
        smplfreq=smplfreq,
        reprframerange=clipfocalrange,
        sgmtquery=sgmtquery,
        enforce_single_action_clips=exclude_mixed_action_clips,
    )
    valset = constant_rate_clip_action_dataset(
        framedir,
        query_by_split_group(framemeta, split, SplitGroup.VAL),
        query_by_split_group(sgmtmeta, split, SplitGroup.VAL),
        action_groups=action_groups,
        clipsize=clipsize,
        clipframediff=cliplength / clipsize,
        clipstepdiff=clipstep,
        smplfreq=smplfreq,
        reprframerange=clipfocalrange,
        sgmtquery=sgmtquery,
        enforce_single_action_clips=exclude_mixed_action_clips,
    )
    # Instantiate data loaders
    trainloader = DataLoader(
        TransformWraper(trainset, default_pretrain_augment_transform()),
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=True,
    )
    valloader = DataLoader(
        TransformWraper(valset, default_transform()),
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=False,
    )
    # Instantiate model
    num_warmup_epochs = 5
    assert num_epochs >= num_warmup_epochs
    _batchsize = batchsize * num_accml_batches
    warmuplr = 1e-6 * _batchsize / 256
    lr = 1.5e-4 * _batchsize / 256
    minlr = 1e-6 * _batchsize / 256
    weight_decay = 0.05
    betas = (0.9, 0.95)
    mask_ratio = 0.9
    mask_factory = videomae_mask_factory(mask_ratio=mask_ratio)
    optimizer_factory = functools.partial(
        adamw_factory,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )
    scheduler_factory = functools.partial(
        warmup_consine_decay_scheduler,
        num_warmup_epochs=num_warmup_epochs,
        num_epochs=num_epochs,
        warmuplr=warmuplr,
        lr=lr,
        minlr=minlr,
    )
    use_pretrained_checkpoint = True
    model = VideoMaePrtrnModule(
        use_pretrained_checkpoint=use_pretrained_checkpoint,
        mask_factory=mask_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    # Instantiate callbacks
    logger = MLFlowLogger(
        tracking_uri=mlflow_server_uri,
        experiment_name=mlflow_expname,
        run_name=mlflow_runname,
        tags={"Description": mlflow_rundesc},
    )
    ckptmetric = "Loss/val"
    negate_ckptmetric = True
    ckptcb = ModelCheckpoint(
        dirpath=ckptdir,
        filename=selected_ckptname,
        save_weights_only=True,
        save_last=True,
        monitor=ckptmetric,
        mode="min" if negate_ckptmetric else "max",
        enable_version_counter=False,
    )
    callbacks = [ckptcb, LearningRateMonitor(), TQDMProgressBar()]
    # Train the model
    runner = Trainer(
        accelerator=device,
        devices=device_indices if device_indices is not None else "auto",
        max_epochs=num_epochs,
        accumulate_grad_batches=num_accml_batches,
        logger=logger,
        callbacks=callbacks,
    )
    runner.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    # Log information about the selected checkpoint
    logger.log_hyperparams(
        {
            "best_ckptpath": ckptcb.best_model_path,
            "last_ckptpath": ckptcb.last_model_path,
            "best_ckptmetric_value": (
                ckptcb.best_model_score.item()
                if ckptcb.best_model_score is not None
                else None
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[
            frame_dataset_parser(),
            query_parser(),
            clip_dataset_parser(),
            split_parser(),
            compute_parser(),
            train_parser(),
        ]
    )
    args = parser.parse_args()
    pathlib.Path(args.ckptdir).mkdir(parents=True, exist_ok=True)
    main(
        framedir=pathlib.Path(args.framedir),
        framemetapath=pathlib.Path(args.framemetapath),
        sgmtmetapath=pathlib.Path(args.sgmtmetapath),
        smplfreq=args.smplfreq,
        phases=phases_from_texts(args.phase),
        action_groups=action_groups_from_arguments(args.action, args.action_group),
        scores=scores_from_texts(args.score),
        exclude_idle=args.exclude_idle,
        clipsize=args.clipsize,
        cliplength=args.cliplength,
        clipstep=args.clipstep,
        clipfocalrange=args.clipfocalrange,
        exclude_mixed_action_clips=args.exclude_mixed_action_clips,
        splitpath=pathlib.Path(args.splitpath),
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        device=args.device,
        device_indices=args.device_indices,
        num_epochs=args.num_epochs,
        num_accml_batches=args.num_accml_batches,
        mlflow_server_uri=args.mlflow_server_uri,
        mlflow_expname=args.mlflow_expname,
        mlflow_runname=args.mlflow_runname,
        mlflow_rundesc=args.mlflow_rundesc,
        ckptdir=pathlib.Path(args.ckptdir),
        selected_ckptname=args.ckptname,
    )
