"""Train a VideoMAE model."""

import argparse
from typing import Literal
import pathlib
import functools
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
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
from annotation import (
    Phase,
    Action,
    ActionScore,
    phases_from_texts,
    action_groups_from_arguments,
    scores_from_texts,
)
from query import (
    query_by_split_group,
    SplitGroup,
    query_by_phase,
    query_by_performance,
    query_chaining,
)
from dataset import constant_rate_clip_action_dataset
from transform import TransformWraper, default_transform, default_augment_transform
from videomae import (
    VideoMaeClsfrModule,
    layerwise_lr_decay_paramgroups,
    VideoMaePrtrnModule,
    load_pretrained_into_classifier,
)
from criterion import balanced_cross_entropy_loss, FocalCrossEntropyLoss
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
    losstype: Literal[
        "cross_entropy_loss",
        "balanced_focal_cross_entropy_loss",
        "focal_loss_gamma_2",
        "focal_loss_gamma_4",
    ],
    batchsize: int,
    num_workers: int,
    prtrnckptpath: pathlib.Path | None,
    device: str,
    device_indices: list[int] | None,
    num_epochs: int,
    num_accml_batches: int,
    logging_interval: int,
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
        TransformWraper(trainset, default_augment_transform()),
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
    lr = 5e-4 * _batchsize / 256
    # lr = 5e-3 * _batchsize / 256
    minlr = 1e-6 * _batchsize / 256
    lr_decay = 0.75
    weight_decay = 0.05
    match losstype:
        case "focal_loss_gamma_4":
            criterion = FocalCrossEntropyLoss(gamma=4)
        case "focal_loss_gamma_2":
            criterion = FocalCrossEntropyLoss(gamma=2)
        case "balanced_focal_cross_entropy_loss":
            criterion = balanced_cross_entropy_loss(
                trainset, num_classes=len(action_groups)
            )
        case _:
            criterion = torch.nn.CrossEntropyLoss()
    optimizer_factory = functools.partial(
        adamw_factory,
        paramgroups_factory=functools.partial(
            layerwise_lr_decay_paramgroups, baselr=lr, decay=lr_decay
        ),
        weight_decay=weight_decay,
    )
    scheduler_factory = functools.partial(
        warmup_consine_decay_scheduler,
        num_warmup_epochs=num_warmup_epochs,
        num_epochs=num_epochs,
        warmuplr=warmuplr,
        lr=lr,
        minlr=minlr,
    )
    use_pretrained_checkpoint = prtrnckptpath is None
    model = VideoMaeClsfrModule(
        num_classes=len(action_groups),
        use_pretrained_checkpoint=use_pretrained_checkpoint,
        criterion=criterion,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )
    # Load pre-trained checkpoint
    if prtrnckptpath is not None:
        load_pretrained_into_classifier(
            VideoMaePrtrnModule.load_from_checkpoint(
                prtrnckptpath, map_location="cpu", strict=True
            ),
            model,
        )
    # Instantiate callbacks
    logger = MLFlowLogger(
        tracking_uri=mlflow_server_uri,
        experiment_name=mlflow_expname,
        run_name=mlflow_runname,
        tags={"Description": mlflow_rundesc},
    )
    # ckptmetric = "Loss/val"
    # negate_ckptmetric = True
    # ckptmetric = "AveragePrecision/val"
    # negate_ckptmetric = False
    ckptmetric = "Accuracy/val"
    negate_ckptmetric = False
    ckptcb = ModelCheckpoint(
        dirpath=ckptdir,
        filename=selected_ckptname,
        save_weights_only=True,
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
        log_every_n_steps=logging_interval,
        logger=logger,
        callbacks=callbacks,
    )
    runner.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    # Log information about the selected checkpoint
    logger.log_hyperparams(
        {
            "best_ckptpath": ckptcb.best_model_path,
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
    parser.add_argument("--prtrnckptpath", type=str, default=None)
    parser.add_argument(
        "--losstype",
        choices=[
            "cross_entropy_loss",
            "balanced_cross_entropy_loss",
            "focal_loss_gamma_2",
            "focal_loss_gamma_4",
        ],
        default="cross_entropy_loss",
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
        losstype=args.losstype,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        prtrnckptpath=(
            pathlib.Path(args.prtrnckptpath) if args.prtrnckptpath is not None else None
        ),
        device=args.device,
        device_indices=args.device_indices,
        num_epochs=args.num_epochs,
        num_accml_batches=args.num_accml_batches,
        logging_interval=args.logging_interval,
        mlflow_server_uri=args.mlflow_server_uri,
        mlflow_expname=args.mlflow_expname,
        mlflow_runname=args.mlflow_runname,
        mlflow_rundesc=args.mlflow_rundesc,
        ckptdir=pathlib.Path(args.ckptdir),
        selected_ckptname=args.ckptname,
    )
