"""Generate prediction of a VideoMAE model."""

import argparse
import functools
import pathlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from parser import (
    frame_dataset_parser,
    query_parser,
    clip_dataset_parser,
    split_parser,
    compute_parser,
)
from schema import (
    read_frame_metadata,
    read_segment_metadata,
    read_splitting,
    CLIPPRED_COMMONCOLS,
)
from annotation import (
    Phase,
    Action,
    ActionScore,
    action_group_text_mapping,
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
from transform import TransformWraper, default_transform
from videomae import VideoMaeClsfrModule


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
    ckptpath: pathlib.Path,
    batchsize: int,
    num_workers: int,
    device: str,
    device_indices: list[int] | None,
    predpath: pathlib.Path,
) -> None:
    """Save prediction of a VideoMAE model."""
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
    testset = constant_rate_clip_action_dataset(
        framedir,
        query_by_split_group(framemeta, split, SplitGroup.TEST),
        query_by_split_group(sgmtmeta, split, SplitGroup.TEST),
        action_groups=action_groups,
        clipsize=clipsize,
        clipframediff=cliplength / clipsize,
        clipstepdiff=clipstep,
        smplfreq=smplfreq,
        reprframerange=clipfocalrange,
        sgmtquery=sgmtquery,
        enforce_single_action_clips=exclude_mixed_action_clips,
    )
    # Instantiate data loader
    testloader = DataLoader(
        TransformWraper(testset, default_transform()),
        batch_size=batchsize,
        num_workers=num_workers,
        shuffle=False,
    )
    # Instantiate and load the model
    model = VideoMaeClsfrModule.load_from_checkpoint(
        ckptpath,
        map_location="cpu",
        strict=False,  # to ignore criterion
    )
    # Generate prediction
    runner = Trainer(
        accelerator=device,
        devices=device_indices if device_indices is not None else "auto",
        logger=False,
    )
    _predprobs = runner.predict(model, testloader)
    predprobs: list[list[float]] = (
        torch.cat([probs.cpu() for probs in _predprobs], dim=0).numpy().tolist()
        if _predprobs is not None
        else []
    )
    # Save prediction
    action_group_text = action_group_text_mapping(action_groups)
    prediction = []
    for (_, clipgrp), probs in zip(testset.clipgroup, predprobs):
        video_id = clipgrp["video_id"].iloc[0]
        clip_id = clipgrp["clip_id"].iloc[0]
        clip_frame_id = clipgrp["clip_frame_id"].iloc[0]
        frame_ids = "_".join(clipgrp["frame_id"])
        segment_id = testset.framemeta.loc[(video_id, clip_frame_id), "segment_id"]
        label = testset.sgmtmeta.loc[(video_id, segment_id), "action"]
        prediction.append([video_id, clip_id, clip_frame_id, frame_ids, label, *probs])
    pd.DataFrame(
        prediction,
        columns=(
            CLIPPRED_COMMONCOLS + [action_group_text[group] for group in action_groups]
        ),
    ).to_csv(predpath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[
            frame_dataset_parser(),
            query_parser(),
            clip_dataset_parser(),
            split_parser(),
            compute_parser(),
        ]
    )
    parser.add_argument("--ckptpath", type=str, required=True)
    parser.add_argument("--predpath", type=str, required=True)
    args = parser.parse_args()
    pathlib.Path(args.predpath).parent.mkdir(parents=True, exist_ok=True)
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
        ckptpath=pathlib.Path(args.ckptpath),
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        device=args.device,
        device_indices=args.device_indices,
        predpath=pathlib.Path(args.predpath),
    )
