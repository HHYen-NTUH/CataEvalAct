"""Utilities of datasets."""

from collections.abc import Callable
import pathlib
import functools
import math
import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tvtrnsfrm
import torchvision.tv_tensors as tvtnsr
from annotation import (
    Action,
    action_group_label_mapping,
    action_group_mapping,
    actions_from_groups,
    TEXT_FROM_ACTION,
)
from utils import filepath_from_video_frame_id
from query import (
    query_by_action,
    query_frames_of_segments,
    query_clips_of_segments,
    query_single_action_clips,
)


class ClipActionDataset(Dataset):
    """Dataset of video clips and their action annotation."""

    def __init__(
        self,
        framedir: pathlib.Path,
        sgmtmeta: pd.DataFrame,
        framemeta: pd.DataFrame,
        clipmeta: pd.DataFrame,
        action_groups: list[Action],
    ) -> None:
        """
        Arguments
        ---------
        framedir: Directory containing images of frames
        sgmtmeta: Metadata of annotated segments
        framemeta: Metadata of frames
        clipmeta: Metadata of clips
        action_groups: Groups of actions to label the clips

        Notes
        -----
        - `sgmtmeta` has columns `video_id | segment_id | start | end | phase | action | performance | chunk_id`.
        - `framemeta` has columns `video_id | frame_id | timestamp | segment_id`.
        - `clipmeta` has columns `video_id | frame_id | timestamp | clip_id | clip_frame_id`.
        """
        super().__init__()
        self.framedir = framedir
        self.sgmtmeta = sgmtmeta.set_index(["video_id", "segment_id"])
        self.framemeta = framemeta.set_index(["video_id", "frame_id"])
        self.clipmeta = clipmeta
        self.clipgroup = self.clipmeta.groupby(["video_id", "clip_id"])
        self.clipgrpkeys = list(self.clipgroup.groups.keys())
        _action_group = action_group_mapping(action_groups)
        _action_group_label = action_group_label_mapping(action_groups)
        self.label_from_text = {
            TEXT_FROM_ACTION[action]: _action_group_label[_action_group[action]]
            for action in actions_from_groups(action_groups)
        }

    def __len__(self) -> int:
        return len(self.clipgrpkeys)

    def __getitem__(self, ind: int) -> tuple[tvtnsr.Video, int]:
        meta = self.clipgroup.get_group(self.clipgrpkeys[ind])
        if isinstance(meta, pd.Series):
            meta = meta.to_frame()
        video_id = meta["video_id"].iloc[0]
        clip_frame_id = meta["clip_frame_id"].iloc[0]
        sgmt_id = self.framemeta.loc[(video_id, clip_frame_id), "segment_id"]
        action_text = self.sgmtmeta.loc[(video_id, sgmt_id), "action"]
        clip = tvtnsr.Video(
            torch.stack(
                [
                    tvtrnsfrm.functional.to_image(
                        PIL.Image.open(
                            filepath_from_video_frame_id(
                                video_id, frame_id, self.framedir
                            )
                        )
                    )
                    for frame_id in meta["frame_id"]
                ],
                dim=0,
            )
        )
        label = self.label_from_text[action_text]
        return clip, label


def resample_frames(
    framemeta: pd.DataFrame, sgmtmeta: pd.DataFrame, smplfreq: float
) -> pd.DataFrame:
    """
    Return metadata of frames re-sampled at the given frequency.

    Note
    ----
    `framemeta` has columns `video_id | frame_id | timestamp | segment_id`.
    `sgmtmeta` at least has columns `video_id | segment_id | chunk_id`.
    """
    meta = framemeta.merge(
        sgmtmeta.loc[:, ["video_id", "segment_id", "chunk_id"]],
        on=["video_id", "segment_id"],
    ).sort_values(["video_id", "timestamp"])
    meta["_timestamp"] = pd.to_timedelta(meta.loc[:, "timestamp"], unit="s")
    meta = meta.set_index("_timestamp")
    sampled = (
        meta.groupby(by=["video_id", "chunk_id"])
        .resample(f"{1 / smplfreq}s", include_groups=False)
        .first()
        .reset_index()
    )
    return sampled.loc[:, framemeta.columns]


def clips_by_time_differences(
    frameactiongrp: pd.DataFrame,
    clipstepdiff: float,
    framediffs: list[float],
    reprframerange: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Return group of frame metadata of clips structured by time intervals between frames.

    Arguments
    ---------
    frameactiongrp: Group of frame metadata of columns `frame_id | timestamp | segment_id | action`
    clipstepdiff: Time difference between starting timestamps of consecutive clips
    framediffs: Time differences to the first frame of frames after it
    reprframerange: Range of time differences to the first frame in which frames can be
    selected to the representative frame; default to the whole clip

    Return
    ------
    Dataframe of columns `frame_id | timestamp | clip_id | clip_frame_id | segment_id`.

    Note
    ----
    The representative frame of a clip is deemed the first frame of the majority segment
    of the majority action within the clip.
    """
    clipsize = 1 + len(framediffs)
    ts = frameactiongrp["timestamp"].to_numpy()
    diffs = np.array([0] + framediffs)
    num_clips = math.floor((ts.max() - ts.min() - diffs.max()) / clipstepdiff) + 1
    if num_clips < 1:  # return empty dataframe if no clip is found in the segment
        return pd.DataFrame(
            [],
            columns=["frame_id", "timestamp", "clip_id", "clip_frame_id", "segment_id"],
        )
    # Find expected timestamps of frames in clips
    ts_clipstarts = np.arange(num_clips) * clipstepdiff + ts.min()
    ts_frames = (ts_clipstarts[:, None] + diffs).flatten()
    # Collect frames of timestamps closest to the expected values
    ind_frames = np.abs((ts[:, None] - ts_frames)).argmin(axis=0)
    clipgrp = frameactiongrp.iloc[ind_frames, :].copy()
    # Find representative frame of each clip
    clipgrp["clip_id"] = np.arange(num_clips).repeat(clipsize)
    if reprframerange is not None:
        assert diffs.max() >= reprframerange[0]
        assert diffs.min() < reprframerange[1]
    clipgrp["_is_cand"] = np.tile(
        (
            (diffs >= reprframerange[0]) & (diffs < reprframerange[1])
            if reprframerange is not None
            else np.ones(clipsize, dtype=bool)
        ),
        num_clips,
    )
    clipgrp = clipgrp.merge(
        (
            clipgrp.query("_is_cand == True")
            .groupby("clip_id")
            .apply(
                lambda grp: (
                    grp.merge(
                        grp.loc[:, "action"].value_counts().to_frame("action_count"),
                        on="action",
                    )
                    .merge(
                        grp.loc[:, "segment_id"]
                        .value_counts()
                        .to_frame("segment_count"),
                        on="segment_id",
                    )
                    .sort_values(
                        by=["action_count", "segment_count", "timestamp"],
                        ascending=[False, False, True],
                    )
                    .loc[:, ["frame_id"]]
                    .iloc[0]
                ),
                include_groups=False,
            )
            .reset_index()
            .rename(columns={"frame_id": "clip_frame_id"})
        ),
        on="clip_id",
    )
    return clipgrp.loc[
        :, ["frame_id", "timestamp", "clip_id", "clip_frame_id", "segment_id"]
    ]


def constant_rate_clip_action_dataset(
    framedir: pathlib.Path,
    framemeta: pd.DataFrame,
    sgmtmeta: pd.DataFrame,
    action_groups: list[Action],
    clipsize: int,
    clipframediff: float,
    clipstepdiff: float,
    smplfreq: float | None = None,
    reprframerange: tuple[float, float] | None = None,
    sgmtquery: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    enforce_single_action_clips: bool = False,
) -> ClipActionDataset:
    """Return clip dataset of a constant frame sampling rate."""
    framemeta = (
        resample_frames(framemeta, sgmtmeta, smplfreq)
        if smplfreq is not None
        else framemeta.sort_values(["video_id", "timestamp"])
    )
    clipmeta = (
        framemeta.merge(
            sgmtmeta.loc[:, ["video_id", "segment_id", "action", "chunk_id"]],
            on=["video_id", "segment_id"],
        )
        .groupby(["video_id", "chunk_id"])
        .apply(
            functools.partial(
                clips_by_time_differences,
                clipstepdiff=clipstepdiff,
                framediffs=[clipframediff * i for i in range(1, clipsize)],
                reprframerange=reprframerange,
            ),
            include_groups=False,
        )
        .reset_index(level=["video_id", "chunk_id"])
    )
    clipmeta["clip_id"] = (  # assign unique ID for each clip in each video
        clipmeta.loc[:, ["video_id", "chunk_id", "clip_id"]]
        .drop_duplicates()
        .groupby("video_id")
        .cumcount()
        .repeat(clipsize)
        .to_numpy()
    )
    # Query segments and clips
    if sgmtquery is not None:
        sgmtmeta = sgmtquery(sgmtmeta)
    sgmtmeta = query_by_action(sgmtmeta, action_groups)  # enforce focal actions
    framemeta = query_frames_of_segments(framemeta, sgmtmeta)
    clipmeta = query_clips_of_segments(clipmeta, sgmtmeta)
    if enforce_single_action_clips:
        clipmeta = query_single_action_clips(clipmeta, sgmtmeta)
    clipmeta = clipmeta.loc[
        :,
        ["video_id", "frame_id", "timestamp", "clip_id", "clip_frame_id"],
    ]
    return ClipActionDataset(framedir, sgmtmeta, framemeta, clipmeta, action_groups)


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
