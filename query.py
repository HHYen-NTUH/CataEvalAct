"""Utilities to query metadata."""

from collections.abc import Callable
import enum
import functools
import pandas as pd
from annotation import (
    Action,
    actions_from_groups,
    TEXT_FROM_ACTION,
    Phase,
    TEXT_FROM_PHASE,
    ActionScore,
    NUMBER_FROM_SCORE,
)
from schema import SPLITCOL, TRAINGRP, VALGRP, TESTGRP, VIDEO_ID


def query_by_action(
    sgmtmeta: pd.DataFrame, action_groups: list[Action]
) -> pd.DataFrame:
    """Return segment metadata of the actions."""
    return sgmtmeta.loc[
        sgmtmeta["action"].isin(
            [TEXT_FROM_ACTION[action] for action in actions_from_groups(action_groups)]
        ),
        :,
    ]


def query_by_phase(sgmtmeta: pd.DataFrame, phases: Phase) -> pd.DataFrame:
    """Return segment metadata of the phases."""
    return sgmtmeta.loc[
        sgmtmeta["phase"].isin([TEXT_FROM_PHASE[phase] for phase in phases]), :
    ]


def query_by_performance(
    sgmtmeta: pd.DataFrame,
    scores: ActionScore,
    include_idle: bool = True,
    include_oof: bool = False,
) -> pd.DataFrame:
    """Return segment metadata of actions of specific performance scores."""
    queried = sgmtmeta["performance"].isin(
        [NUMBER_FROM_SCORE[scr] for scr in scores if scr is not ActionScore.NAN]
    )
    if ActionScore.NAN in scores:
        queried = queried | (
            sgmtmeta["performance"].isna() & (sgmtmeta["action"] != "Idle")
        )
    if include_idle:  # handle Idle action separately since it only has NaN performance
        queried = queried | (sgmtmeta["action"] == "Idle")
    if include_oof:  # OOF segments may be regarded as low quality
        queried = queried | (sgmtmeta["out_of_frame"])
    return sgmtmeta.loc[queried, :]


def exclude_oof_segments(sgmtmeta: pd.DataFrame) -> pd.DataFrame:
    """Return segment metadata where out-of-frame segments are excluded."""
    return sgmtmeta.loc[~sgmtmeta["out_of_frame"], :]


def query_frames_of_segments(
    framemeta: pd.DataFrame, sgmtmeta: pd.DataFrame
) -> pd.DataFrame:
    """Return frame metadata of segments."""
    return framemeta.merge(
        sgmtmeta[["video_id", "segment_id"]], on=["video_id", "segment_id"]
    )


def query_clips_of_segments(
    clipmeta: pd.DataFrame, sgmtmeta: pd.DataFrame
) -> pd.DataFrame:
    """Return clip metadata of segments."""
    return clipmeta.merge(
        (
            clipmeta.query("frame_id == clip_frame_id")
            .merge(
                sgmtmeta.loc[:, ["video_id", "segment_id"]],
                on=["video_id", "segment_id"],
            )
            .loc[:, ["video_id", "clip_id"]]
        ),
        on=["video_id", "clip_id"],
    )


def query_single_action_clips(
    clipmeta: pd.DataFrame, sgmtmeta: pd.DataFrame
) -> pd.DataFrame:
    """Return clip metadata of those containing a single action."""
    clips_single_action = (
        clipmeta.merge(
            sgmtmeta.loc[:, ["video_id", "segment_id", "action"]],
            on=["video_id", "segment_id"],
        )
        .groupby(["video_id", "clip_frame_id"])["action"]
        .nunique()
        .reset_index()
        .rename(columns={"action": "num_actions"})
        .query("num_actions == 1")
        .loc[:, ["video_id", "clip_frame_id"]]
    )
    clipmeta = clipmeta.merge(clips_single_action, on=["video_id", "clip_frame_id"])
    return clipmeta


def query_chaining(
    *queries: Callable[[pd.DataFrame], pd.DataFrame],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return chaining of multiple metadata queries."""
    return lambda meta: functools.reduce(lambda v, f: f(v), queries, meta)


class SplitGroup(enum.Enum):
    """Group of dataset splitting."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def query_by_split_group(
    metadata: pd.DataFrame, split: pd.DataFrame, group: SplitGroup
) -> pd.DataFrame:
    """Return metadata of the splitting group."""
    match group:
        case SplitGroup.TRAIN:
            cases = split.loc[split[SPLITCOL] == TRAINGRP, VIDEO_ID]
        case SplitGroup.VAL:
            cases = split.loc[split[SPLITCOL] == VALGRP, VIDEO_ID]
        case SplitGroup.TEST:
            cases = split.loc[split[SPLITCOL] == TESTGRP, VIDEO_ID]
    return metadata.loc[metadata[VIDEO_ID].isin(cases), :]


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
