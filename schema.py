"""Utilities of data frame schema."""

from collections import namedtuple
import pathlib
import pandas as pd


VIDEO_ID = "video_id"
SGMT_ID = "segment_id"
FRAME_ID = "frame_id"


SgmtMetaRow = namedtuple(
    "SgmtMetaRow",
    [
        VIDEO_ID,
        SGMT_ID,
        "start",
        "end",
        "phase",
        "action",
        "performance",
        "out_of_frame",
        "chunk_id",
    ],
)
FrameMetaRow = namedtuple("FrameMetaRow", [VIDEO_ID, FRAME_ID, "timestamp", SGMT_ID])


def read_segment_metadata(filepath: pathlib.Path) -> pd.DataFrame:
    """Return metadata of video segments."""
    return pd.read_csv(
        filepath,
        dtype={
            VIDEO_ID: "object",
            SGMT_ID: "int64",
        },
    )


def read_frame_metadata(filepath: pathlib.Path) -> pd.DataFrame:
    """Return metadata of frames."""
    return pd.read_csv(
        filepath,
        dtype={
            VIDEO_ID: "object",
            FRAME_ID: "object",
            "timestamp": "float64",
            SGMT_ID: "int64",
        },
    )


SPLITCOL = "split"
TRAINGRP = "training"
VALGRP = "validation"
TESTGRP = "testing"


def read_splitting(filepath: pathlib.Path) -> pd.DataFrame:
    """Return dataset splitting."""
    return pd.read_csv(filepath, dtype={VIDEO_ID: "object"})


CLIPPRED_COMMONCOLS = ["video_id", "clip_id", "clip_frame_id", "frame_ids", "label"]
FRAMEPRED_COMMONCOLS = ["video_id", "frame_id", "label"]


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
