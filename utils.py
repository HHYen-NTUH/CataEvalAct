"""Utilities."""

import pathlib
from glob import glob
import itertools
import re


def glob_filepaths(queries: list[str]) -> list[pathlib.Path]:
    """Return filepaths obtained via a sequence of glob query strings."""
    return sorted(
        map(
            pathlib.Path,
            itertools.chain.from_iterable(glob(query) for query in queries),
        )
    )


def video_id_from_filepath(videopath: pathlib.Path) -> str:
    """Return video ID given the file path."""
    match = re.match(r"video(\d+)", videopath.stem)
    if match is None:
        raise ValueError("video file path format not recognized")
    else:
        return match.group(1)


def filepath_from_video_frame_id(
    video_id: str, frame_id: str, dirpath: pathlib.Path
) -> pathlib.Path:
    """Return file path of a video frame given its video and frame ID."""
    return dirpath / f"{video_id}_{frame_id}.jpg"


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
