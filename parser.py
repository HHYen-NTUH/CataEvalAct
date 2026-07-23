"""Utilities of command-line arugment parser."""

import argparse
from annotation import (
    PHASE_FROM_TEXT,
    COMPOSITE_PHASE_FROM_TEXT,
    ACTION_FROM_TEXT,
    COMPOSITE_ACTION_FROM_TEXT,
    SCORE_FROM_TEXT,
    COMPOSITE_SCORE_FROM_TEXT,
)


def frame_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--framedir", type=str, required=True)
    parser.add_argument("--framemetapath", type=str, required=True)
    parser.add_argument("--sgmtmetapath", type=str, required=True)
    parser.add_argument("--smplfreq", type=float, default=None)
    return parser


def query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase",
        choices=list(PHASE_FROM_TEXT.keys()) + list(COMPOSITE_PHASE_FROM_TEXT.keys()),
        nargs="+",
        default=["all"],
    )
    parser.add_argument(
        "--action",
        choices=list(ACTION_FROM_TEXT.keys()) + list(COMPOSITE_ACTION_FROM_TEXT.keys()),
        nargs="+",
        default=["All"],
    )
    parser.add_argument(
        "--action_group",
        choices=list(ACTION_FROM_TEXT.keys()),
        nargs="+",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--score",
        choices=list(SCORE_FROM_TEXT.keys()) + list(COMPOSITE_SCORE_FROM_TEXT.keys()),
        nargs="+",
        default=["all"],
    )
    parser.add_argument("--exclude_idle", default=False, action="store_true")
    parser.add_argument("--include_oof", default=False, action="store_true")
    return parser


def clip_dataset_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clipsize", type=int, default=16)
    parser.add_argument("--clipstepsize", type=int, default=None)
    parser.add_argument("--cliplength", type=float, default=1)
    parser.add_argument("--clipstep", type=float, default=1)
    parser.add_argument("--clipfocalrange", type=float, nargs=2, default=None)
    parser.add_argument(
        "--exclude_mixed_action_clips", default=False, action="store_true"
    )
    return parser


def split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--splitpath", type=str, required=True)
    return parser


def compute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_indices", type=int, default=None, nargs="+")
    return parser


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_accml_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients and step the optimizer",
    )
    parser.add_argument(
        "--logging_interval",
        type=int,
        default=50,
        help="Number of steps between consecutive logging",
    )
    parser.add_argument(
        "--mlflow_server_uri",
        type=str,
        required=True,
        help=(
            "URI of tracking server to store MLflow databases; if it is a local "
            "directory, it must have the 'file:' prefix"
        ),
    )
    parser.add_argument(
        "--mlflow_expname",
        type=str,
        required=True,
        help="Experiment name for MLflow logging",
    )
    parser.add_argument(
        "--mlflow_runname",
        type=str,
        required=True,
        help="Name of the experiemnt run for MLflow logging",
    )
    parser.add_argument(
        "--mlflow_rundesc",
        type=str,
        default=None,
        help="Description of the experiment run for MLflow logging",
    )
    parser.add_argument(
        "--ckptdir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--ckptname",
        type=str,
        required=True,
        help="File name of the selected model checkpoint",
    )
    return parser


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
