"""Evaluation prediction of action classification."""

import argparse
import pathlib
from typing import Literal
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from parser import query_parser
from annotation import (
    Action,
    ACTION_FROM_TEXT,
    action_group_mapping,
    action_group_text_mapping,
    action_group_label_mapping,
    action_groups_from_arguments,
)
from schema import CLIPPRED_COMMONCOLS, FRAMEPRED_COMMONCOLS
from metric import classification_metrics, classification_metrics_classwise


DTYPE = {
    "clip": {
        "video_id": "object",
        "clip_id": "int64",
        "clip_frame_id": "object",
        "frame_ids": "object",
        "label": "object",
    },
    "frame": {
        "video_id": "object",
        "frame_id": "object",
        "label": "object",
    },
}


def read_prediction(
    predpath: pathlib.Path, mode: Literal["clip", "frame"] = "clip"
) -> pd.DataFrame:
    """Return prediction saved at the file path."""
    return pd.read_csv(predpath, dtype=DTYPE[mode])


def combine_prediction(
    prediction: list[pd.DataFrame],
    action_groups: list[Action],
    mode: Literal["clip", "frame"] = "clip",
) -> pd.DataFrame:
    """
    Return combined prediction onto given actions, where predicted probabilities of
    missing actions are filled by 0.
    """
    match mode:
        case "frame":
            commoncols = FRAMEPRED_COMMONCOLS
        case _:  # clip
            commoncols = CLIPPRED_COMMONCOLS
    for pred in prediction:
        assert all(col in pred.columns for col in commoncols)
    action_group_text = action_group_text_mapping(action_groups)
    columns = commoncols + [action_group_text[group] for group in action_groups]
    return pd.concat(
        [pred.reindex(columns=columns, fill_value=0) for pred in prediction],
        axis=0,
        ignore_index=True,
    )


def main(
    predpath: list[pathlib.Path],
    action_groups: list[Action],
    mode: Literal["clip", "frame"],
    metricpath: pathlib.Path,
    confmatpath: pathlib.Path,
) -> None:
    """Save evaluation metrics and confusion matrix."""
    action_group_text = action_group_text_mapping(action_groups)
    group_of_action = action_group_mapping(action_groups)
    action_group_index = action_group_label_mapping(action_groups)
    prediction = combine_prediction(
        [read_prediction(_predpath, mode) for _predpath in predpath],
        action_groups,
        mode,
    )
    preds = prediction.loc[
        :, [action_group_text[group] for group in action_groups]
    ].to_numpy()
    trgts = np.array(
        [
            action_group_index[group_of_action[ACTION_FROM_TEXT[text]]]
            for text in prediction.loc[:, "label"]
        ]
    )
    # Compute evaluation metrics
    _preds, _trgts = torch.tensor(preds), torch.tensor(trgts)
    average_metrics = classification_metrics(num_classes=len(action_groups))
    _average_metrics = average_metrics(_preds, _trgts)
    classwise_metrics = classification_metrics_classwise(
        classes=[action_group_text[group] for group in action_groups]
    )
    _classwise_metrics = classwise_metrics(_preds, _trgts)
    metrics = pd.concat(
        [
            pd.DataFrame(
                {"Class": "Overall"}
                | {name: [value.item()] for name, value in _average_metrics.items()}
            ),
            *[
                pd.DataFrame(
                    {"Class": action_group_text[group]}
                    | {
                        name: [
                            _classwise_metrics[
                                f"{name}/{action_group_text[group]}"
                            ].item()
                        ]
                        for name in _average_metrics.keys()
                    }
                )
                for group in action_groups
            ],
        ],
        axis=0,
        ignore_index=True,
    )
    # Compute confusion matrix
    confmat, ax = plt.subplots(
        figsize=(3 + 0.75 * len(action_groups), 3 + 0.75 * len(action_groups))
    )
    ConfusionMatrixDisplay.from_predictions(
        trgts,
        preds.argmax(axis=1),
        labels=range(len(action_groups)),
        display_labels=[action_group_text[group] for group in action_groups],
        colorbar=False,
        cmap="Blues",
        ax=ax,
        xticks_rotation="vertical",
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Annotation")
    confmat.tight_layout()
    # Display evaluation metrics
    print(metrics)
    # Save evaluation metrics and confusion matrix
    metrics.round(decimals=4).to_csv(metricpath, index=False)
    confmat.savefig(confmatpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[query_parser()])
    parser.add_argument("--predpath", type=str, required=True, nargs="+")
    parser.add_argument("--mode", choices=["clip", "frame"], default="clip")
    parser.add_argument("--metricpath", type=str, required=True)
    parser.add_argument("--confmatpath", type=str, required=True)
    args = parser.parse_args()
    pathlib.Path(args.metricpath).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.confmatpath).parent.mkdir(parents=True, exist_ok=True)
    main(
        predpath=[pathlib.Path(predpath) for predpath in args.predpath],
        action_groups=action_groups_from_arguments(args.action, args.action_group),
        mode=args.mode,
        metricpath=pathlib.Path(args.metricpath),
        confmatpath=pathlib.Path(args.confmatpath),
    )
