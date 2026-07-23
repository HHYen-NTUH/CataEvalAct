"""Utilities of evaluation metrics."""

from typing import Literal
import torchmetrics as tm
from torchmetrics.wrappers import ClasswiseWrapper


def classification_metrics(
    num_classes: int, mode: Literal["multiclass", "multilabel"] = "multiclass"
) -> tm.MetricCollection:
    match mode:
        case "multilabel":
            kwargs = {
                "task": "multilabel",
                "num_labels": num_classes,
                "average": "macro",
            }
        case _:  # multiclass
            kwargs = {
                "task": "multiclass",
                "num_classes": num_classes,
                "average": "macro",
            }
    return tm.MetricCollection(
        {
            "Accuracy": tm.Accuracy(**kwargs | {"average": "micro"}),
            "Recall": tm.Recall(**kwargs),
            "Precision": tm.Precision(**kwargs),
            "F1Score": tm.F1Score(**kwargs),
            "AUROC": tm.AUROC(**kwargs),
            "AveragePrecision": tm.AveragePrecision(**kwargs),
        }
    )


def classification_metrics_classwise(classes: list[str]) -> tm.MetricCollection:
    num_classes = len(classes)
    return tm.MetricCollection(
        {
            "Accuracy": ClasswiseWrapper(
                tm.Accuracy(task="multiclass", num_classes=num_classes, average="none"),
                labels=classes,
                prefix="Accuracy/",
            ),
            "Recall": ClasswiseWrapper(
                tm.Recall(task="multiclass", num_classes=num_classes, average="none"),
                labels=classes,
                prefix="Recall/",
            ),
            "Precision": ClasswiseWrapper(
                tm.Precision(
                    task="multiclass", num_classes=num_classes, average="none"
                ),
                labels=classes,
                prefix="Precision/",
            ),
            "F1Score": ClasswiseWrapper(
                tm.F1Score(task="multiclass", num_classes=num_classes, average="none"),
                labels=classes,
                prefix="F1Score/",
            ),
            "AUROC": ClasswiseWrapper(
                tm.AUROC(task="multiclass", num_classes=num_classes, average="none"),
                labels=classes,
                prefix="AUROC/",
            ),
            "AveragePrecision": ClasswiseWrapper(
                tm.AveragePrecision(
                    task="multiclass", num_classes=num_classes, average="none"
                ),
                labels=classes,
                prefix="AveragePrecision/",
            ),
        }
    )


def main() -> None:
    """Empty main."""


if __name__ == "__main__":
    main()
