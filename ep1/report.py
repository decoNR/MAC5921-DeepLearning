"""Reporting utilities for model comparison and visualization."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

# Use a writable directory for matplotlib cache in sandboxed or restricted environments.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curves(
    histories: Dict[str, Dict[str, List[float]]],
    out_path: Union[str, Path],
) -> None:
    """Plots loss and accuracy curves for multiple models on the same axes and saves to PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key().get(
        "color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )

    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])

        epochs_train = range(1, len(train_loss) + 1)
        epochs_val = range(1, len(val_loss) + 1)

        if train_loss:
            ax_loss.plot(
                epochs_train,
                train_loss,
                linestyle="--",
                color=color,
                label=f"{model_name} (Train)",
            )
        if val_loss:
            ax_loss.plot(
                epochs_val,
                val_loss,
                linestyle="-",
                color=color,
                label=f"{model_name} (Val)",
            )

        if train_acc:
            ax_acc.plot(
                epochs_train,
                train_acc,
                linestyle="--",
                color=color,
                label=f"{model_name} (Train)",
            )
        if val_acc:
            ax_acc.plot(
                epochs_val,
                val_acc,
                linestyle="-",
                color=color,
                label=f"{model_name} (Val)",
            )

    ax_loss.set_title("Training and Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend()

    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_comparison_table(summaries: Sequence[Dict[str, Any]]) -> str:
    """Formats summaries into a markdown-compatible text table with one row per model."""
    headers = [
        "Model",
        "Params",
        "Best Val Acc",
        "Best Epoch",
        "Epochs to Conv",
        "Time/Epoch (s)",
        "Total Time (s)",
        "Test Acc",
    ]

    rows = []
    for s in summaries:
        row = [
            str(s.get("model_name", "-")),
            str(s.get("num_params", "-")),
            f"{s.get('best_val_acc', 0.0):.4f}",
            str(s.get("best_val_epoch", "-")),
            str(s.get("epochs_to_convergence", "-")),
            f"{s.get('mean_epoch_time', 0.0):.2f}",
            f"{s.get('total_time', 0.0):.2f}",
            f"{s.get('test_acc', 0.0):.4f}",
        ]
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    sep_line = (
        "|-"
        + "-|-".join("-" * col_widths[i] for i in range(len(headers)))
        + "-|"
    )
    row_lines = [
        "| "
        + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
        + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + row_lines)


def print_comparison(
    summaries: Union[Dict[str, Dict[str, Any]], Sequence[Dict[str, Any]]],
    out_path: Optional[Union[str, Path]] = "results/summary.json",
) -> str:
    """Prints a textual comparison table of model summaries and saves summary.json."""
    if isinstance(summaries, dict):
        summary_list = list(summaries.values())
        save_data = summaries
    else:
        summary_list = list(summaries)
        save_data = summaries

    table = _format_comparison_table(summary_list)
    print(table)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

    return table
