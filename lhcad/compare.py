"""Put the results of all evaluated runs side by side.

Usage (from the project root):

    python -m lhcad.compare              # every run in runs/ that has been evaluated
    python -m lhcad.compare --fpr 1e-4   # show the TPR at another false-positive rate

Prints a Markdown table, ready to paste into README.md or a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import CHALLENGE_FPR, DATASETS, SIGNALS


def collect(runs_dir) -> list[dict]:
    """Read metrics.json and train_config.json of every evaluated run, oldest first."""
    rows = []
    for metrics_path in sorted(Path(runs_dir).glob("*/metrics.json"), key=lambda p: p.stat().st_mtime):
        metrics = json.loads(metrics_path.read_text())
        config_path = metrics_path.with_name("train_config.json")
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        rows.append({"run": metrics["run"], "metrics": metrics, "config": config})
    return rows


def markdown_table(rows: list[dict], fpr: float = CHALLENGE_FPR) -> str:
    key = f"{fpr:g}"
    signals = [name for name in SIGNALS if any(name in row["metrics"]["signals"] for row in rows)]
    header = ["run", "scaling", "layers", "parameters"]
    header += [f"{DATASETS[name]['label']}: AUC / TPR@{key}" for name in signals]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        config, metrics = row["config"], row["metrics"]
        layers = "-".join(str(n) for n in (*config.get("hidden", []), config.get("latent_dim", "?")))
        cells = [row["run"], str(metrics.get("scaling", "?")), layers, f"{metrics['parameters']:,}"]
        for name in signals:
            entry = metrics["signals"].get(name)
            if entry is None:
                cells.append("-")
                continue
            tpr = entry["tpr_at_fpr"].get(key)
            cells.append(f"{entry['auc']:.3f} / " + (f"{100 * tpr:.3f}%" if tpr is not None else "n/a"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare evaluated runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--fpr", type=float, default=CHALLENGE_FPR)
    args = parser.parse_args(argv)
    rows = collect(args.runs_dir)
    if not rows:
        print(f"ERROR: no evaluated runs in {args.runs_dir}/ . Run 'python -m lhcad.evaluate' first.",
              file=sys.stderr)
        return 1
    print(markdown_table(rows, args.fpr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
