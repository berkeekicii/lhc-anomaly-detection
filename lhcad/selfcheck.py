"""Check the installation in about half a minute, without downloading anything.

Usage (from the project root):

    python -m lhcad.selfcheck

Runs the whole pipeline (prepare, train, evaluate) on a small set of fake
events in a temporary folder that is deleted afterwards. If the last line says
"Self-check passed", Python and all libraries are installed correctly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from . import evaluate, prepare, synthetic, train


def main(argv: list[str] | None = None) -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        raw, processed, runs = Path(folder) / "raw", Path(folder) / "processed", Path(folder) / "runs"
        print("1/4 writing fake events")
        synthetic.write_dataset(raw, n_background=60_000, n_signal=5_000)
        print("2/4 preparing arrays")
        prepare.prepare(raw, processed, max_events=None, quiet=True)
        print("3/4 training for 3 epochs")
        if train.main(["--processed-dir", str(processed), "--runs-dir", str(runs),
                       "--name", "selfcheck", "--epochs", "3"]) != 0:
            return 1
        print("4/4 evaluating")
        if evaluate.main(["--run", str(runs / "selfcheck"), "--processed-dir", str(processed)]) != 0:
            return 1
    print("\nSelf-check passed: the installation works.")
    print("The numbers above come from fake events and mean nothing. Next: python -m lhcad.download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
