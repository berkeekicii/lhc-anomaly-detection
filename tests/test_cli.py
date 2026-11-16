"""Run the commands of the README one after the other on fake data."""

import json

from lhcad import compare, evaluate, explore, prepare, selfcheck, synthetic, train


def test_the_whole_pipeline_from_the_command_line(tmp_path, capsys):
    raw, processed, runs = tmp_path / "raw", tmp_path / "processed", tmp_path / "runs"

    assert synthetic.main(["--out-dir", str(raw), "--n-background", "12000", "--n-signal", "1500"]) == 0
    assert prepare.main(["--raw-dir", str(raw), "--out-dir", str(processed)]) == 0
    assert train.main(["--processed-dir", str(processed), "--runs-dir", str(runs), "--name", "cli",
                       "--epochs", "2", "--batch-size", "256"]) == 0
    for name in ("model.pt", "scaler.npz", "history.json", "train_config.json", "loss.png"):
        assert (runs / "cli" / name).exists(), name

    capsys.readouterr()  # forget what the earlier commands printed
    # no --run given: the newest run directory is picked up
    assert evaluate.main(["--runs-dir", str(runs), "--processed-dir", str(processed)]) == 0
    metrics = json.loads((runs / "cli" / "metrics.json").read_text())
    assert len(metrics["signals"]) == 4
    output = capsys.readouterr().out
    assert "TPR@1e-05" in output and "A -> 4 leptons" in output

    # a second run with other settings, then both side by side
    assert train.main(["--processed-dir", str(processed), "--runs-dir", str(runs), "--name", "raw",
                       "--epochs", "1", "--scaling", "none", "--hidden", "16", "--no-batch-norm"]) == 0
    assert evaluate.main(["--run", str(runs / "raw"), "--processed-dir", str(processed)]) == 0
    capsys.readouterr()
    assert compare.main(["--runs-dir", str(runs), "--fpr", "1e-3"]) == 0
    table = capsys.readouterr().out.strip().splitlines()
    assert len(table) == 4, table  # header, separator, two runs
    assert table[2].startswith("| cli | quantity | no | 2 (best ") and "| 32-16-3 | 5,122 |" in table[2]
    # 57->16->3->16->57 with biases: 928 + 51 + 64 + 969 = 2,012
    assert table[3].startswith("| raw | none | no | 1 (best 1) |") and "| 16-3 | 2,012 |" in table[3]


def test_selfcheck(capsys):
    assert selfcheck.main([]) == 0
    assert "Self-check passed" in capsys.readouterr().out


def test_explore(raw_dir, tmp_path, capsys):
    path = raw_dir / "synthetic_Ato4l.h5"
    assert explore.main([str(path), "--events", "1000", "--figures-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "FORMAT CHECK: OK" in output and "events in file: 2,000" in output
    assert (tmp_path / "explore_synthetic_Ato4l.png").stat().st_size > 0


def test_missing_data_gives_a_clear_error_instead_of_a_traceback(tmp_path, capsys):
    assert prepare.main(["--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 1
    assert "python -m lhcad.download" in capsys.readouterr().err
    assert train.main(["--processed-dir", str(tmp_path)]) == 1
    assert evaluate.main(["--runs-dir", str(tmp_path / "nothing")]) == 1
    assert compare.main(["--runs-dir", str(tmp_path / "nothing")]) == 1


def test_export_and_figures(tmp_path):
    import base64
    import io
    import tarfile

    import numpy as np
    import pytest

    from lhcad import export, figures

    raw, processed, runs, results = (tmp_path / n for n in ("raw", "processed", "runs", "results"))
    assert synthetic.main(["--out-dir", str(raw), "--n-background", "12000", "--n-signal", "1500"]) == 0
    assert prepare.main(["--raw-dir", str(raw), "--out-dir", str(processed)]) == 0
    assert train.main(["--processed-dir", str(processed), "--runs-dir", str(runs), "--name", "r",
                       "--epochs", "1"]) == 0
    assert export.main(["--run", str(runs / "r")]) == 1, "not evaluated yet"
    assert evaluate.main(["--run", str(runs / "r"), "--processed-dir", str(processed)]) == 0
    assert export.main(["--run", str(runs / "r"), "--processed-dir", str(processed),
                        "--raw-dir", str(raw), "--results-dir", str(results)]) == 0
    folder = results / "r"
    curves = json.loads((folder / "curves.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    # the reduced ROC curve passes exactly through the metric at FPR 1e-3
    n_test = metrics["background_test_events"]
    for name, curve in curves["roc"].items():
        fpr, tpr = np.asarray(curve["fpr"]), np.asarray(curve["tpr"])
        index = int(np.flatnonzero(np.isclose(fpr, int(1e-3 * n_test) / n_test))[0])
        assert tpr[index] == pytest.approx(metrics["signals"][name]["tpr_at_fpr"]["0.001"])
    assert set(curves["inputs"]["files"]) == {"background", "Ato4l", "leptoquark", "hToTauTau", "hChToTauNu"}

    assert figures.main([str(folder)]) == 0
    for name in ("loss.png", "roc.png", "scores.png", "inputs.png"):
        assert (folder / name).stat().st_size > 0

    text = export.archive_text(folder)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(text)), mode="r:gz") as tar:
        assert "model.pt" in tar.getnames() and "curves.json" in tar.getnames()
