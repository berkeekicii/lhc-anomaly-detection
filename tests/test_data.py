import json

import h5py
import numpy as np
import pytest

from lhcad.constants import CLASS_INDEX, INPUT_DIM, OBJECT_SLOTS, SIGNALS
from lhcad.data import (Scaler, count_events, describe_file, find_raw_files, iter_events,
                        load_events, split_background)
from lhcad.synthetic import TRIGGER_PT, make_events

from conftest import N_BACKGROUND, N_SIGNAL


def test_synthetic_events_follow_the_documented_layout():
    events = make_events(5_000, seed=3)
    assert events.shape == (5_000, 19, 4)
    assert np.all(events[:, 0, 3] == CLASS_INDEX["MET"])
    assert np.all(events[:, 0, 1] == 0), "MET has no eta"
    for name in ("electron", "muon", "jet"):
        first, last = OBJECT_SLOTS[name]
        group = events[:, first:last]
        assert set(np.unique(group[:, :, 3])) <= {0, CLASS_INDEX[name]}
        empty = group[:, :, 3] == 0
        assert np.all(group[empty] == 0), "empty slots are zero padded"
        assert np.all(np.diff(group[:, :, 0], axis=1) <= 0), "pT is sorted, highest first"
    leading_lepton = np.maximum(events[:, 1, 0], events[:, 5, 0])
    assert np.all(leading_lepton > TRIGGER_PT), "every event passes the lepton trigger"


def test_describe_and_count(raw_dir):
    path = raw_dir / "synthetic_background.h5"
    assert count_events(path) == N_BACKGROUND
    datasets = describe_file(path)["datasets"]
    assert datasets["Particles"]["shape"] == (N_BACKGROUND, 19, 4)
    assert datasets["Particles_Names"]["values"] == ["Pt", "Eta", "Phi", "Class"]


def test_files_are_recognised_by_name(tmp_path):
    names = ["background_for_training.h5", "Ato4l_lepFilter_13TeV_filtered.h5",
             "leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5", "hToTauTau_13TeV_PU20_filtered.h5",
             "hChToTauNu_13TeV_PU20_filtered.h5", "BlackBox_background_mix.h5"]
    for name in names:
        (tmp_path / name).touch()
    found = find_raw_files(tmp_path)
    assert {key: path.name for key, path in found.items()} == {
        "background": names[0], "Ato4l": names[1], "leptoquark": names[2],
        "hToTauTau": names[3], "hChToTauNu": names[4]}


def test_manifest_takes_priority_over_file_names(tmp_path):
    (tmp_path / "renamed.h5").touch()
    (tmp_path / "manifest.json").write_text(json.dumps({"background": "renamed.h5"}))
    assert find_raw_files(tmp_path)["background"].name == "renamed.h5"


def test_load_events_returns_the_first_three_columns_flattened(raw_dir):
    path = raw_dir / "synthetic_background.h5"
    events = load_events(path, chunk_size=3_000)  # chunk size that does not divide N
    with h5py.File(path, "r") as handle:
        expected = handle["Particles"][:, :, :3].astype(np.float32).reshape(-1, INPUT_DIM)
    assert events.dtype == np.float32
    np.testing.assert_array_equal(events, expected)


def test_iter_events_covers_every_event_once(raw_dir):
    starts, total = [], 0
    for start, block in iter_events(raw_dir / "synthetic_background.h5", chunk_size=7_000):
        starts.append(start)
        total += len(block)
        assert block.shape[1] == INPUT_DIM
    assert starts == [0, 7_000, 14_000] and total == N_BACKGROUND


def test_subset_is_reproducible_and_in_file_order(raw_dir):
    path = raw_dir / "synthetic_background.h5"
    everything = load_events(path)
    subset = load_events(path, max_events=5_000, seed=7)
    assert subset.shape == (5_000, INPUT_DIM)
    np.testing.assert_array_equal(subset, load_events(path, max_events=5_000, seed=7))
    assert not np.array_equal(subset, load_events(path, max_events=5_000, seed=8))
    # every selected event exists in the file, and the order of the file is kept
    lookup = {row.tobytes(): index for index, row in enumerate(everything)}
    positions = [lookup[row.tobytes()] for row in subset]
    assert positions == sorted(positions)


def test_split_sizes_and_content():
    events = np.arange(1_000 * 4, dtype=np.float32).reshape(1_000, 4)
    original = events.copy()
    train, val, test = split_background(events, fractions=(0.5, 0.1, 0.4), seed=0)
    assert (len(train), len(val), len(test)) == (500, 100, 400)
    together = np.concatenate([train, val, test])
    assert not np.array_equal(together, original), "events are shuffled"
    np.testing.assert_array_equal(together[np.argsort(together[:, 0])], original)


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_background(np.zeros((10, 4), dtype=np.float32), fractions=(0.5, 0.5, 0.5))


def test_prepare_writes_all_arrays(processed_dir):
    meta = json.loads((processed_dir / "meta.json").read_text())
    assert meta["background"] == {"file": "synthetic_background.h5", "events_in_file": N_BACKGROUND,
                                  "events_used": N_BACKGROUND, "train": 10_000, "val": 2_000,
                                  "test": 8_000}
    assert np.load(processed_dir / "background_train.npy").shape == (10_000, INPUT_DIM)
    for name in SIGNALS:
        assert np.load(processed_dir / f"signal_{name}.npy").shape == (N_SIGNAL, INPUT_DIM)


@pytest.fixture(scope="module")
def train_events(processed_dir):
    return np.load(processed_dir / "background_train.npy")


def test_feature_scaling_standardises_every_input(train_events):
    scaled = Scaler.fit(train_events, mode="feature", chunk_size=3_000).transform(train_events)
    assert scaled.dtype == np.float32 and np.all(np.isfinite(scaled))
    varying = train_events.std(axis=0) > 0
    np.testing.assert_allclose(scaled[:, varying].mean(axis=0), 0, atol=2e-3)
    np.testing.assert_allclose(scaled[:, varying].std(axis=0), 1, atol=2e-3)
    assert not varying[1], "feature 1 is the MET eta, which is always zero"
    assert np.all(scaled[:, ~varying] == 0), "constant features stay at zero instead of becoming NaN"


def test_quantity_scaling_uses_one_divisor_per_physical_quantity(train_events):
    scaler = Scaler.fit(train_events, mode="quantity", chunk_size=3_000)
    assert np.all(scaler.mean == 0), "nothing is shifted"
    per_slot = scaler.std.reshape(19, 3)
    assert np.all(per_slot == per_slot[0]), "every slot is divided by the same (pT, eta, phi) numbers"
    scaled = scaler.transform(train_events)
    assert np.all(scaled[train_events == 0] == 0), "empty slots stay exactly zero"
    rms = np.sqrt(np.mean(np.square(scaled.reshape(-1, 19, 3), dtype=np.float64), axis=(0, 1)))
    np.testing.assert_allclose(rms, 1, rtol=1e-3)


def test_no_scaling_and_unknown_scaling(train_events):
    np.testing.assert_array_equal(Scaler.fit(train_events, mode="none").transform(train_events[:100]),
                                  train_events[:100])
    with pytest.raises(ValueError, match="unknown scaling"):
        Scaler.fit(train_events, mode="magic")


@pytest.mark.parametrize("mode", ["quantity", "feature", "none"])
def test_scaler_survives_saving_and_loading(train_events, tmp_path, mode):
    scaler = Scaler.fit(train_events, mode=mode)
    scaler.save(tmp_path / "scaler.npz")
    reloaded = Scaler.load(tmp_path / "scaler.npz")
    assert reloaded.mode == mode
    np.testing.assert_array_equal(reloaded.transform(train_events[:100]), scaler.transform(train_events[:100]))
