"""Reading ADC2021 files and turning events into model inputs.

One event is stored as a (19, 4) table: 19 object slots times (pT, eta, phi,
particle type). The model sees the first three columns flattened into 57
numbers, exactly as in the organisers' example code
(https://github.com/mpp-hep/ADC2021-examplecode, create_datasets.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .constants import DATASETS, INPUT_DIM, N_KINEMATIC, N_OBJECTS, PARTICLES_KEY

DEFAULT_CHUNK = 200_000  # events read from disk at a time (about 46 MB as float32)


# --------------------------------------------------------------------------- files

def _particles(handle: h5py.File) -> h5py.Dataset:
    """Return the "Particles" dataset after checking that it looks right."""
    if PARTICLES_KEY not in handle:
        raise KeyError(f"no '{PARTICLES_KEY}' dataset in {handle.filename}; "
                       f"found {list(handle.keys())}. Is this an ADC2021 file?")
    dataset = handle[PARTICLES_KEY]
    if dataset.ndim != 3 or dataset.shape[1] != N_OBJECTS or dataset.shape[2] < N_KINEMATIC:
        raise ValueError(f"'{PARTICLES_KEY}' in {handle.filename} has shape {dataset.shape}; "
                         f"expected (N, {N_OBJECTS}, 4)")
    return dataset


def count_events(path) -> int:
    """Number of events in an ADC2021 file."""
    with h5py.File(path, "r") as handle:
        return int(_particles(handle).shape[0])


def describe_file(path) -> dict:
    """Names, shapes and types of everything stored in an HDF5 file."""
    description = {"path": str(path), "datasets": {}}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            dataset = handle[key]
            entry = {"shape": tuple(int(n) for n in dataset.shape), "dtype": str(dataset.dtype)}
            if dataset.size <= 16:  # the two small label datasets
                entry["values"] = [v.decode() if isinstance(v, bytes) else str(v)
                                   for v in np.ravel(dataset[()])]
            description["datasets"][key] = entry
    return description


def find_raw_files(raw_dir) -> dict[str, Path]:
    """Map dataset names ("background", "Ato4l", ...) to the files in raw_dir.

    Uses manifest.json written by lhcad.download when it exists, and otherwise
    recognises files by their names, so files downloaded by hand work too.
    """
    raw_dir = Path(raw_dir)
    found: dict[str, Path] = {}
    manifest = raw_dir / "manifest.json"
    if manifest.exists():
        for name, filename in json.loads(manifest.read_text()).items():
            if name in DATASETS and (raw_dir / filename).exists():
                found[name] = raw_dir / filename
    candidates = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    for path in candidates:
        lower = path.name.lower()
        if "blackbox" in lower:  # the challenge's unlabelled test file, not used here
            continue
        for name, info in DATASETS.items():
            if name not in found and info["hint"] in lower:
                found[name] = path
    return found


# -------------------------------------------------------------------------- events

def iter_events(path, chunk_size: int = DEFAULT_CHUNK):
    """Yield (start_index, block) pairs; block has shape (n, 57), dtype float32.

    Reading in blocks keeps memory use small even though the background file
    holds millions of events.
    """
    with h5py.File(path, "r") as handle:
        dataset = _particles(handle)
        n_events = dataset.shape[0]
        for start in range(0, n_events, chunk_size):
            block = dataset[start:start + chunk_size, :, :N_KINEMATIC]
            yield start, block.astype(np.float32).reshape(len(block), INPUT_DIM)


def load_events(path, max_events: int | None = None, seed: int = 0,
                chunk_size: int = DEFAULT_CHUNK) -> np.ndarray:
    """Load events as an (n, 57) float32 array.

    If the file holds more than max_events, a random subset is taken (the same
    one every time for a given seed), in file order.
    """
    n_total = count_events(path)
    if max_events is None or max_events >= n_total:
        keep, n_keep = None, n_total
    else:
        rng = np.random.default_rng(seed)
        keep = np.zeros(n_total, dtype=bool)
        keep[rng.choice(n_total, size=max_events, replace=False)] = True
        n_keep = max_events

    events = np.empty((n_keep, INPUT_DIM), dtype=np.float32)
    filled = 0
    for start, block in iter_events(path, chunk_size):
        if keep is not None:
            block = block[keep[start:start + len(block)]]
        events[filled:filled + len(block)] = block
        filled += len(block)
    if filled != n_keep:
        raise RuntimeError(f"read {filled} events from {path}, expected {n_keep}")
    return events


def split_background(events: np.ndarray, fractions=(0.5, 0.1, 0.4), seed: int = 0):
    """Shuffle events IN PLACE and return (train, validation, test) views.

    The default 50% / 10% / 40% split is the one used by Govorkova et al.
    (arXiv:2108.03986). The test part is large on purpose: the challenge metric
    looks at the 1-in-100,000 most anomalous background events.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must be three numbers that sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    rng.shuffle(events, axis=0)
    n_train = int(round(fractions[0] * len(events)))
    n_val = int(round(fractions[1] * len(events)))
    return events[:n_train], events[n_train:n_train + n_val], events[n_train + n_val:]


# ------------------------------------------------------------------- input scaling

SCALINGS = ("quantity", "feature", "none")


class Scaler:
    """Bring the inputs to a similar numerical size before they enter the network.

    pT is measured in GeV and reaches hundreds, eta and phi stay within a few
    units. Without scaling, the loss would be decided almost entirely by pT.
    Three ways of scaling are available, and comparing them is an experiment
    worth doing:

    "quantity" (default)
        One divisor for all pT values, one for all eta, one for all phi: the
        root mean square of that quantity over all 19 slots. Nothing is
        shifted, so an empty slot stays exactly (0, 0, 0), and a 40 GeV
        electron weighs the same whether it sits in the first or in the fourth
        electron slot.
    "feature"
        Classic standardisation: each of the 57 inputs gets its own mean and
        standard deviation. Slots that are almost always empty (a 4th electron)
        then have a tiny standard deviation, so the rare events that fill them
        produce very large inputs and dominate the loss.
    "none"
        Raw values, as in the organisers' example notebook.

    The numbers are measured on the training events only and then reused,
    unchanged, for validation, test and signal events.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, mode: str = "feature"):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mode = mode

    @classmethod
    def fit(cls, events: np.ndarray, mode: str = "quantity",
            chunk_size: int = DEFAULT_CHUNK) -> "Scaler":
        if mode not in SCALINGS:
            raise ValueError(f"unknown scaling {mode!r}; choose one of {SCALINGS}")
        n_features = events.shape[1]
        if mode == "none":
            return cls(np.zeros(n_features), np.ones(n_features), mode)

        # Sums are accumulated block by block in float64: precise, and no
        # second copy of the whole training set is needed.
        total = np.zeros(n_features, dtype=np.float64)
        total_sq = np.zeros(n_features, dtype=np.float64)
        for start in range(0, len(events), chunk_size):
            block = np.asarray(events[start:start + chunk_size], dtype=np.float64)
            total += block.sum(axis=0)
            total_sq += np.square(block).sum(axis=0)

        if mode == "feature":
            mean = total / len(events)
            std = np.sqrt(np.maximum(total_sq / len(events) - np.square(mean), 0.0))
        else:  # "quantity"
            if n_features != INPUT_DIM:
                raise ValueError(f"'quantity' scaling needs {INPUT_DIM} features, got {n_features}")
            mean = np.zeros(n_features)
            mean_square = (total_sq / len(events)).reshape(N_OBJECTS, N_KINEMATIC).mean(axis=0)
            std = np.tile(np.sqrt(mean_square), N_OBJECTS)  # (pT, eta, phi) repeated per slot
        # A feature that never changes (the MET eta is always 0) would give a
        # division by zero; dividing by 1 leaves it unchanged instead.
        std[std < 1e-6] = 1.0
        return cls(mean, std, mode)

    def transform(self, events: np.ndarray) -> np.ndarray:
        out = np.asarray(events, dtype=np.float32) - self.mean
        out /= self.std
        return out

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, std=self.std, mode=np.array(self.mode))

    @classmethod
    def load(cls, path) -> "Scaler":
        with np.load(path) as stored:
            return cls(stored["mean"], stored["std"], str(stored["mode"]))
