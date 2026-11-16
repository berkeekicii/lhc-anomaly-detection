"""Fake events in the ADC2021 file format.

Purpose: check that the code and your installation work BEFORE downloading
800 MB of real data, and feed the automated tests. The numbers are random and
have nothing to do with physics. Never report results obtained on them.

Usage (from the project root):

    python -m lhcad.synthetic        # writes five small files to data/synthetic/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .constants import BACKGROUND, CLASS_INDEX, N_OBJECTS, OBJECT_SLOTS, PARTICLES_KEY, SIGNALS

# (object, |eta| limit, minimum pT) as in the dataset paper, section "Dataset".
OBJECTS = (("electron", 3.0, 3.0), ("muon", 2.1, 3.0), ("jet", 4.0, 15.0))
TRIGGER_PT = 23.0  # every event has an electron or a muon above this pT

# How the invented "processes" differ from each other. Arbitrary numbers.
PROFILES = {
    BACKGROUND:   {"met": 25.0, "electron": 0.45, "muon": 0.45, "jet": 2.0, "lepton_pt": 15.0, "jet_pt": 25.0},
    "Ato4l":      {"met": 20.0, "electron": 1.80, "muon": 1.80, "jet": 1.5, "lepton_pt": 20.0, "jet_pt": 25.0},
    "leptoquark": {"met": 50.0, "electron": 0.50, "muon": 0.50, "jet": 4.5, "lepton_pt": 18.0, "jet_pt": 70.0},
    "hToTauTau":  {"met": 40.0, "electron": 0.60, "muon": 0.60, "jet": 2.5, "lepton_pt": 22.0, "jet_pt": 30.0},
    "hChToTauNu": {"met": 80.0, "electron": 0.50, "muon": 0.50, "jet": 2.5, "lepton_pt": 18.0, "jet_pt": 30.0},
}


def make_events(n_events: int, kind: str = BACKGROUND, seed: int = 0) -> np.ndarray:
    """Return an (n_events, 19, 4) array laid out like the real "Particles" dataset."""
    profile = PROFILES[kind]
    rng = np.random.default_rng(seed)
    events = np.zeros((n_events, N_OBJECTS, 4), dtype=np.float64)

    # One hidden number per event, an overall "energy", multiplies every pT and
    # raises the number of jets. It gives the fake data some structure for an
    # autoencoder to find (real events are correlated through their physics).
    energy = rng.lognormal(mean=0.0, sigma=0.5, size=n_events)

    # slot 0: missing transverse energy (it has no eta)
    events[:, 0, 0] = rng.exponential(profile["met"], n_events) * energy
    events[:, 0, 2] = rng.uniform(-np.pi, np.pi, n_events)
    events[:, 0, 3] = CLASS_INDEX["MET"]

    for name, eta_max, pt_min in OBJECTS:
        first, last = OBJECT_SLOTS[name]
        slots = last - first
        mean_count = profile[name] * energy if name == "jet" else profile[name]
        count = np.minimum(rng.poisson(mean_count, n_events), slots)
        present = np.arange(slots)[None, :] < count[:, None]
        scale = profile["jet_pt"] if name == "jet" else profile["lepton_pt"]
        pt = pt_min + rng.exponential(scale, (n_events, slots)) * energy[:, None]
        pt = -np.sort(-pt, axis=1)  # highest first
        events[:, first:last, 0] = pt * present
        events[:, first:last, 1] = rng.uniform(-eta_max, eta_max, (n_events, slots)) * present
        events[:, first:last, 2] = rng.uniform(-np.pi, np.pi, (n_events, slots)) * present
        events[:, first:last, 3] = CLASS_INDEX[name] * present

    # Events without a lepton above the trigger threshold get one in the leading slot.
    e_slot, mu_slot = OBJECT_SLOTS["electron"][0], OBJECT_SLOTS["muon"][0]
    failing = np.flatnonzero((events[:, e_slot, 0] <= TRIGGER_PT) & (events[:, mu_slot, 0] <= TRIGGER_PT))
    use_muon = rng.random(len(failing)) < 0.5
    for slot, name, eta_max, chosen in ((e_slot, "electron", 3.0, failing[~use_muon]),
                                        (mu_slot, "muon", 2.1, failing[use_muon])):
        events[chosen, slot, 0] = TRIGGER_PT + rng.exponential(profile["lepton_pt"], len(chosen)) * energy[chosen]
        events[chosen, slot, 1] = rng.uniform(-eta_max, eta_max, len(chosen))
        events[chosen, slot, 2] = rng.uniform(-np.pi, np.pi, len(chosen))
        events[chosen, slot, 3] = CLASS_INDEX[name]
    return events


def write_file(path, events: np.ndarray) -> None:
    """Write events to an HDF5 file with the three datasets of a real ADC2021 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(PARTICLES_KEY, data=events, compression="gzip")
        handle.create_dataset("Particles_Classes",
                              data=np.array([b"MET_class_1", b"Four_Ele_class_2",
                                             b"Four_Mu_class_3", b"Ten_Jet_class_4"]))
        handle.create_dataset("Particles_Names", data=np.array([b"Pt", b"Eta", b"Phi", b"Class"]))


def write_dataset(out_dir, n_background: int = 300_000, n_signal: int = 20_000, seed: int = 0) -> dict:
    """Write one background and four signal files; return {name: path}."""
    out_dir = Path(out_dir)
    paths = {}
    for index, name in enumerate((BACKGROUND, *SIGNALS)):
        n_events = n_background if name == BACKGROUND else n_signal
        paths[name] = out_dir / f"synthetic_{name}.h5"
        write_file(paths[name], make_events(n_events, kind=name, seed=seed + index))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write fake ADC2021-format files for testing.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-background", type=int, default=300_000)
    parser.add_argument("--n-signal", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for name, path in write_dataset(args.out_dir, args.n_background, args.n_signal, args.seed).items():
        print(f"wrote {path}  ({name})")
    print("These files are random numbers for testing only, not physics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
