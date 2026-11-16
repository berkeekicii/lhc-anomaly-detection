"""Shared test data: small fake files in the ADC2021 format (see lhcad/synthetic.py)."""

import pytest

from lhcad.prepare import prepare
from lhcad.synthetic import write_dataset

N_BACKGROUND = 20_000
N_SIGNAL = 2_000


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw")
    write_dataset(path, n_background=N_BACKGROUND, n_signal=N_SIGNAL, seed=1)
    return path


@pytest.fixture(scope="session")
def processed_dir(raw_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("processed")
    prepare(raw_dir, path, max_events=None, seed=0, quiet=True)
    return path
