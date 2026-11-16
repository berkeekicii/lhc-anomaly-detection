# Anomaly detection on LHC collision data

Unsupervised detection of unusual proton-proton collision events with a small
autoencoder, on the public [ADC2021](https://mpp-hep.github.io/ADC2021/) dataset.

**Status: work in progress (started November 2026).** The data pipeline, a
baseline model, the evaluation and the tests are in place, and a first
baseline has been trained and evaluated on the real dataset; see
[Results](#results).

Author: Berke Ekici, Computer Engineering, Mersin University.

## The problem

The LHC collides protons 40 million times per second. Almost all collisions
have to be thrown away within microseconds by the trigger system, and
whatever the trigger does not select is lost. A trigger that only looks for
what theory predicts cannot find the unexpected. The idea studied here: train
a model on ordinary events only, and keep the events it finds hard to
reconstruct.

The ADC2021 challenge provides a simulated stream of Standard Model events
for training and four simulated "new physics" signals for testing. Each event
is described by up to 19 reconstructed objects (missing transverse energy,
4 electrons, 4 muons, 10 jets) with three numbers each (pT, eta, phi), so a
model sees 57 numbers per event. The challenge metric is the true-positive
rate at a false-positive rate of 1e-5.

## Dataset

All files are hosted on Zenodo under CC-BY-4.0.

| Sample | Size | Events in file | Zenodo record |
|---|---|---|---|
| Standard Model background (training) | 646.4 MB | 13,451,915 | [5046428](https://zenodo.org/records/5046428) |
| A -> 4 leptons | 5.4 MB | 55,969 | [7152590](https://zenodo.org/records/7152590) |
| LQ -> b tau | 28.6 MB | 340,544 | [7152599](https://zenodo.org/records/7152599) |
| h0 -> tau tau | 50.2 MB | 691,283 | [7152614](https://zenodo.org/records/7152614) |
| h+- -> tau nu | 65.9 MB | 760,272 | [7152617](https://zenodo.org/records/7152617) |

Event counts were read from the files with `python -m lhcad.explore`. By
default `python -m lhcad.prepare` uses a random subset of 4,000,000 background
events.

## Quick start

Python 3.11, 3.12 or 3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m lhcad.selfcheck            # 30 s, no download: is everything installed?
python -m pytest                     # the automated tests

python -m lhcad.download             # about 800 MB into data/raw/
python -m lhcad.explore data/raw/background_for_training.h5
python -m lhcad.prepare              # data/processed/*.npy (about 1.3 GB)
python -m lhcad.train --name baseline
python -m lhcad.evaluate --run runs/baseline
python -m lhcad.compare              # all evaluated runs as a Markdown table
```

If the download script fails on your network, download the five files from
the record pages in the table above into `data/raw/` and continue with
`python -m lhcad.prepare`.

No local setup: the notebook
[notebooks/reproduce_baseline.ipynb](notebooks/reproduce_baseline.ipynb) runs
the same steps on a free Google Colab machine.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeekicii/lhc-anomaly-detection/blob/main/notebooks/reproduce_baseline.ipynb)

Resources, measured on a 2-core machine with fake data of the same size as
the real dataset: preparing 4 million events takes about 10 s and 1.5 GB of
memory, training on 2 million events about 5 s per epoch and 2 GB, evaluation
about 5 s. No GPU is needed. With less than 8 GB of RAM use
`python -m lhcad.prepare --max-events 1000000`.

## How it works

| Step | Module | What it does |
|---|---|---|
| Download | `lhcad/download.py` | Fetches the files from Zenodo and verifies their checksums. |
| Explore | `lhcad/explore.py` | Prints the file structure, an example event and object statistics; checks the slot layout. |
| Prepare | `lhcad/prepare.py` | Reads the HDF5 files block by block, keeps (pT, eta, phi), shuffles and splits the background 50% / 10% / 40% into train / validation / test. |
| Scale | `lhcad/data.py` | `Scaler` brings the inputs to a similar size. Three options: `quantity` (default), `feature`, `none`. |
| Model | `lhcad/model.py` | Dense autoencoder 57 -> 32 -> 16 -> 3 -> 16 -> 32 -> 57 with batch normalisation and leaky ReLU, 5,122 parameters. |
| Train | `lhcad/train.py` | Adam, mean squared error, early stopping on the validation loss. |
| Evaluate | `lhcad/evaluate.py` | Anomaly score = reconstruction error. AUC and TPR at FPR 1e-3, 1e-4 and 1e-5 per signal; ROC and score plots. |
| Check | `lhcad/selfcheck.py`, `lhcad/synthetic.py` | Runs everything on fake events in the same file format. Fake events are for testing only. |

## Results

Published reference (Govorkova et al., Table I, floating-point DNN autoencoder,
reconstruction-error score):

| | A -> 4 leptons | LQ -> b tau | h0 -> tau tau | h+- -> tau nu |
|---|---|---|---|---|
| AUC | 0.96 | 0.95 | 0.87 | 0.96 |
| TPR at FPR 1e-5 | 3.56% | 0.05% | 0.09% | 0.06% |

This repository, first baseline (`python -m lhcad.train --name baseline`,
one run, seed 0; trained on 2,000,000 background events for 20 epochs,
tested on 1,600,000 background events; run on a free Google Colab CPU with
[the notebook](notebooks/reproduce_baseline.ipynb)):

| | A -> 4 leptons | LQ -> b tau | h0 -> tau tau | h+- -> tau nu |
|---|---|---|---|---|
| AUC | 0.855 | 0.843 | 0.750 | 0.894 |
| TPR at FPR 1e-3 | 1.660% | 0.714% | 2.125% | 6.032% |
| TPR at FPR 1e-4 | 0.180% | 0.031% | 0.273% | 1.010% |
| TPR at FPR 1e-5 | 0.020% | 0.002% | 0.033% | 0.098% |

At FPR 1e-5 only 16 background test events lie above the threshold, so the
last row is a rough estimate. The validation loss was still falling at
epoch 20, so longer training is the first thing to try.

The first baseline here is simpler than the reference: plain mean squared
error over all 57 inputs (the paper ignores empty slots and treats eta and phi
separately), one global scaling per quantity (the paper normalises with batch
normalisation), 20 epochs (the paper trains up to 100) and 4 million background
events (the paper uses 8 million). Lower numbers than the reference are
expected at first; measuring and closing that gap is the first part of the
project.

## Project layout

```
lhcad/            the Python package (one module per step, see the table above)
tests/            automated tests, run with "python -m pytest"
notebooks/        reproduce_baseline.ipynb: the whole pipeline on Google Colab
requirements.txt  the libraries the project needs
data/             created by the download and prepare steps (not in git)
runs/             one folder per trained model (not in git)
```

## Roadmap

1. Reproduce the baseline on the real dataset; repeat with several random seeds.
2. Study the effect of input scaling, latent size and a loss that ignores empty slots.
3. Shrink the model (quantization, pruning) and measure detection performance
   against inference latency on a CPU.
4. Serve the model behind a small API in a container, with latency measurements.
5. Apply the same method to a second domain (vehicle or electric-bus telemetry).

## Tests

`python -m pytest` runs 39 tests in about 10 seconds. They use small fake files
in the ADC2021 format, so they need neither the dataset nor a network connection.

## References

- E. Govorkova et al., "LHC physics dataset for unsupervised New Physics
  detection at 40 MHz", [arXiv:2107.02157](https://arxiv.org/abs/2107.02157).
  Data format, event selection and sample composition.
- E. Govorkova et al., "Autoencoders on FPGAs for real-time, unsupervised new
  physics detection at 40 MHz at the Large Hadron Collider",
  [arXiv:2108.03986](https://arxiv.org/abs/2108.03986),
  doi:10.1038/s42256-022-00441-3. Reference architecture and results.
- [ADC2021 challenge page](https://mpp-hep.github.io/ADC2021/) and
  [example code](https://github.com/mpp-hep/ADC2021-examplecode) by the organisers.

## License

The code in this repository is released under the MIT License, see
[LICENSE](LICENSE). The dataset is not part of this repository; it is
distributed by its authors on Zenodo under CC-BY-4.0.
