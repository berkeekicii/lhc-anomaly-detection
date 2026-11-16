"""Facts about the ADC2021 dataset that the rest of the code relies on.

Data format (Govorkova et al., "LHC physics dataset for unsupervised New Physics
detection at 40 MHz", arXiv:2107.02157, section "Data Records"):

* every file is HDF5 and holds three datasets: "Particles", "Particles_Classes"
  and "Particles_Names";
* "Particles" has shape (N, 19, 4): N events, 19 physics objects per event,
  4 numbers per object;
* the 19 object slots are fixed: slot 0 is the missing transverse energy (MET),
  slots 1-4 are electrons, slots 5-8 are muons, slots 9-18 are jets;
* the 4 numbers are pT, eta, phi and a particle-type index
  (1 = MET, 2 = electron, 3 = muon, 4 = jet, 0 = empty slot);
* events with fewer objects are zero padded.
"""

N_OBJECTS = 19          # MET + 4 electrons + 4 muons + 10 jets
N_KINEMATIC = 3         # pT, eta, phi (the 4th column is the particle-type index)
INPUT_DIM = N_OBJECTS * N_KINEMATIC   # 57 numbers per event go into the model

PARTICLES_KEY = "Particles"
FEATURE_NAMES = ("pT", "eta", "phi")

# Which object lives in which slot of the second axis.
OBJECT_SLOTS = {
    "MET": (0, 1),
    "electron": (1, 5),
    "muon": (5, 9),
    "jet": (9, 19),
}
CLASS_INDEX = {"empty": 0, "MET": 1, "electron": 2, "muon": 3, "jet": 4}

# The five public files of the challenge, all hosted on Zenodo (CC-BY-4.0).
#   record : Zenodo record id (https://zenodo.org/records/<record>)
#   md5    : checksum shown on the record page on 2026-11-15 (cross-check only;
#            the checksum reported by Zenodo at download time is authoritative)
#   hint   : text that appears in the file name, used to recognise the file
#   label  : human-readable name used in tables and plots
DATASETS = {
    "background": {
        "record": 5046428,
        "md5": "08cb4801229f9d480567f8d3f5cac134",
        "hint": "background",
        "label": "SM background",
    },
    "Ato4l": {
        "record": 7152590,
        "md5": "8735ed96a96c39dd13fde7d5f7c85967",
        "hint": "ato4l",
        "label": "A -> 4 leptons",
    },
    "leptoquark": {
        "record": 7152599,
        "md5": "ecd9750e5c51aef54918689eec1e7415",
        "hint": "leptoquark",
        "label": "LQ -> b tau",
    },
    "hToTauTau": {
        "record": 7152614,
        "md5": "b66e930ca0e5d5841ae9de43dea7f839",
        "hint": "htotautau",
        "label": "h0 -> tau tau",
    },
    "hChToTauNu": {
        "record": 7152617,
        "md5": "5e01840e4669632769951f51c483e344",
        "hint": "hchtotaunu",
        "label": "h+- -> tau nu",
    },
}

BACKGROUND = "background"
SIGNALS = tuple(name for name in DATASETS if name != BACKGROUND)

# The challenge metric: true-positive rate at this false-positive rate.
CHALLENGE_FPR = 1e-5
