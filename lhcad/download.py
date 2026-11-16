"""Download the ADC2021 files from Zenodo and check their integrity.

Usage (from the project root):

    python -m lhcad.download                 # everything (about 800 MB)
    python -m lhcad.download --only signals  # just the four small signal files

If this script does not work on your network, download the files by hand from
the record pages listed in README.md and put them into data/raw/ . The next
step (python -m lhcad.prepare) only needs the files to be there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from .constants import BACKGROUND, DATASETS, SIGNALS

ZENODO_API = "https://zenodo.org/api/records/{record}"
USER_AGENT = "lhcad/0.1 (student project; python-urllib)"
CHUNK = 1 << 20  # read and write 1 MiB at a time


def md5sum(path: Path) -> str:
    """Return the MD5 checksum of a file as a hex string."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str, timeout: float = 60.0) -> dict:
    """GET a URL and parse the response as JSON."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_zenodo_record(record: dict) -> list[dict]:
    """Turn a Zenodo API record into a list of {name, size, md5, urls} entries.

    "urls" lists the addresses the file can be downloaded from, best first:
    the links given by the API, then the address of the download button on
    the record page.
    """
    record_id = record.get("id") or record.get("recid")
    files = []
    for entry in record.get("files", []):
        name = entry.get("key") or entry.get("filename")
        if not name:
            continue
        checksum = entry.get("checksum") or ""
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
        links = entry.get("links") or {}
        urls = [links.get("content"), links.get("self"),
                f"https://zenodo.org/records/{record_id}/files/{name}?download=1"]
        urls = list(dict.fromkeys(url for url in urls if url))  # drop missing and repeated ones
        files.append({"name": name, "size": entry.get("size"), "md5": md5, "urls": urls})
    return files


def download_file(url: str, dest: Path, expected_md5: str | None = None,
                  timeout: float = 60.0, quiet: bool = False) -> Path:
    """Download url to dest. Skips the download if dest is already complete.

    The data is first written to "<dest>.part" and only renamed once the
    checksum is right, so an interrupted download never looks finished.
    """
    dest = Path(dest)
    if dest.exists():
        if expected_md5 is None or md5sum(dest) == expected_md5:
            if not quiet:
                print(f"  already there: {dest.name}")
            return dest
        if not quiet:
            print(f"  checksum mismatch, downloading again: {dest.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_print = 0.0
        while block := response.read(CHUNK):
            out.write(block)
            done += len(block)
            now = time.time()
            if not quiet and now - last_print > 1.0:
                last_print = now
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                          f"({100 * done / total:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {dest.name}: {done / 1e6:7.1f} MB", end="", flush=True)
    if not quiet:
        print(f"\r  {dest.name}: {done / 1e6:.1f} MB in {time.time() - started:.0f} s" + " " * 20)

    if expected_md5 is not None:
        actual = md5sum(part)
        if actual != expected_md5:
            part.unlink()
            raise IOError(f"{dest.name}: checksum {actual} does not match the expected "
                          f"{expected_md5}. The download was probably cut off; run again.")
    part.replace(dest)
    return dest


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download one of the five ADC2021 files (see constants.DATASETS)."""
    info = DATASETS[name]
    record = fetch_json(ZENODO_API.format(record=info["record"]))
    files = [f for f in parse_zenodo_record(record) if f["name"].lower().endswith((".h5", ".hdf5"))]
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one HDF5 file in Zenodo record {info['record']}, "
                           f"found {[f['name'] for f in files]}")
    entry = files[0]
    if entry["md5"] and entry["md5"] != info["md5"]:
        print(f"  note: Zenodo now reports checksum {entry['md5']} for {entry['name']}; "
              f"this project was written against {info['md5']} (the file may have been updated).")
    last_error: Exception | None = None
    for url in entry["urls"]:
        try:
            return download_file(url, Path(data_dir) / entry["name"],
                                 expected_md5=entry["md5"] or info["md5"])
        except OSError as error:  # wrong content or no connection: try the next address
            last_error = error
    raise RuntimeError(f"could not download {entry['name']}: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the ADC2021 files from Zenodo.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                        help="where to put the files (default: data/raw)")
    parser.add_argument("--only", choices=["all", "background", "signals"], default="all",
                        help="download only part of the dataset")
    args = parser.parse_args(argv)

    names = {"all": (BACKGROUND, *SIGNALS), "background": (BACKGROUND,), "signals": SIGNALS}[args.only]
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name in names:
        print(f"{name} ({DATASETS[name]['label']}), Zenodo record {DATASETS[name]['record']}")
        try:
            path = download_dataset(name, args.data_dir)
        except Exception as error:  # network problems should not hide the files we already have
            print(f"  FAILED: {error}", file=sys.stderr)
            print("  You can download this file by hand from "
                  f"https://zenodo.org/records/{DATASETS[name]['record']} into {args.data_dir}/",
                  file=sys.stderr)
            return 1
        manifest[name] = path.name
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Done. Files are in {args.data_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
