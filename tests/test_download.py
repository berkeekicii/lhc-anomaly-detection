import functools
import hashlib
import http.server
import threading

import pytest

from lhcad.download import download_file, md5sum, parse_zenodo_record

CONTENT = b"not really an HDF5 file" * 5_000
CONTENT_MD5 = hashlib.md5(CONTENT).hexdigest()


@pytest.fixture()
def server(tmp_path):
    """A web server on this computer that serves one file, standing in for Zenodo."""
    served = tmp_path / "served"
    served.mkdir()
    (served / "file.h5").write_bytes(CONTENT)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                            functools.partial(QuietHandler, directory=str(served)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.h5"
    httpd.shutdown()


def test_md5sum(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(CONTENT)
    assert md5sum(path) == CONTENT_MD5


def test_download_and_verify(server, tmp_path):
    dest = tmp_path / "out" / "file.h5"
    assert download_file(server, dest, expected_md5=CONTENT_MD5, quiet=True) == dest
    assert dest.read_bytes() == CONTENT
    assert not dest.with_name("file.h5.part").exists()


def test_wrong_checksum_leaves_no_file_behind(server, tmp_path):
    dest = tmp_path / "file.h5"
    with pytest.raises(IOError, match="does not match"):
        download_file(server, dest, expected_md5="0" * 32, quiet=True)
    assert not dest.exists() and not dest.with_name("file.h5.part").exists()


def test_complete_file_is_not_downloaded_again(tmp_path):
    dest = tmp_path / "file.h5"
    dest.write_bytes(CONTENT)
    # the address does not exist: this only passes if no download is attempted
    download_file("http://127.0.0.1:9/none", dest, expected_md5=CONTENT_MD5, quiet=True)


def test_damaged_file_is_replaced(server, tmp_path):
    dest = tmp_path / "file.h5"
    dest.write_bytes(CONTENT[:100])
    download_file(server, dest, expected_md5=CONTENT_MD5, quiet=True)
    assert dest.read_bytes() == CONTENT


def test_parse_zenodo_record():
    record = {"id": 7152617, "files": [
        {"key": "hChToTauNu_13TeV_PU20_filtered.h5", "size": 65_900_000,
         "checksum": "md5:5e01840e4669632769951f51c483e344",
         "links": {"self": "https://zenodo.org/api/records/7152617/files/hChToTauNu_13TeV_PU20_filtered.h5/content"}},
        {"key": "no_links.h5", "size": 1, "checksum": "sha256:abc"},
    ]}
    first, second = parse_zenodo_record(record)
    assert first == {"name": "hChToTauNu_13TeV_PU20_filtered.h5", "size": 65_900_000,
                     "md5": "5e01840e4669632769951f51c483e344",
                     "urls": ["https://zenodo.org/api/records/7152617/files/hChToTauNu_13TeV_PU20_filtered.h5/content",
                              "https://zenodo.org/records/7152617/files/hChToTauNu_13TeV_PU20_filtered.h5?download=1"]}
    assert second["md5"] is None
    assert second["urls"] == ["https://zenodo.org/records/7152617/files/no_links.h5?download=1"]


def test_content_link_is_preferred_over_self_link():
    record = {"id": 1, "files": [{"key": "a.h5", "links": {"self": "https://x/api/files/a.h5",
                                                           "content": "https://x/api/files/a.h5/content"}}]}
    assert parse_zenodo_record(record)[0]["urls"] == [
        "https://x/api/files/a.h5/content", "https://x/api/files/a.h5",
        "https://zenodo.org/records/1/files/a.h5?download=1"]


def test_next_address_is_tried_when_the_first_one_is_wrong(server, tmp_path, monkeypatch):
    """Zenodo is replaced by the local server: the first address returns the wrong content."""
    import lhcad.download as download

    wrong = server.replace("file.h5", "other.bin")
    (tmp_path / "served" / "other.bin").write_bytes(b"this is not the file")
    monkeypatch.setattr(download, "fetch_json", lambda url: {"id": 5046428, "files": [
        {"key": "background_for_training.h5", "checksum": f"md5:{CONTENT_MD5}",
         "links": {"content": wrong, "self": server}}]})
    path = download.download_dataset("background", tmp_path / "raw")
    assert path.name == "background_for_training.h5" and path.read_bytes() == CONTENT
