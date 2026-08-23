import hashlib
import io
import json

import pytest

from chimorse import datasets

# ----------------------------------------------------------------------

class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _metadata_bytes():
    metadata = {
        "molecule": {
            "id": "PA",
            "screw_step_deg": 20,
            "isolated_helix_energy_ev": -1.0,
        },
        "interactions": ["EP"],
        "files": {"EP": "E_all_EP.dat"},
    }
    return json.dumps(metadata).encode()

# ----------------------------------------------------------------------

def test_ensure_reference_data_uses_existing_complete_dataset(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "PA"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_bytes(_metadata_bytes())
    (dataset_dir / "E_all_EP.dat").write_text("0\t0\t0\t9\t-2\n")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network access should not occur")

    monkeypatch.setattr(datasets, "_open_url", fail_if_called)

    result = datasets.ensure_reference_data("PA", data_root=tmp_path)
    assert result == dataset_dir


def test_download_zenodo_record_selects_files_and_checks_checksum(tmp_path, monkeypatch):
    data_bytes = b"0\t0\t0\t9\t-2\n"
    metadata_bytes = _metadata_bytes()

    record = {
        "files": [
            {
                "key": "metadata.json",
                "checksum": "md5:" + hashlib.md5(metadata_bytes).hexdigest(),
                "links": {"self": "https://example.test/metadata.json"},
            },
            {
                "key": "E_all_EP.dat",
                "checksum": "md5:" + hashlib.md5(data_bytes).hexdigest(),
                "links": {"self": "https://example.test/E_all_EP.dat"},
            },
            {
                "key": "notes.txt",
                "links": {"self": "https://example.test/notes.txt"},
            },
        ]
    }

    responses = {
        "https://zenodo.org/api/records/21904448": json.dumps(record).encode(),
        "https://example.test/metadata.json": metadata_bytes,
        "https://example.test/E_all_EP.dat": data_bytes,
    }

    def fake_open(url, timeout):
        return _Response(responses[url])

    monkeypatch.setattr(datasets, "_open_url", fake_open)

    result = datasets.download_zenodo_record(
        "10.5281/zenodo.21904448",
        tmp_path,
        file_globs=("*.dat", "metadata.json"),
    )

    assert result == tmp_path
    assert (tmp_path / "metadata.json").read_bytes() == metadata_bytes
    assert (tmp_path / "E_all_EP.dat").read_bytes() == data_bytes
    assert not (tmp_path / "notes.txt").exists()

# ----------------------------------------------------------------------

def test_download_zenodo_record_rejects_bad_checksum(tmp_path, monkeypatch):
    record = {
        "files": [
            {
                "key": "bad.dat",
                "checksum": "md5:" + "0" * 32,
                "links": {"self": "https://example.test/bad.dat"},
            }
        ]
    }

    responses = {
        "https://zenodo.org/api/records/21904448": json.dumps(record).encode(),
        "https://example.test/bad.dat": b"not-the-expected-content",
    }

    monkeypatch.setattr(
        datasets,
        "_open_url",
        lambda url, timeout: _Response(responses[url]),
    )

    with pytest.raises(IOError, match="Checksum mismatch"):
        datasets.download_zenodo_record(
            "21904448",
            tmp_path,
            file_globs=("*.dat",),
        )

    assert not (tmp_path / "bad.dat").exists()
    assert not (tmp_path / "bad.dat.part").exists()
