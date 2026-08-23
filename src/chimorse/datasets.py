"""Download and validate external reference datasets used by CHIMORSE examples.

The scientific reference data are intentionally kept outside the Python package.
This module provides a small, explicit acquisition layer while ``chimorse.dataio``
remains responsible only for parsing local files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

# ----------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """Description of one externally archived CHIMORSE reference dataset."""

    molecule_id: str
    record_or_doi: str
    file_globs: tuple[str, ...] = ("*.dat", "metadata.json")


REFERENCE_DATASETS = {
    "PA": DatasetSpec(
        molecule_id="PA",
        record_or_doi="10.5281/zenodo.21904448",
    ),
}

# ----------------------------------------------------------------------

def _zenodo_record_id(record_or_doi: str) -> str:
    """Extract a Zenodo record id from a record id, DOI, or Zenodo DOI URL."""
    value = str(record_or_doi).strip().rstrip("/")
    if value.isdigit():
        return value

    marker = "zenodo."
    lower = value.lower()
    if marker in lower:
        candidate = value[lower.rfind(marker) + len(marker) :]
        candidate = candidate.split("/", 1)[0].split("?", 1)[0]
        if candidate.isdigit():
            return candidate

    raise ValueError(
        "Expected a Zenodo record id or DOI such as "
        "'21904448' or '10.5281/zenodo.21904448'."
    )

# ----------------------------------------------------------------------

def _open_url(url: str, timeout: float):
    request = Request(url, headers={"User-Agent": "chimorse-data-downloader"})
    return urlopen(request, timeout=timeout)

# ----------------------------------------------------------------------

def _read_json(url: str, timeout: float) -> dict:
    with _open_url(url, timeout) as response:
        return json.load(response)

# ----------------------------------------------------------------------

def _iter_record_files(record: dict) -> Iterable[dict]:
    """Yield file records from old and new Zenodo record representations."""
    files = record.get("files", [])

    if isinstance(files, list):
        yield from files
        return

    if isinstance(files, dict):
        entries = files.get("entries", {})
        if isinstance(entries, dict):
            for key, value in entries.items():
                item = dict(value)
                item.setdefault("key", key)
                yield item
            return

    raise ValueError("Unexpected Zenodo record file metadata format.")

# ----------------------------------------------------------------------

def _file_name(file_record: dict) -> str:
    name = file_record.get("key") or file_record.get("filename")
    if not name:
        raise ValueError("Zenodo file metadata is missing a file name.")
    return str(name)

# ----------------------------------------------------------------------

def _file_url(file_record: dict) -> str:
    links = file_record.get("links", {})
    url = links.get("content") or links.get("self") or file_record.get("links")
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"Zenodo file metadata for {_file_name(file_record)!r} "
            "does not contain a download URL."
        )
    return url

# ----------------------------------------------------------------------

def _verify_checksum(path: Path, checksum: str | None) -> None:
    if not checksum:
        return

    if ":" in checksum:
        algorithm, expected = checksum.split(":", 1)
    else:
        algorithm, expected = "md5", checksum

    algorithm = algorithm.lower()
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}") from exc

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest().lower() != expected.lower():
        raise IOError(f"Checksum mismatch for downloaded file: {path}")

# ----------------------------------------------------------------------

def download_zenodo_record(
    record_or_doi: str,
    output_dir: str | Path,
    *,
    file_globs: Iterable[str] = ("*",),
    overwrite: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Download selected files from one public Zenodo record.

    Parameters
    ----------
    record_or_doi
        Zenodo record id or DOI. A version-specific DOI is recommended for
        reproducible scientific workflows.
    output_dir
        Directory in which the selected record files will be stored.
    file_globs
        Glob patterns used to select record files.
    overwrite
        Replace existing local files when ``True``.
    timeout
        Network timeout in seconds for each request.
    """
    record_id = _zenodo_record_id(record_or_doi)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    record = _read_json(
        f"https://zenodo.org/api/records/{record_id}",
        timeout,
    )
    patterns = tuple(file_globs)
    selected = [
        item
        for item in _iter_record_files(record)
        if any(fnmatch(_file_name(item), pattern) for pattern in patterns)
    ]

    if not selected:
        raise FileNotFoundError(
            f"No files in Zenodo record {record_or_doi!r} matched {patterns!r}."
        )

    for item in selected:
        name = _file_name(item)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe file name in Zenodo record: {name!r}")

        destination = output_dir / relative
        if destination.exists() and not overwrite:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")

        try:
            with _open_url(_file_url(item), timeout) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file)
            _verify_checksum(temporary, item.get("checksum"))
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    return output_dir

# ----------------------------------------------------------------------

def _dataset_complete(dataset_dir: Path) -> bool:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        interactions = set(metadata["interactions"])
        files = metadata["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False

    return all(
        interaction in files and (dataset_dir / files[interaction]).is_file()
        for interaction in interactions
    )

# ----------------------------------------------------------------------

def ensure_reference_data(
    molecule_id: str,
    data_root: str | Path = "data",
    *,
    force: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Ensure that a registered reference dataset is available locally.

    The function is idempotent: when ``metadata.json`` and all files declared
    by that metadata are present, no network request is made.
    """
    molecule_id = molecule_id.upper()
    try:
        spec = REFERENCE_DATASETS[molecule_id]
    except KeyError as exc:
        raise KeyError(
            f"No reference dataset is registered for {molecule_id!r}. "
            f"Available datasets: {sorted(REFERENCE_DATASETS)}"
        ) from exc

    dataset_dir = Path(data_root).expanduser() / molecule_id
    if not force and _dataset_complete(dataset_dir):
        return dataset_dir

    download_zenodo_record(
        spec.record_or_doi,
        dataset_dir,
        file_globs=spec.file_globs,
        overwrite=force,
        timeout=timeout,
    )

    if not _dataset_complete(dataset_dir):
        raise FileNotFoundError(
            f"Downloaded dataset in {dataset_dir} is incomplete or has invalid metadata."
        )

    return dataset_dir
