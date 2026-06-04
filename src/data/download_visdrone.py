"""Download and extract the VisDrone dataset (resumable, cached).

VisDrone is distributed as zip archives (official mirrors on Google Drive
and the AISKYEYE site). We support:

  * HTTP(S) direct links  -> streamed with HTTP Range resume.
  * Google Drive file ids -> delegated to ``gdown`` if installed.

Already-extracted archives are detected and skipped, so re-running is
cheap. Archive URLs change over time, so they live in a small registry
below and can be overridden with ``--urls path/to/urls.yaml``.

Usage:
    python -m src.data.download_visdrone --config configs/paths.yaml
    python -m src.data.download_visdrone --only VisDrone2019-DET-train
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

from src.data.config import load_paths

# --- Default archive registry ------------------------------------------------
# These are the official VisDrone2019 archive names. The `url` field is the
# best-effort public link; mirrors rotate, so if a download 404s, pass your
# own --urls file mapping the same archive names to fresh links or local
# paths. `gdrive_id` (if set) is used via gdown when `url` is empty.
DEFAULT_ARCHIVES: dict[str, dict[str, str]] = {
    "VisDrone2019-DET-train": {
        "url": "",
        "gdrive_id": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "split": "det-train",
    },
    "VisDrone2019-DET-val": {
        "url": "",
        "gdrive_id": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
        "split": "det-val",
    },
    "VisDrone2019-MOT-val": {
        "url": "",
        "gdrive_id": "1-qX2d-P1Xj4M2v3pCfV1V3Z8b2j3VYy0",
        "split": "mot-val",
    },
}


def load_archive_registry(urls_path: str | None) -> dict[str, dict[str, str]]:
    """Load the archive registry, optionally overridden by a YAML file."""
    if not urls_path:
        return DEFAULT_ARCHIVES
    with Path(urls_path).open("r", encoding="utf-8") as fh:
        override = yaml.safe_load(fh) or {}
    registry = {k: dict(v) for k, v in DEFAULT_ARCHIVES.items()}
    for name, spec in override.items():
        registry.setdefault(name, {}).update(spec)
    return registry


def _is_extracted(raw_dir: Path, archive_name: str) -> bool:
    """An archive counts as extracted if its top-level folder exists."""
    target = raw_dir / archive_name
    return target.is_dir() and any(target.iterdir())


def _download_http(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a URL to ``dest`` with HTTP Range resume support."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if existing and resp.status_code == 200:
            # Server ignored Range; restart cleanly.
            existing = 0
            mode = "wb"
        elif resp.status_code in (200, 206):
            mode = "ab" if existing else "wb"
        else:
            resp.raise_for_status()

        total = int(resp.headers.get("Content-Length", 0)) + existing
        done = existing
        with dest.open(mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print(
                        f"\r  {dest.name}: {done >> 20} / {total >> 20} MB ({pct:4.1f}%)",
                        end="",
                        flush=True,
                    )
        print()


def _download_gdrive(file_id: str, dest: Path) -> None:
    """Download a Google Drive file by id using gdown (handles big files)."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Google Drive download requires 'gdown'. Install it with "
            "`pip install gdown`, or pass --urls with direct HTTP links."
        ) from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)


def _extract(zip_path: Path, raw_dir: Path) -> None:
    print(f"  extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)


def download_archive(name: str, spec: dict[str, str], raw_dir: Path) -> bool:
    """Ensure a single archive is downloaded + extracted.

    Returns True on success, False if it could not be obtained (caller
    decides whether that is fatal).
    """
    if _is_extracted(raw_dir, name):
        print(f"[skip] {name} already extracted")
        return True

    zip_path = raw_dir / f"{name}.zip"
    url = spec.get("url", "").strip()
    gid = spec.get("gdrive_id", "").strip()

    try:
        if url:
            print(f"[get ] {name} via HTTP")
            _download_http(url, zip_path)
        elif gid:
            print(f"[get ] {name} via Google Drive id={gid}")
            _download_gdrive(gid, zip_path)
        else:
            print(
                f"[warn] no source configured for {name}; "
                f"drop {name}.zip into {raw_dir} manually or pass --urls"
            )
            return zip_path.exists() and _safe_extract(zip_path, raw_dir, name)
    except Exception as exc:  # noqa: BLE001 - surface and continue
        print(f"[fail] {name}: {exc}")
        return False

    return _safe_extract(zip_path, raw_dir, name)


def _safe_extract(zip_path: Path, raw_dir: Path, name: str) -> bool:
    if not zip_path.exists():
        print(f"[fail] {name}: archive not found at {zip_path}")
        return False
    try:
        _extract(zip_path, raw_dir)
    except zipfile.BadZipFile:
        print(f"[fail] {name}: corrupt archive; delete {zip_path} and retry")
        return False
    return _is_extracted(raw_dir, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download + extract VisDrone (resumable).")
    parser.add_argument("--config", default="configs/paths.yaml", help="paths config")
    parser.add_argument("--urls", default=None, help="optional YAML overriding archive sources")
    parser.add_argument("--only", default=None, help="download just one archive by name")
    args = parser.parse_args(argv)

    paths = load_paths(args.config)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    registry = load_archive_registry(args.urls)

    if args.only:
        if args.only not in registry:
            print(f"unknown archive {args.only!r}; known: {list(registry)}")
            return 2
        registry = {args.only: registry[args.only]}

    print(f"Target raw dir: {paths.raw_dir}")
    ok = True
    for name, spec in registry.items():
        ok = download_archive(name, spec, paths.raw_dir) and ok

    if not ok:
        print(
            "\nSome archives could not be downloaded automatically (mirrors "
            "rotate frequently). Options:\n"
            "  1. Get fresh links from http://aiskyeye.com/download/ and pass "
            "them via --urls urls.yaml\n"
            "  2. Download the zips manually into the raw dir, then re-run "
            "this script to extract them."
        )
        return 1

    print("\nAll archives present and extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
