#!/usr/bin/env python3
"""Fetch simulation assets and install local DeformGen symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import OmegaConf


def _load_registry(path: Path) -> dict[str, Any]:
    """Load and validate the asset source registry."""
    if not path.is_file():
        raise FileNotFoundError(f"Asset source registry does not exist: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict) or not isinstance(data.get("sources"), dict):
        raise ValueError(f"Invalid source registry: {path}")
    return data


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requested_revision(source: dict[str, Any], source_name: str) -> str | None:
    """Return an optional source revision; None follows the Hub default branch."""
    revision = source.get("revision")
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision or revision.startswith("PENDING_"):
        raise ValueError(
            f"Source {source_name} has an invalid revision in assets/sources.yaml. "
            "Use a non-empty Hugging Face revision or omit the field."
        )
    return revision


def _revision_label(revision: str | None) -> str:
    """Return a human-readable label for an optional Hub revision."""
    return revision or "default"


def _revision_kwargs(revision: str | None) -> dict[str, str]:
    """Build Hub keyword arguments without forcing a mutable branch name."""
    return {} if revision is None else {"revision": revision}


def _hf_import() -> tuple[Any, Any]:
    """Import Hugging Face download functions with an actionable error."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install with: uv pip install huggingface_hub"
        ) from exc
    return hf_hub_download, snapshot_download


def _set_endpoint(endpoint: str | None) -> None:
    """Configure an optional Hugging Face endpoint for the current process."""
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint


def _with_download_retries(operation: Any, description: str, attempts: int = 5) -> Any:
    """Retry transient Hub failures, including mirror rate limiting, with backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # Hub errors have several concrete exception types.
            last_error = exc
            if attempt == attempts:
                break
            delay = min(30, 2 ** (attempt - 1))
            print(
                f"Download retry {attempt}/{attempts} for {description}: {exc}. "
                f"Waiting {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _download_snapshot(
    source_name: str,
    source: dict[str, Any],
    cache_root: Path,
    endpoint: str | None,
    allow_patterns: list[str] | None = None,
    force_download: bool = False,
) -> Path:
    """Download a Hugging Face snapshot into the persistent asset cache."""
    _set_endpoint(endpoint)
    _, snapshot_download = _hf_import()
    revision = _requested_revision(source, source_name)
    return Path(
        _with_download_retries(
            lambda: snapshot_download(
                repo_id=str(source["repo_id"]),
                repo_type=str(source["repo_type"]),
                **_revision_kwargs(revision),
                cache_dir=str(cache_root / "huggingface"),
                allow_patterns=allow_patterns,
                force_download=force_download,
            ),
            f"{source_name}@{_revision_label(revision)}",
        )
    )


def _download_subtree(
    source_name: str,
    source: dict[str, Any],
    cache_root: Path,
    endpoint: str | None,
    relative_path: str,
) -> Path:
    """Download a required subtree and retry once if an interrupted cache is incomplete."""
    pattern = relative_path.rstrip("/") + "/**"
    root = _download_snapshot(
        source_name, source, cache_root, endpoint, allow_patterns=[pattern]
    )
    target = root / relative_path
    if target.exists():
        return target
    root = _download_snapshot(
        source_name,
        source,
        cache_root,
        endpoint,
        allow_patterns=[pattern],
        force_download=True,
    )
    target = root / relative_path
    if not target.exists():
        raise FileNotFoundError(
            f"Downloaded source {source_name} does not contain required path: {relative_path}"
        )
    return target


def _download_gs_archive(
    source_name: str,
    source: dict[str, Any],
    cache_root: Path,
    endpoint: str | None,
) -> Path:
    """Download and extract the upstream GS archive once per resolved content hash."""
    _set_endpoint(endpoint)
    hf_hub_download, _ = _hf_import()
    revision = _requested_revision(source, source_name)
    archive_name = str(source["archive"])
    archive = Path(
        _with_download_retries(
            lambda: hf_hub_download(
                repo_id=str(source["repo_id"]),
                repo_type=str(source["repo_type"]),
                filename=archive_name,
                **_revision_kwargs(revision),
                cache_dir=str(cache_root / "huggingface"),
            ),
            f"{source_name}/{archive_name}@{_revision_label(revision)}",
        )
    )
    archive_sha256 = _sha256(archive)
    cache_version = revision or f"default-{archive_sha256[:16]}"
    destination = cache_root / "sim-assets" / source_name / cache_version / "extracted"
    marker = destination / ".deformgen_extract_complete.json"
    if marker.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deformgen_gs_extract_", dir=destination.parent) as temp_dir:
        temporary = Path(temp_dir) / "extracted"
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(temporary), str(destination))
    marker.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": archive_sha256,
                "requested_revision": _revision_label(revision),
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def _find_named_dir(root: Path, name: str) -> Path:
    """Find one scan directory by name in the extracted upstream archive."""
    candidates = [path for path in root.rglob(name) if path.is_dir() and path.name == name]
    preferred = [path for path in candidates if path.parent.name == "scans"]
    candidates = preferred or candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory named {name!r} under {root}; found {candidates}"
        )
    return candidates[0]


def _same_target(existing: Path, source: Path) -> bool:
    """Check whether an existing path already resolves to the requested source."""
    try:
        return existing.resolve() == source.resolve()
    except OSError:
        return False


def _install_link(source: Path, target: Path, force: bool) -> str:
    """Install a symlink only after all source paths have been downloaded."""
    if not source.exists():
        raise FileNotFoundError(f"Cannot link missing asset source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and _same_target(target, source):
        return "already-linked"
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(
                f"Refusing to replace existing target {target}. Use --force only after verifying it."
            )
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=source.is_dir())
    return "linked"


def _case_sources(
    registry: dict[str, Any],
    case: str,
    cache_root: Path,
    endpoint: str | None,
) -> list[tuple[Path, Path, dict[str, str]]]:
    """Resolve required source/target pairs for one case without touching repo links."""
    sources = registry["sources"]
    case_cfg = registry["cases"][case]
    links: list[tuple[Path, Path, dict[str, str]]] = []

    if case == "cloth3":
        for scan_name in case_cfg["gs_dirs"]:
            relative = f"cloth3/log/gs/scans/{scan_name}"
            source = _download_subtree(
                "deformgen_simassets", sources["deformgen_simassets"], cache_root, endpoint, relative
            )
            links.append((source, Path("log/gs/scans") / str(scan_name), {"source": "deformgen_simassets"}))
    else:
        gs_root = _download_gs_archive(
            "upstream_gs_scans", sources["upstream_gs_scans"], cache_root, endpoint
        )
        for scan_name in case_cfg["gs_dirs"]:
            source = _find_named_dir(gs_root, str(scan_name))
            links.append((source, Path("log/gs/scans") / str(scan_name), {"source": "upstream_gs_scans"}))

    physics_name = str(case_cfg["physics_source"])
    physics_relative = str(case_cfg.get("physics_relpath", "."))
    if physics_relative == ".":
        physics_root = _download_snapshot(physics_name, sources[physics_name], cache_root, endpoint)
        physics_source = physics_root
    else:
        physics_source = _download_subtree(
            physics_name, sources[physics_name], cache_root, endpoint, physics_relative
        )
    links.append((physics_source, Path("log/phystwin") / case, {"source": physics_name}))

    demo_source = _download_subtree(
        "deformgen_simassets",
        sources["deformgen_simassets"],
        cache_root,
        endpoint,
        str(case_cfg["demo_relpath"]),
    )
    links.append((demo_source, Path(str(case_cfg["demo_target"])), {"source": "deformgen_simassets"}))
    return links


def _write_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    """Write a local reproducibility manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _iter_cases(requested: str) -> Iterable[str]:
    """Expand one case or all supported cases."""
    return ["rope", "sloth", "cloth3"] if requested == "all" else [requested]


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch released simulation assets and link them into a DeformGen checkout."""
    parser = argparse.ArgumentParser(description="Fetch DeformGen simulation assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assets_parser = subparsers.add_parser("sim-assets", help="Install GS, PhysTwin, and demo assets.")
    assets_parser.add_argument("--case", choices=["rope", "sloth", "cloth3", "all"], default="all")
    assets_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    assets_parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache" / "deformgen")
    assets_parser.add_argument("--sources", type=Path, default=None)
    assets_parser.add_argument("--endpoint", default=None, help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    assets_parser.add_argument("--force", action="store_true", help="Replace conflicting local log targets.")
    assets_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    registry_path = args.sources or repo_root / "assets" / "sources.yaml"
    _set_endpoint(getattr(args, "endpoint", None))
    registry = _load_registry(registry_path)
    entries: list[dict[str, Any]] = []
    try:
        if args.dry_run:
            for case in _iter_cases(args.case):
                case_cfg = registry["cases"][case]
                names = [str(case_cfg["physics_source"]), "deformgen_simassets"]
                if case != "cloth3":
                    names.insert(0, "upstream_gs_scans")
                for source_name in dict.fromkeys(names):
                    source = registry["sources"][source_name]
                    revision = _requested_revision(source, source_name)
                    print(
                        f"PLAN source={source_name} repo={source['repo_id']} "
                        f"revision={_revision_label(revision)} case={case}"
                    )
            return 0
        for case in _iter_cases(args.case):
            for source, relative_target, metadata in _case_sources(registry, case, args.cache_root, args.endpoint):
                target = repo_root / relative_target
                row = {
                    "case": case,
                    "source": metadata["source"],
                    "source_path": str(source),
                    "requested_revision": _revision_label(
                        _requested_revision(registry["sources"][metadata["source"]], metadata["source"])
                    ),
                    "target": str(target),
                }
                row["status"] = _install_link(source, target, args.force)
                print(f"{row['status'].upper()} {source} -> {target}")
                entries.append(row)
    except Exception as exc:
        print(f"deformgen-fetch: {exc}", file=sys.stderr)
        return 2

    if not args.dry_run:
        _write_manifest(repo_root / "log" / "external_assets" / "resolved_manifest.json", entries)
        print(f"manifest={repo_root / 'log' / 'external_assets' / 'resolved_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
