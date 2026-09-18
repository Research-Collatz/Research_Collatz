"""Shared environment metadata for reproducible experiment artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def get_git_commit_sha(cwd: str | Path | None = None) -> str | None:
    """Return the current Git commit SHA, or ``None`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _torch_metadata() -> dict[str, Any]:
    """Return optional PyTorch and CUDA details without requiring PyTorch."""
    if importlib.util.find_spec("torch") is None:
        return {}
    try:
        import torch
    except ImportError:
        return {}

    metadata: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        metadata["gpu_count"] = torch.cuda.device_count()
        metadata["gpu_names"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return metadata


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "networkx", "tqdm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def get_environment_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect JSON-serializable runtime, host, and source metadata."""
    package_versions = _package_versions()
    metadata: dict[str, Any] = {
        "git_commit_sha": get_git_commit_sha(cwd),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": package_versions,
    }
    for package in ("numpy", "networkx", "pandas"):
        if package in package_versions:
            metadata[f"{package}_version"] = package_versions[package]
    metadata.update(_torch_metadata())
    return metadata


def write_experiment_manifest(
    output_directory: str | Path,
    config: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the exact experiment configuration and runtime metadata."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": output / "config.json",
        "metadata": output / "metadata.json",
    }
    paths["config"].write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_metadata = get_environment_metadata()
    if metadata:
        manifest_metadata.update(metadata)
        artifact_directories = metadata.get("artifact_directories", {})
        if isinstance(artifact_directories, dict):
            for relative_path in artifact_directories.values():
                (output / str(relative_path)).mkdir(parents=True, exist_ok=True)
    paths["metadata"].write_text(
        json.dumps(manifest_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths
