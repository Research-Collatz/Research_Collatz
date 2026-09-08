"""Tests for experiment-level reproducibility manifests."""

import json

from collatz_graph.reproducibility import write_experiment_manifest


def test_experiment_manifest_records_config_and_environment(tmp_path) -> None:
    paths = write_experiment_manifest(
        tmp_path / "experiment",
        {"max_nodes": [10, 20], "seed": 42, "node2vec": {"dimensions": 8}},
        metadata={"experiment_name": "test-run"},
    )

    assert paths == {
        "config": tmp_path / "experiment" / "config.json",
        "metadata": tmp_path / "experiment" / "metadata.json",
    }
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert config["max_nodes"] == [10, 20]
    assert config["node2vec"]["dimensions"] == 8
    assert config["seed"] == 42
    assert metadata["experiment_name"] == "test-run"
    assert metadata["git_commit_sha"]
    assert metadata["python_version"]
    assert metadata["numpy_version"]
    assert metadata["networkx_version"]
    assert metadata["package_versions"]["numpy"]
