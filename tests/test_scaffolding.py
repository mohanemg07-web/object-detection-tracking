"""Phase 0 smoke tests: verify the repo scaffolding is importable and
the committed configs are well-formed. These run on CPU with no dataset
or model present, so CI stays green from the start.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_src_package_imports():
    import src  # noqa: F401
    import src.data  # noqa: F401
    import src.inference  # noqa: F401
    import src.optimization  # noqa: F401
    import src.preprocessing  # noqa: F401
    import src.tracking  # noqa: F401
    import src.training  # noqa: F401


def test_expected_directories_exist():
    for rel in [
        "configs",
        "data",
        "src",
        "app",
        "notebooks",
        "tests",
        "scripts",
        ".github/workflows",
    ]:
        assert (REPO_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_config_files_are_valid_yaml():
    for name in [
        "paths.yaml",
        "data.yaml",
        "train.yaml",
        "bytetrack.yaml",
        "botsort.yaml",
        "preprocess.yaml",
    ]:
        path = CONFIGS / name
        assert path.is_file(), f"missing config: {name}"
        with path.open("r", encoding="utf-8") as fh:
            yaml.safe_load(fh)


def test_data_yaml_has_ten_visdrone_classes():
    with (CONFIGS / "data.yaml").open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data["nc"] == 10
    names = data["names"]
    assert len(names) == 10
    assert names[0] == "pedestrian"
    assert names[9] == "motor"
    # YOLO class ids must be a contiguous 0..9 range
    assert sorted(names.keys()) == list(range(10))
