"""Tests for the VisDrone downloader registry + optional-archive handling.

Guards the fix that moved DET archives to the Ultralytics GitHub mirror and
made MOT-val optional (so its failure never breaks DET training data prep).
No network access — only the in-memory registry contract is checked.
"""

from src.data.download_visdrone import DEFAULT_ARCHIVES, load_archive_registry

ULTRALYTICS_PREFIX = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"


def test_det_archives_use_ultralytics_http_mirror():
    for name in ("VisDrone2019-DET-train", "VisDrone2019-DET-val"):
        spec = DEFAULT_ARCHIVES[name]
        assert spec["url"] == f"{ULTRALYTICS_PREFIX}{name}.zip"
        # Google-Drive id retained as a fallback.
        assert spec["gdrive_id"]


def test_det_archives_are_required():
    for name in ("VisDrone2019-DET-train", "VisDrone2019-DET-val"):
        assert DEFAULT_ARCHIVES[name]["optional"] is False


def test_mot_val_is_optional_and_has_no_invented_url():
    spec = DEFAULT_ARCHIVES["VisDrone2019-MOT-val"]
    assert spec["optional"] is True
    # We must NOT invent a MOT URL; only the gdrive fallback may exist.
    assert spec["url"] == ""


def test_registry_override_merges(tmp_path):
    override = tmp_path / "urls.yaml"
    override.write_text(
        "VisDrone2019-DET-train:\n  url: https://example.com/custom.zip\n",
        encoding="utf-8",
    )
    reg = load_archive_registry(str(override))
    # overridden url wins, other fields preserved
    assert reg["VisDrone2019-DET-train"]["url"] == "https://example.com/custom.zip"
    assert reg["VisDrone2019-DET-train"]["gdrive_id"]
    # untouched entries remain
    assert reg["VisDrone2019-MOT-val"]["optional"] is True


def test_registry_default_when_no_override():
    reg = load_archive_registry(None)
    assert set(reg) == set(DEFAULT_ARCHIVES)
