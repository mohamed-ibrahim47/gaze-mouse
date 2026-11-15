import json

import numpy as np
import pytest

from gazemouse.gaze_model import GazeModel
from gazemouse.profiles import gazemouse_home, load_profile, save_profile


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GAZEMOUSE_HOME", str(tmp_path))
    return tmp_path


def fitted_model(seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, (50, 8))
    T = F[:, :2] * [500, 300] + [1280, 800]
    m = GazeModel()
    m.fit(F, T)
    return m, F


def test_roundtrip_same_predictions_and_meta():
    m, F = fitted_model()
    meta = {"error_pct": 4.2, "screen": [2560, 1600], "date": "2026-11-02"}
    path = save_profile("default", m, meta)
    assert path == gazemouse_home() / "profiles" / "default.json"
    loaded = load_profile("default")
    assert loaded is not None
    m2, meta2 = loaded
    assert np.allclose(m2.predict(F[3]), m.predict(F[3]))
    assert meta2 == meta


def test_missing_profile_is_none():
    assert load_profile("nobody") is None


def test_corrupt_file_is_none(home):
    (home / "profiles").mkdir()
    (home / "profiles" / "default.json").write_text("{not json")
    assert load_profile("default") is None


def test_unknown_version_is_none(home):
    m, _ = fitted_model()
    path = save_profile("default", m, {})
    data = json.loads(path.read_text())
    data["version"] = 2
    path.write_text(json.dumps(data))
    assert load_profile("default") is None


def test_profile_from_other_feature_set_is_none(home):
    # An older app version used different features: recalibrate, don't crash.
    m, _ = fitted_model()
    path = save_profile("default", m, {})
    data = json.loads(path.read_text())
    data["model"]["used"] = [0, 1]
    path.write_text(json.dumps(data))
    assert load_profile("default") is None


def test_save_leaves_no_temp_files(home):
    m, _ = fitted_model()
    save_profile("default", m, {})
    save_profile("default", m, {"again": True})
    assert [p.name for p in (home / "profiles").iterdir()] == ["default.json"]
    assert load_profile("default")[1] == {"again": True}


@pytest.mark.parametrize("name", ["../evil", "a/b", "a\\b", "", "con:"])
def test_bad_names_rejected(name):
    m, _ = fitted_model()
    with pytest.raises(ValueError):
        save_profile(name, m, {})
    assert load_profile(name) is None


def test_home_defaults_to_appdata(monkeypatch, tmp_path):
    monkeypatch.delenv("GAZEMOUSE_HOME")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    home = gazemouse_home()
    assert home == tmp_path / "roaming" / "GazeMouse" and home.is_dir()
