# PyInstaller spec: one portable folder holding GazeMouse.exe (tray app) and CamCheck.exe (console).
#   .venv/Scripts/python -m PyInstaller packaging/gazemouse.spec --distpath <out> --workpath <tmp>
# Needs src/gazemouse/models/face_landmarker.task first (scripts/fetch_model.py).
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
MODEL = SRC / "gazemouse" / "models" / "face_landmarker.task"
if not MODEL.exists():
    raise SystemExit(f"missing {MODEL}: run scripts/fetch_model.py first")

mp_datas, mp_bins, mp_hidden = collect_all("mediapipe")
datas = mp_datas + [(str(MODEL), "gazemouse/models"), (str(ROOT / "LICENSE"), ".")]
hidden = mp_hidden + collect_submodules("gazemouse") + collect_submodules("uiautomation") + ["comtypes.stream"]
# Qt modules the app never uses; WebEngine alone is ~200 MB.
excludes = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
            "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtPdf", "PySide6.QtCharts",
            "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "tkinter", "pytest"]
# matplotlib stays: mediapipe.tasks.python.vision imports it at load time.


def analysis(script):
    return Analysis([str(ROOT / "packaging" / script)], pathex=[str(SRC)], binaries=mp_bins, datas=datas,
                    hiddenimports=hidden, excludes=excludes, noarchive=False)


app_a = analysis("launch_app.py")
cam_a = analysis("launch_camcheck.py")

app_exe = EXE(PYZ(app_a.pure), app_a.scripts, [], exclude_binaries=True, name="GazeMouse",
              console=False, upx=False)
cam_exe = EXE(PYZ(cam_a.pure), cam_a.scripts, [], exclude_binaries=True, name="CamCheck",
              console=True, upx=False)

COLLECT(app_exe, app_a.binaries, app_a.datas, cam_exe, cam_a.binaries, cam_a.datas,
        name="GazeMouse", upx=False)
