# PyInstaller spec for the throwaway pursuit-calibration spike (console exe, portable folder).
#   <gaze-mouse>/.venv/Scripts/python -m PyInstaller spikes/pursuit/pursuit.spec --distpath <out> --workpath <tmp>
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

HERE = Path(SPECPATH)
GAZE_SRC = HERE.parent.parent / "src"
FACE_MODEL = GAZE_SRC / "gazemouse" / "models" / "face_landmarker.task"

mp_datas, mp_bins, mp_hidden = collect_all("mediapipe")
datas = mp_datas + [(str(FACE_MODEL), "gazemouse/models"),
                    (str(HERE / "wetlib" / "LICENSE"), "wetlib")]
excludes = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
            "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtPdf", "PySide6.QtCharts",
            "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "tkinter", "pytest", "uiautomation"]

a = Analysis([str(HERE / "pursuit_spike.py")], pathex=[str(HERE), str(GAZE_SRC)], binaries=mp_bins, datas=datas,
             hiddenimports=mp_hidden, excludes=excludes, noarchive=False)
exe = EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name="PursuitSpike", console=True, upx=False)
COLLECT(exe, a.binaries, a.datas, name="PursuitSpike", upx=False)
