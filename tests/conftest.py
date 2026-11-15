import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # widgets paint without a real display


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
