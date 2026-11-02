import numpy as np
import pytest

from gazemouse.landmarks import MODEL_PATH, FaceTracker

pytestmark = [pytest.mark.model,
              pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/fetch_model.py first")]


@pytest.fixture(scope="module")
def tracker():
    return FaceTracker(MODEL_PATH)


def test_black_frame_has_no_face(tracker):
    assert tracker.process(np.zeros((720, 1280, 3), np.uint8), 1.0) is None


def test_out_of_order_times_do_not_raise(tracker):
    frame = np.zeros((480, 640, 3), np.uint8)
    tracker.process(frame, 5.0)
    tracker.process(frame, 4.0)
    tracker.process(frame, 5.0)
    assert tracker.process(frame, 5.0) is None
