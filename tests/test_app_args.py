from gazemouse.app import parse_args


def test_defaults():
    a = parse_args([])
    assert a.profile == "default" and a.calibrate is False and a.camera is None


def test_flags():
    a = parse_args(["--profile", "sara", "--calibrate", "--camera", "1"])
    assert a.profile == "sara" and a.calibrate is True and a.camera == 1
