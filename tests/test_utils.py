import dmri_segmenter.utils as utils
import numpy as np
import os
import pytest

inarr = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 1, 1, 1, 0, 1, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

expcarr = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

expoarr = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

expfarr = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

inp3d = np.zeros(inarr.shape + (5,), dtype=np.uint8)
for v in [1, 2, 3]:
    inp3d[..., v] = inarr
inp3d[2, 4, 2] = 0                 # Make hole in cube
expfharr = inp3d.copy()
expfharr[2, 4, 2] = 1
expfaharr = expfharr.copy()
expfaharr[2, 6, 1:4] = 1


def test_binary_closing():
    carr = utils.binary_closing(inarr)
    if not (carr == expcarr).all():
        print carr
    assert (carr == expcarr).all() == True     # noqa - in this case True == True but True not is True


def test_binary_opening():
    oarr = utils.binary_opening(inarr)
    if not (oarr == expoarr).all():
        print oarr
    assert (oarr == expoarr).all() == True     # noqa - in this case True == True but True not is True


def test_fill_holes():
    farr, errval = utils.fill_holes(inp3d, np.eye(4))
    if not (farr == expfharr).all():
        print farr[..., 1]
    assert errval == 0
    assert (farr == expfharr).all() == True    # noqa - in this case True == True but True not is True


def test_fill_axial_holes():
    farr, errval = utils.fill_axial_holes(inp3d)
    if not (farr == expfaharr).all():
        print farr[..., 1]
    assert errval == 0
    assert (farr == expfaharr).all() == True   # noqa - in this case True == True but True not is True


def test_get_1_file():
    cand = utils.get_1_file_or_hurl(utils.__file__)
    assert cand == utils.__file__


def test_get_1_file_and_hurl():
    dsdir = os.path.dirname(utils.__file__)
    with pytest.raises(ValueError):
        utils.get_1_file_or_hurl(os.path.join(dsdir, 'd*'))


def test_instaprint(capsys):
    msg = "Hello!\n"
    utils.instaprint(msg)
    # captured = capsys.readouterr()
    # This doesn't work like I thought it would according to
    # https://docs.pytest.org/en/latest/capture.html
    # assert captured.out == msg
