from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
import dmri_segmenter.utils as utils   # noqa E402
from io import StringIO   # noqa E402
import numpy as np   # noqa E402
import os   # noqa E402
import pytest   # noqa E402

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
        print(carr)
    assert (carr == expcarr).all() == True     # noqa - in this case True == True but True not is True


def test_binary_opening():
    oarr = utils.binary_opening(inarr)
    if not (oarr == expoarr).all():
        print(oarr)
    assert (oarr == expoarr).all() == True     # noqa - in this case True == True but True not is True


def test_fill_holes():
    farr, errval = utils.fill_holes(inp3d, np.eye(4))
    if not (farr == expfharr).all():
        print(farr[..., 1])
    assert errval == 0
    assert (farr == expfharr).all() == True    # noqa - in this case True == True but True not is True


def test_fill_axial_holes():
    farr, errval = utils.fill_axial_holes(inp3d)
    if not (farr == expfaharr).all():
        print(farr[..., 1])
    assert errval == 0
    assert (farr == expfaharr).all() == True   # noqa - in this case True == True but True not is True


def test_get_1_file():
    cand = utils.get_1_file_or_hurl(utils.__file__)
    assert cand == utils.__file__


def test_get_1_file_and_hurl():
    dsdir = os.path.dirname(utils.__file__)
    with pytest.raises(ValueError):
        utils.get_1_file_or_hurl(os.path.join(dsdir, 'd*'))


def test_instaprint():
    msg = "Hello!\n"
    s = StringIO()
    utils.instaprint(msg, s)
    assert s.getvalue() == msg + "\n"


def test_cond_to_mask():
    seg = np.arange(5)
    mask = utils.cond_to_mask(seg, 2)
    assert (mask == np.array([0, 0, 1, 0, 0])).all()
