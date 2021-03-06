from builtins import str
import dmri_segmenter.make_comparisons as mc
import numpy as np
import os
import pytest


@pytest.mark.parametrize("test_input,expected",
                         [(([1, 1], [1, 1]), (0, 1, 1)),
                          (([0, 1], [1, 0]), (2, 0, 0)),
                          (([0, 1, 1], [1, 0, 1]), (1, 0.5, 0.333))])
def test_get_re_dc_and_ji(test_input, expected):
    rdj = mc.get_re_dc_and_ji(*test_input)
    assert (np.abs(rdj - np.array(expected)) < 0.01).all()


def test_mask_from_possible_filename():
    inp = 0.5 * np.arange(5)
    out = mc.mask_from_possible_filename(inp, thresh=0.7)
    exp = np.array([0, 0, 1, 1, 1])
    assert (out == exp).all()


def test_make_3way_comparison_image(tmpdir, fakedata):
    m1 = fakedata.phantom.copy()
    m1[40, 40:50, ::5] = 0
    m1[40, 51] = 1
    m1fn = str(tmpdir.join('m1.nii'))
    assert m1fn != 'm1.nii'
    mc.save_mask(m1, fakedata.aff, m1fn)
    m2 = fakedata.phantom.copy()
    m2[40, 40:50, ::2] = 0
    m2[40, 52] = 1
    m2fn = str(tmpdir.join('m2.nii'))
    mc.save_mask(m2, fakedata.aff, m2fn)
    gfn = str(tmpdir.join('g.nii'))
    mc.save_mask(fakedata.phantom, fakedata.aff, gfn)
    res = mc.make_3way_comparison_image(m1fn, m2fn, gfn)

    for fn in (m1fn, m2fn, gfn):
        assert os.path.isfile(fn)

    expsums = {'g': 205927,
               'g only': 50,
               'm1 and g only': 220,
               'm1 and m2 only': 0,
               'm1 only': 7,
               'm2 and g only': 50,
               'm2 only': 7}
    assert res['sums'] == expsums

    expre = {'m1': 0.000519601606394499, 'm2': 0.001345136868890432}
    for k, v in expre.items():
        assert abs(res['relative errors'][k] - v) < 1e-5
