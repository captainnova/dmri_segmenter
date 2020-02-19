import dmri_segmenter.train as train
import nibabel as nib
import numpy as np
import os


def test_make_fvecs(fvecfn, fakedata):
    assert os.path.isfile(fvecfn)
    fvnii = nib.load(fvecfn)
    assert (fvnii.affine == fakedata.aff).all()
    assert np.linalg.norm(fvnii.dataobj[40, 45, 35] -
                          [-2.11442034, -1.91812569, -1.64265231, -1.45012367, 1.0]) < 0.01

    # In python 3 get_content() returns a bytes object. str() doesn't work as
    # wanted (it prepends a b and escapes the \t and \n), so it needs to be
    # decoded. ASCII would also work, and this is safe for both python 2(.7)
    # and 3.
    assert "\n\tsmoothrad = 4.000000 mm\n" in fvnii.header.extensions[0].get_content().decode("utf-8")


def test_make_segmentation(fvecfn):
    dsdir = os.path.dirname(train.__file__)
    clffn = os.path.join(dsdir, 'RFC_classifier.pickle')
    segfn = train.make_segmentation(fvecfn, clffn)
    assert os.path.isfile(segfn)
    segnii = nib.load(segfn)
    assert (segnii.dataobj[30, 20, -8:-2] == [0, 0, 3, 3, 2, 0]).all()
