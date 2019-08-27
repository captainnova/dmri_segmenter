import dmri_segmenter.train as train
import nibabel as nib
import numpy as np
import os


def test_make_fvecs(fvecfn, fakedata):
    assert os.path.isfile(fvecfn)
    fvnii = nib.load(fvecfn)
    assert (fvnii.affine == fakedata.aff).all()
    assert np.linalg.norm(fvnii.dataobj[40, 45, 35] -
                          [-0.00191109, 0.0, -0.68160387, -0.68159452, 1.0]) < 0.01
    assert "\n\tsmoothrad = 4.000000 mm\n" in fvnii.header.extensions[0].get_content()


def test_make_segmentation(fvecfn):
    dsdir = os.path.dirname(train.__file__)
    clffn = os.path.join(dsdir, 'RFC_classifier.pickle')
    segfn = train.make_segmentation(fvecfn, clffn)
    assert os.path.isfile(segfn)
    segnii = nib.load(segfn)
    assert (segnii.dataobj[37, 42, ::10] == [0, 3, 3, 3, 3, 3, 3]).all()
