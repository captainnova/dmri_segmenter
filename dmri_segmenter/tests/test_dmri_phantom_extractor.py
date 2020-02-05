import dmri_segmenter.dmri_phantom_extractor as dpe
import dmri_segmenter.make_comparisons as mc
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import convolve


def test_make_phantom_mask(fakedata):
    dnii = nib.nifti1.Nifti1Image(fakedata.data, fakedata.aff)
    mask = dpe.make_phantom_mask(dnii, fakedata.bvals, 0.0)
    softener = np.ones((3, 3, 3))
    softener /= softener.sum()
    s0 = convolve(fakedata.data[..., 0], softener, mode='constant')
    expmask = np.zeros_like(fakedata.phantom)
    expmask[s0 > 5] = 1
    ji = mc.jaccard_index(expmask, mask)
    assert ji > 0.85                # The convolutions make things sloppy.
