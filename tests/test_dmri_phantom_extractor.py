import dmri_segmenter.dmri_phantom_extractor as dpe
import dmri_segmenter.make_comparisons as mc
import nibabel as nib
import numpy as np
from scipy.ndimage.filters import convolve


class Phantom(object):
    def __init__(self):
        # Don't make phantom too small; make_phantom_mask, at least with the
        # defaults, does some convolutions with nominal scales.
        phantom = np.zeros((80, 90, 70), dtype=np.uint8)

        phantom[10:65, 10:70, 2:65] = 1
        nvols = 7
        bvals = np.zeros(nvols)
        bvals[1:] = 1000

        s0 = np.zeros(phantom.shape)
        s0[phantom > 0] = 100.0

        # Give it some shading and striping, and shift and soften the edge, to both
        # make it more realistic and prevent otsu() from going haywire.
        s0[..., ::2] += 2.0
        for z in range(2, 25):
            s0[..., z] *= 1.0 + 0.005 * z
        softener = np.zeros((3, 3, 3))
        softener[:, 1, 1] = [0.5, 0.5, 0]
        softener[1, :, 1] = [0.3, 0.7, 0]
        softener[1, 1, :] = [0, 0.3, 0.7]
        softener /= softener.sum()
        s0 = convolve(s0, softener, mode='constant')

        data = np.zeros(s0.shape + (nvols,))
        for v, b in enumerate(bvals):
            data[..., v] = np.exp(-0.0021 * b) * s0
        aff = np.eye(4)

        # Make the voxels bigger and anisotropic.
        for i in range(3):
            aff[i, i] = 2.0 + 0.25 * i

        self.phantom = phantom
        self.aff = aff
        self.bvals = bvals
        self.data = data


def test_make_phantom_mask():
    testdata = Phantom()
    dnii = nib.nifti1.Nifti1Image(testdata.data, testdata.aff)
    mask = dpe.make_phantom_mask(dnii, testdata.bvals, 0.0)
    softener = np.ones((3, 3, 3))
    softener /= softener.sum()
    s0 = convolve(testdata.data[..., 0], softener, mode='constant')
    expmask = np.zeros_like(testdata.phantom)
    expmask[s0 > 5] = 1
    ji = mc.jaccard_index(expmask, mask)
    assert ji > 0.85                # The convolutions make things sloppy.
