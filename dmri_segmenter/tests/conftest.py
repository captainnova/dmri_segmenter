from __future__ import print_function
from builtins import str
from builtins import map
from builtins import range
from builtins import object
import nibabel as nib
import numpy as np
import os
import pytest
from scipy.ndimage.filters import convolve

#print("__name__", __name__)

import dmri_segmenter.train as train
#from .. import train


class Phantom(object):
    def __init__(self):
        # Don't make phantom too small; make_phantom_mask, at least with the
        # defaults, does some convolutions with nominal scales.
        phantom = np.zeros((80, 90, 70), dtype=np.uint8)
        phantom[10:65, 10:70, 2:65] = 2       # CSF rim
        phantom[13:62, 13:67, 5:62] = 1       # brain

        # Make a hole.
        phantom[40:50, 30:50, 30:40] = 0

        # Add a disconnected blip.
        phantom[2:5, 2:5, 66:69] = 2

        nvols = 7
        bvals = np.zeros(nvols)
        bvals[1:] = 1000

        s0 = 50.0 * phantom

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
            data[..., v] = s0
            data[10:65, 10:70, 2:65, v] *= np.exp(-0.003 * b)
            data[13:62, 13:67, 5:62, v] *= np.exp(0.0023 * b)  # Back CSF diffusion to brain level
        aff = np.eye(4)

        # Make the voxels bigger and anisotropic.
        for i in range(3):
            aff[i, i] = 2.0 + 0.25 * i

        self.phantom = phantom
        self.aff = aff
        self.bvals = bvals
        self.data = data


@pytest.fixture(scope="session")
def fakedata():
    return Phantom()


@pytest.fixture(scope="session")
def fvecfn(tmpdir_factory, fakedata):
    testdir = str(tmpdir_factory.getbasetemp())
    dwfn = os.path.join(testdir, 'dw.nii')

    # The coverage is better if this isn't dwfn.bval.
    bvalfn = os.path.join(testdir, 'bvalfn')

    with open(bvalfn, 'w') as f:
        f.write("%s\n" % " ".join(map(str, fakedata.bvals)))
    nib.save(nib.nifti1.Nifti1Image(fakedata.data, fakedata.aff), dwfn)
    return train.make_fvecs(dwfn, smoothrad=4.0, Dt=0.0014, Dcsf=0.0021)
