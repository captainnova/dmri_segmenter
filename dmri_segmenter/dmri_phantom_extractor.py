from __future__ import absolute_import
import nibabel as nib
import numpy as np

from . import dmri_brain_extractor as dbe
from . import utils


def make_phantom_mask(img, bvals, closerad=3, dtype=np.uint8,
                      Dt=0.0015, Dcsf=0.0021, ncomponents=None):
    """
    Make a mask for the "interesting" voxels of img.

    Parameters
    ----------
    img: str or nib.Nifti1Image
        A 4D diffusion MRI NIfTI image, or its filename.
    bvals: array
        The diffusion weightings of each volume in img.
    closerad: float
        How many mm to close by at the end. Set it to a bit more than half the
        thickness of any plastic features you want to include.
    dtype: type
        The data type for the output.
    Dt: float
        The axial diffusiivity of the restricted component (if any),
        in reciprocal units of bvals.
    Dcsf: float
        The mean diffusiivity of the unrestricted component,
        in reciprocal units of bvals. The defaults are suitable for
        a (mostly) water phantom at ~18C.
    ncomponents: None or int > 0
        The number of separate regions to keep. If None it will be determined
        by Otsu thresholding between large bright and small dim regions.

    Returns
    -------
    mask: 3d array with dtype np.uint8
        1 where the voxels are bright enough in either diffusion weighted or diffusion unweighted volumes,
        0 elsewhere.
    """
    if not hasattr(img, 'get_data'):
        dnii = nib.load(img)
    else:
        dnii = img
    data = dnii.get_data()
    s0 = utils.calc_average_s0(data, bvals)
    madje, bscale = dbe.make_mean_adje(data, bvals, Dt=Dt, Dcsf=Dcsf)
    gtiv = dbe.make_grad_based_TIV(s0, madje, dnii.affine, ncomponents=ncomponents,
                                   is_phantom=True)

    # gtiv has had hole filling, but for a phantom try harder to fill in dark patches
    # around plastic structures.
    ball = utils.make_structural_sphere(dnii.affine, 3 * closerad)
    rind = utils.binary_dilation(gtiv, ball)
    rind[gtiv > 0] = 0
    thresh = dbe.otsu(s0[rind > 0])

    rindmin = s0[rind > 0].min()
    if rindmin >= thresh:        # Mainly happens in testing.
        thresh = 0.5 * (rindmin + np.median(s0[gtiv > 0]))

    rind[s0 < thresh] = 0
    ball = utils.make_structural_sphere(dnii.affine, closerad)
    mask = utils.binary_closing(gtiv + rind, ball)

    return mask.astype(dtype)
