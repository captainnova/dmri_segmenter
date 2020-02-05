from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import range
#from past.utils import old_div
from glob import glob
from multiprocessing import cpu_count
import numpy as np
import scipy.ndimage as ndi
try:
    from skimage.filter import threshold_otsu as otsu
except Exception:
    from dipy.segment.threshold import otsu
from skimage.morphology import reconstruction
import sys


def _check_img_and_selem(img, structure):
    """
    Massage img and selem as lightly as possible to make them work with the
    binary_ morphological functions, or throw an exception if they won't work.
    """
    img = np.asarray(img)
    if structure is None:
        structure = ndi.generate_binary_structure(img.ndim, 1)
    else:
        structure = np.asarray(structure).astype(bool)
    if structure.ndim != img.ndim:
        raise RuntimeError('structure and input must have same dimensionality')
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if not structure.flags.contiguous:
        structure = structure.copy()
    if np.product(structure.shape, axis=0) < 1:
        raise RuntimeError('structure must not be empty')

    if np.sum(structure) <= 255:
        conv = np.empty_like(img, dtype=np.uint8)
    else:
        conv = np.empty_like(img, dtype=np.uint)
    binary = (img > 0).view(np.uint8)

    return binary, conv, structure


def binary_closing(arr, structure=None, out=None, mode='constant', cval=0,
                   origin=0):
    """
    Multidimensional binary closing with a given structuring element.

    Like scipy.ndimage.binary_closing() except it
    * supports changing the way array borders are handled, and
    * does not support iterations, mask, or border_value (handle those yourself).

    Parameters
    ----------
    arr: array_like
        Binary image to be closed. Non-zero (True) elements form
        the subset to be closed.
    structure: array_like
        Structuring element used for the closing. Non-zero elements are
        considered True. If no structuring element is provided, a cross-shaped
        (i.e. center + immediate neighbors) element is used.
    out: ndarray
        Array of the same shape as arr, into which the output is placed.
        Defaults to creating a new array.
    mode: str
        ('reflect'|'constant'|'nearest'|'mirror'|'wrap')
        Determines what values are used outside the borders of arr.
        For 'constant' mode, they are set to cval.
        N.B.: using anything other than 'constant' and cval=0 will probably
        produce a nonintuitive result!  This function internally takes
        care to pad arr, so it is not just a dilation followed by an
        erosion.
    cval: float
        See mode
    origin : int or tuple of ints
        Placement of the filter.

    Returns
    -------
    binary_closing : ndarray of bools
        Closing of arr by the structuring element.
    """
    # conv is a dummy.
    binary, conv, structure = _check_img_and_selem(arr, structure)

    pw = np.asarray(structure.shape) // 2
    pad = [(p, p) for p in pw]
    binary = np.pad(binary, pad, mode=mode, constant_values=cval)
    tmp = binary_dilation(binary, structure, mode=mode, cval=cval,
                          origin=origin)
    tmp = binary_erosion(tmp, structure, out=out, mode=mode,
                         cval=cval, origin=origin)
    crop = tuple([slice(p, -p) for p in pw])
    return tmp[crop].copy()


def binary_dilation(arr, structure=None, out=None, mode='nearest', cval=0.0, origin=0):
    """
    Multidimensional binary dilation with a given structuring element.

    Like scipy.ndimage.binary_dilation() except it
    * supports changing the way array borders are handled, and
    * does not support iterations, mask, or border_value (handle those yourself).

    Parameters
    ----------
    arr: array_like
        Binary image to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure: array_like
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided, a cross-shaped
        (i.e. center + immediate neighbors) element is used.
    out: ndarray
        Array of the same shape as arr, into which the output is placed.
        Defaults to creating a new array.
    mode: str
        ('reflect'|'constant'|'nearest'|'mirror'|'wrap')
        Determines what values are used outside the borders of arr.
        For 'constant' mode, they are set to cval.
    cval: float
        See mode
    origin: int or tuple of ints
        Placement of the filter.

    Returns
    -------
    binary_dilation: ndarray of bools
        Dilation of arr by the structuring element.
    """
    binary, conv, structure = _check_img_and_selem(arr, structure)
    ndi.convolve(binary, structure, mode=mode, cval=cval, output=conv, origin=origin)
    if out is None:
        out = np.empty_like(binary, dtype=np.uint8)  # np.bool might be more logical.
    return np.not_equal(conv, 0, out=out)


def binary_erosion(arr, structure=None, out=None, mode='nearest', cval=0.0, origin=0):
    """
    Multidimensional binary erosion with a given structuring element.

    Like scipy.ndimage.binary_erosion() except it
    * supports changing the way array borders are handled, and
    * does not support iterations, mask, or border_value (handle those yourself).

    Parameters
    ----------
    arr: array_like
        Binary image to be eroded. Non-zero (True) elements form
        the subset to be eroded.
    structure: array_like
        Structuring element used for the erosion. Non-zero elements are
        considered True. If no structuring element is provided, a cross-shaped
        (i.e. center + immediate neighbors) element is used.
    out: ndarray
        Array of the same shape as arr, into which the output is placed.
        Defaults to creating a new array.
    mode: str
        ('reflect'|'constant'|'nearest'|'mirror'|'wrap')
        Determines what values are used outside the borders of arr.
        For 'constant' mode, they are set to cval.
    cval: float
        See mode
    origin : int or tuple of ints
        Placement of the filter, by default 0.

    Returns
    -------
    binary_erosion : ndarray of bools
    Erosion of arr by the structuring element.
    """
    binary, conv, structure = _check_img_and_selem(arr, structure)
    ndi.convolve(binary, structure, output=conv, mode=mode, cval=cval, origin=origin)
    if out is None:
        out = np.empty_like(binary, dtype=np.uint8)  # np.bool might be more logical.
    return np.equal(conv, np.sum(structure), out=out)


def binary_opening(arr, structure=None, out=None, mode='nearest', cval=0.0,
                   origin=0):
    """
    Multidimensional binary opening with a given structuring element.

    Like scipy.ndimage.binary_opening() except it
        * supports changing the way array borders are handled, and
        * does not support iterations, mask, or border_value (handle those yourself).

    Parameters
    ----------
    arr: array_like
        Binary image to be opened. Non-zero (True) elements form
        the subset to be opened.
    structure: array_like
        Structuring element used for the opening. Non-zero elements are
        considered True. If no structuring element is provided, a cross-shaped
        (i.e. center + immediate neighbors) element is used.
    out: ndarray
        Array of the same shape as arr, into which the output is placed.
        Defaults to creating a new array.
    mode: str
        ('reflect'|'constant'|'nearest'|'mirror'|'wrap')
        Determines what values are used outside the borders of arr.
        For 'constant' mode, they are set to cval.
    cval: float
        See mode
    origin : int or tuple of ints
        Placement of the filter.

    Returns
    -------
    binary_opening : ndarray of bools
        Opening of arr by the structuring element.
    """
    binary, conv, structure = _check_img_and_selem(arr, structure)
    tmp = binary_erosion(binary, structure, mode=mode, cval=cval,
                         origin=origin)
    return binary_dilation(tmp, structure, out=out, mode=mode,
                           cval=cval, origin=origin)


def bcut_from_rel(bvals, relbthresh=0.04):
    """
    Given a sequence of b values and a relative b threshhold between
    "undiffusion weighted" and "diffusion weighted", return the absolute b
    corresponding to the threshhold.
    """
    minb = np.min(bvals)
    maxb = np.max(bvals)
    return minb + relbthresh * (maxb - minb)


def calc_average_s0(data, bvals, relbthresh=0.02, bcut=None,
                    estimator=np.mean):
    if bcut is None:
        bcut = bcut_from_rel(bvals, relbthresh)
    return estimator(data[..., bvals <= bcut], axis=-1)


def cond_to_mask(seg, cond):
    mask = np.zeros_like(seg)
    mask[seg == cond] = 1
    return mask


def fill_holes(msk, aff, dilrad=-1, verbose=True, inplace=False):
    """
    Fill holes in mask, where holes are defined as places in (a possibly
    dilated) mask that are False and do not connect to the outside edge of
    mask's box.

    Parameters
    ----------
    msk: array like
        The binary mask to hole fill.  Will NOT be modified unless inplace is True.
    aff: array like
        The affine matrix of mask
    dilrad: float
        If > 0, temporarily dilate by this amount (in the units of aff)
        before hole filling to include regions that are almost or practically
        holes.
        N.B.: The dilation does not work well if 0 < dilrad < max pixel dimension,
              and no check is done for this!
    verbose: bool
        Chattiness.
    inplace: bool
        Iff True, modify msk in place.

    Output
    ------
    mask: array like
        The input mask with its holes filled.
    errval: 0 or Exception
        0: Everything went well
        1: Undiagnosed error
        Exception: an at least partially diagnosed error.

        Note that fill_holes does not *throw* an exception on failure,
        which can be often caused by missing skimage.morphology.reconstruction,
        so using verbose and/or errval is important to distinguish problems
        from a simple lack of holes.
    """
    errval = 1
    try:
        if inplace:
            mask = msk
        else:
            mask = msk.copy()
        if verbose:
            print("Filling holes")
        # Based on http://scikit-image.org/docs/dev/auto_examples/plot_holes_and_peaks.html
        # hmask = np.zeros(mask.shape)
        # for z in xrange(mask.shape[2]):
        #     seed = np.copy(mask[:, :, z])
        #     seed[1:-1, 1:-1] = 1
        #     hmask[:, :, z] = reconstruction(seed, mask[:, :, z], method='erosion')
        # #if verbose:
        # #    print "Filling holes in coronal slices"
        # for y in xrange(mask.shape[1]):
        #     seed = np.copy(mask[:, y, :])
        #     seed[1:-1, 1:-1] = 1
        #     hmask[:, y, :] += reconstruction(seed, mask[:, y, :], method='erosion')
        # #if verbose:
        # #    print "Filling holes in sagittal slices"
        # for x in xrange(mask.shape[0]):
        #     seed = np.copy(mask[x, :, :])
        #     seed[1:-1, 1:-1] = 1
        #     hmask[x, :, :] += reconstruction(seed, mask[x, :, :], method='erosion')
        # mask[hmask > 1] = 1

        if dilrad > 0:
            ball = make_structural_sphere(aff, dilrad)
            dmask = binary_closing(mask, ball)
        else:
            dmask = mask.copy()
        seed = dmask.copy()
        seed[1:-1, 1:-1, 1:-1] = 1
        hmask = reconstruction(seed, dmask, method='erosion')

        if dilrad > 0:
            # Remove dmask's dilation and leave just the holes,
            hmask = binary_erosion(hmask, ball)
            #hmask[dmask > 0] = 0
            # but replace dilation that was part of a hole.
            #hmask = binary_dilation(hmask, ball)

        mask[hmask > 0] = 1
        errval = 0
    except Exception as e:
        if verbose:
            print("Problem trying to fill holes:", e)
            print("...continuing anyway...")
        errval = e
    return mask, errval


def fill_axial_holes(arr):
    mask = arr.copy()
    for z in range(arr.shape[2]):
        seed = arr[..., z].copy()
        seed[1:-1, 1:-1] = 1
        hmask = reconstruction(seed, mask[..., z], method='erosion')
        mask[hmask > 0, z] = 1
    return mask, 0


def get_1_file_or_hurl(pat):
    """
    Find exactly 1 glob match for pat, or raise a ValueError.

    Parameters
    ----------
    pat: glob pattern

    Output
    ------
    The matching filename.
    """
    cands = glob(pat)
    if len(cands) != 1:
        if not cands:
            msg = "No files were found matching %s" % pat
        else:
            msg = "Multiple files matched %s:\n  %s" % (pat, "\n  ".join(cands))
        raise ValueError(msg)
    return cands[0]


def instaprint(msg, stream=sys.stdout):
    stream.write(msg + "\n")
    stream.flush()


def make_structural_sphere(aff, rad=None):
    """
    Returns a ball of radius rad.
    Like ndimage.generate_binary_structure, but allowing for anisotropic voxels.

    Parameters
    ----------
    aff: array-like
        Affine matrix.  N.B.: the voxel dimensions are taken strictly from the
        diagonal, so it may have errors with oblique affine matrices.
    rad: float or bool
        The radius of the ball in units of aff.
        If None, the largest voxel dimension will be used.
    """
    scales = np.abs([aff[i, i] for i in range(3)])
    maxscale = max(scales)
    if not rad:
        rad = maxscale
    if rad < maxscale:
        print("Warning!  rad, %f, is < the largest voxel scale, %f." % (rad, maxscale))

    cent = np.asarray(np.floor(rad / scales), int)
    shape = 2 * cent + 1
    output = np.zeros(shape, dtype=bool)
    r2 = rad**2

    # Loop over an octant
    for i in range(cent[0] + 1):
        xterm = (scales[0] * i)**2
        for j in range(cent[1] + 1):
            yterm = (scales[1] * j)**2
            for k in range(cent[2] + 1):
                zterm = (scales[2] * k)**2
                isin = xterm + yterm + zterm <= r2

                if isin:
                    # Reflect to all octants.  The meridians will be overdone, but oh well.
                    for isign in [-1, 1]:
                        for jsign in [-1, 1]:
                            for ksign in [-1, 1]:
                                output[cent[0] + isign * i,
                                       cent[1] + jsign * j,
                                       cent[2] + ksign * k] = isin
    return output


def remove_disconnected_components(mask, aff=None, dilrad=0, inplace=True, verbose=False,
                                   nkeep=1, weight=None):
    """
    Return a version of mask with all but the largest nkeep regions removed.

    Parameters
    ----------
    mask: array-like
        3D array which is true where there is a component, and 0 elsewhere.
    aff: None or 4x4 float array
        The voxel-to-world coordinate affine array of the mask.
        Must be given if dilrad > 0.
    dilrad: float
        Radius in aff's units (mm by the Nifti standard) to dilate by
        before determining how mask's voxels are connected.  This does not
        affect mask (even if inplace is True), just a copy used in finding
        connected components.  A moderate radius will ensure that tenuously
        tethered components are considered to be connected, but too large a
        radius will make everything appear connected.
    inplace: bool
        Whether to operate on mask in place or return a new array.
    verbose: bool
        Chattiness controller
    nkeep: None or int > 0
        Keep the nkeep largest components. If None it will be determined by Otsu
        thresholding the region sizes, optionally weighted by weight.
    weight: None or array-like
        An optional image with the same shape as mask which can be used to
        weight regions if nkeep is None. It is useful for preferring bright
        medium regions over large regions just above 0.

    Output
    ------
    mask: array-like
        3D array which is true for the largest component, and 0 elsewhere.
    """
    if verbose:
        print("Removing disconnected components")
    cmask = mask.copy()
    if aff is None:
        aff = np.eye(4)
    if dilrad:
        ball = make_structural_sphere(aff, dilrad)
        # Closing isn't really necessary since mask itself is not dilated.
        cmask = ndi.morphology.binary_dilation(cmask, ball)
    labelled, nb_labels = ndi.label(cmask)
    if verbose:
        print("Found %d components" % nb_labels)
    del cmask

    labels = np.arange(1, nb_labels + 1)
    if weight is None:
        weight = np.ones(mask.shape)
    sizes = np.array([weight[labelled == label].sum() for label in labels])
    if nkeep is None:
        if len(sizes) > 1:
            thresh = otsu(sizes)
            nkeep = max(1, len(sizes[sizes > thresh]))
        else:
            nkeep = 1
    keep_indices = np.argpartition(sizes, -nkeep)[-nkeep:]  # O(n), requires numpy >= 1.8

    if inplace:
        mymask = mask
    else:
        mymask = mask.copy()
    mymask[mask > 0] = 0
    for ki in keep_indices:
        mymask[labelled == labels[ki]] = 1
    return mymask


def suggest_number_of_processors(fraction=0.25):
    """
    Crudely guess at how many jobs can be run in parallel without annoying
    other users (or yourself).  I accept no liability if they get annoyed
    anyway.

    Parameters
    ----------
    fraction: float between 0 and 1.
        How much of the server you want to take over.
        0 will yield 1 CPU, 0.99 will yield n_cpus - 1 unless n_cpus > 100.

    Returns
    -------
    n: int >= 1
        The suggested number of jobs to run in parallel.
    """
    return max(1, int(fraction * cpu_count()))


def voxel_sizes(aff):
    """
    Get the lengths of a voxel along each axis.

    Parameters
    ----------
    aff: (n + 1) x (n + 1) array, where n is the number of dimensions.
    Typically a NiFTI affine array in mm.

    Output
    ------
    vs: n x 1 array
    The lengths of a voxel along each axis, in the units of aff.

    Example
    -------
    # n3aff = n3amnii.get_affine()
    >>> n3aff = np.array([[ 1.19919360, 0.00114195282,   0.0366344191, -110.058807],
    [-0.0119082807, 0.970612407,   0.240443468,  -159.362793],
    [-0.0423398949, -0.240645453,  0.969971597,   -73.272484],
    [  0,           0,             0,               1])
    >>> scales = voxel_sizes(n3aff)
    >>> print scales
    [ 1.1999999   0.99999999  1.00000002]
    """
    n = aff.shape[0] - 1
    I = np.eye(n + 1)      # noqa
    return np.array([np.linalg.norm(np.dot(aff, I[i])) for i in range(n)])


def _test():
    """
    Run the doctests of this module.
    """
    import doctest
    from . import utils
    return doctest.testmod(utils)
