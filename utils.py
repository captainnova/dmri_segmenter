from glob import glob
from multiprocessing import cpu_count
import numpy as np
import scipy.ndimage as ndi

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
    if np.product(structure.shape,axis=0) < 1:
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

    # I mean // 2 here, but I'm worried that a from __future__ import division
    # could cause problems.
    pw = np.asarray(structure.shape) / 2

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
    scales = np.abs([aff[i, i] for i in xrange(3)])
    maxscale = max(scales)
    if not rad:
        rad = maxscale
    if rad < maxscale:
        print "Warning!  rad, %f, is < the largest voxel scale, %f." % (rad, maxscale)
        
    cent = np.asarray(np.floor(rad / scales), int)
    shape = 2 * cent + 1
    output = np.zeros(shape, dtype=bool)
    r2 = rad**2

    # Loop over an octant
    for i in xrange(cent[0] + 1):
        xterm = (scales[0] * i)**2
        for j in xrange(cent[1] + 1):
            yterm = (scales[1] * j)**2
            for k in xrange(cent[2] + 1):
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

def remove_disconnected_components(mask, aff=None, dilrad=0, inplace=True, verbose=False):
    """
    Return a version of mask with all but the largest component removed.

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

    Output
    ------
    mask: array-like
        3D array which is true for the largest component, and 0 elsewhere.
    """
    if verbose:
        print "Removing disconnected components"
    cmask = mask.copy()
    if aff is None:
        aff = np.eye(4)
    if dilrad:
        ball = make_structural_sphere(aff, dilrad)
        # Closing isn't really necessary since mask itself is not dilated.
        cmask = ndi.morphology.binary_dilation(cmask, ball)
    labelled, nb_labels = ndi.label(cmask)
    if verbose:
        print "Found %d components" % nb_labels
    del cmask
    # Find the label with the largest volume
    maxvox = 0
    blabel = 0
    for label in xrange(1, nb_labels + 1):
        nvox = len(labelled[labelled == label])
        if nvox > maxvox:
            maxvox = nvox
            blabel = label
    if inplace:
        mymask = mask
    else:
        mymask = mask.copy()
    mymask[labelled != blabel] = 0
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
    n_cpus = multiprocessing.cpu_count()
    return max(1, int(fraction * n_cpus))

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
    I = np.eye(n + 1)
    return np.array([np.linalg.norm(np.dot(aff, I[i])) for i in xrange(n)])

def _test():
    """
    Run the doctests of this module.
    """
    import doctest
    import utils
    return doctest.testmod(utils)
