from glob import glob
from multiprocessing import cpu_count
import numpy as np

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
