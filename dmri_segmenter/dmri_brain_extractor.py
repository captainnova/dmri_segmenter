from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
import datetime
#from icecream import ic
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as ndi
from scipy.ndimage.filters import median_filter
import scipy.special
#import sklearn.externals.joblib as joblib
try:
    from skimage.filter import threshold_otsu as otsu
except ImportError:
    from dipy.segment.threshold import otsu
from skimage.segmentation import morphological_geodesic_active_contour
try:
    import six
except ImportError:
    import dipy.utils.six as six
import socket
import sys
import warnings

try:
    from .classifier_io import choose_default_clf, load_classifier
    from .FLAIRity import FLAIRity
    from . import utils
except Exception:                         # I'm not sure this ever gets used anymore.
    from classifier_io import choose_default_clf, load_classifier
    from FLAIRity import FLAIRity
    import utils

# A combination of semantic versioning and the date. I admit that I do not
# always remember to update this, so use get_version_info() to also try to
# get the most recent commit message.
__version__ = "2.3.0 20240627"


def get_version_info(lcmfn="last_commit_message"):
    """
    Returns a str blurb identifying the version of this file, and if available,
    the last commit message.

    TODO: Replace this with a reading a file written by a (pre-)commit hook,
    since git >= 2.35 refuses to read other people's git repos unless the
    target repo is marked safe. Apparently there is a security risk wheren
    git would read the other repo's configuration and could be tricked into
    doing something nefarious.
    """
    try:
        filename = __file__.replace('.pyc', '.py')
    except Exception:
        filename = "dmri_brain_extractor.py"
    vinfo = filename + " version: " + __version__ + "\n"

    try:
        gitdir = os.path.dirname(os.path.dirname(filename))
        if not os.path.isabs(lcmfn):
            lcmfn = os.path.join(gitdir, lcmfn)
        if os.path.exists(lcmfn):
            with open(lcmfn) as f:
                lcm = f.read()
            vinfo += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            vinfo += "The most recent dmri_segmenter commit was:\n"
            vinfo += lcm
            vinfo += "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
    except Exception as e:
        # I don't think it's really worth worrying anyone if we can't get the
        # last commit message.
        warnings.warn("Trying to get the repository info for %s failed with:\n\t%s\n"
                      % (filename, e), RuntimeWarning, stacklevel=1)
    return vinfo


def save_mask(arr, aff, outfn, exttext='', outtype=np.uint8):
    """
    Saves a numpy array to a nii.
    """
    mnii = nib.Nifti1Image(arr.astype(outtype), aff)
    if exttext:
        mnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                                 exttext.encode('utf-8')))
    nib.save(mnii, outfn)


def get_brain_and_tiv_from_FLAIR_dMRI(dilate_before_chopping, maxscale, flairness, aff, verbose, dilate,
                                      nmed, medrad, whiskradinvox, trim_whiskers, closerad):
    if dilate_before_chopping >= 1:
        dilrad = dilate_before_chopping * maxscale
    else:
        dilrad = 0
    mask = utils.remove_disconnected_components(flairness.mask, aff, dilrad, verbose=verbose)

    if dilate:
        # Dilate once with a big ball instead of multiple times with a small
        # one since the former will better approximate a sphere.
        if dilate is True:
            dilate = nmed * medrad
        if dilate >= 1:
            dilrad = dilate * maxscale
            if verbose:
                print("Dilating with radius %f." % dilrad)
            ball = utils.make_structural_sphere(aff, dilrad)
            mask = utils.binary_dilation(mask, ball)
        else:
            mask = utils.binary_dilation(mask)

    # In principle the brain is a hole in the CSF (depending on how you cut off
    # the spine), but in practice this is just a waste of time - it's better to
    # include dark brain voxels by accepting nonCSF voxels that are mostly
    # surrounded by known brain tissue.
    # csfholes, success = utils.fill_holes(flairity.csfmask, aff, closerad * maxscale, verbose)
    # csfholes[flairity.csfmask > 0] = 0
    # mask[csfholes > 0] = 1
    # mask, success = utils.fill_holes(mask, aff, closerad * maxscale, verbose)

    # Trim any whiskers caused by susceptibility problems.
    whiskrad = whiskradinvox * maxscale
    ball = utils.make_structural_sphere(aff, whiskrad)

    omask = utils.binary_closing(mask, ball)  # Fill in sulci
    omask = utils.binary_opening(omask, ball)
    omask = utils.remove_disconnected_components(omask, aff, 0)
    omask = utils.binary_dilation(omask, ball)
    whiskers = mask.copy()
    whiskers[omask > 0] = 0  # Anything in mask that isn't within whiskrad of the opened mask.
    whiskers = utils.binary_dilation(whiskers, ball)
    #
    # Remove anything within whiskrad of the undilated whiskers, to account for the gap
    # made by dilating omask.
    if verbose:
        print("Removing %d whisker voxels" % sum(mask[whiskers > 0]))
    mask[whiskers > 0] = 0
    mask = utils.remove_disconnected_components(mask, aff, dilrad, verbose=verbose)

    # Now, to get very dark voxels (putamen), close, and then remove CSF.
    tiv = utils.binary_closing(mask, ball)

    if trim_whiskers:
        # More whisker removal
        gaprad = max(closerad, 2 * maxscale)
        ball = utils.make_structural_sphere(aff, gaprad)
        omask = utils.binary_dilation(tiv, ball)
        omask, success = utils.fill_holes(omask, aff, gaprad, verbose)
        flairness.csfmask[omask == 0] = 0
        tiv[flairness.csfmask > 0] = 1
        tiv, success = utils.fill_holes(tiv, aff, 0, verbose)
        #ball = utils.make_structural_sphere(aff, dilrad)
        tiv = utils.binary_erosion(tiv, ball)
        tiv = utils.remove_disconnected_components(tiv, aff, 0, verbose=verbose)
        tiv = utils.binary_dilation(tiv, ball)
        tiv[omask == 0] = 0
        #tiv = utils.binary_closing(tiv, ball)
    return mask, tiv


def _print_flair_msg(isFLAIR, flairness, verbose):
    if isFLAIR is None:
        if flairness.flairity:
            flair_msg = "This appears"
        else:
            flair_msg = "This does not appear"
        flair_msg += " to be a FLAIR acquisition"
    else:
        flair_msg = "Treating this as a "
        if not isFLAIR:
            flair_msg += "non-"
        flair_msg += "FLAIR scan as instructed."
    if verbose:
        print(flair_msg)
    return flair_msg


def get_dmri_brain_and_tiv(data, ecnii, brfn, tivfn, bvals, relbthresh=0.04,
                           medrad=1, nmed=2, verbose=True, dilate=True,
                           dilate_before_chopping=1, closerad=3.7,
                           whiskradinvox=4, Dt=0.0007, DCSF=0.003,
                           isFLAIR=None, trim_whiskers=True,
                           svc=None, t1wtiv=None):
    """
    Make a brain and a TIV mask from 4D diffusion data.

    It selects voxels based on a minimum threshold (not air), maximum
    diffusivity (not CSF), and location relative to other brain voxels
    (e.g. hole filling).

    Parameters
    ----------
    data: array-like
        The 4D data.
    ecnii: array-like
        The opened nii instance of the data (data = ecnii.get_fdata())
    brfn: string
        Filename for the brain mask.  If blank, no file will be
        written.
    tivfn: string
        Filename for the TIV mask.  If blank, no file will be
        written.
    bvals: array  (NOT a list!)
        The diffusion strengths for each volume in data, nominally in
        s/mm**2.  (Adjust Dt and DCSF if using different units.)
    relbthresh: float
        Threshold between b0s and diffusion volumes, as a fraction of
        max(bvals).
    medrad: float
        Radius of the median filter relative to the largest voxel size.
    nmed: int
        Number of times to run the median filter
    verbose: Boolean
        Make it chattier.
    dilate: Bool or float
        If >= 1, dilate with this radius (relative to the largest voxel size).
        If True, use medrad * nmed.
        If between 0 and 1, dilate by a voxel.
        N.B.: It only affects FLAIR DTI!
    dilate_before_chopping: float
        If >= 1, dilate _the_copy_used_for_finding_the_largest_connected_component_
        with this * the largest voxel size before removing disconnected components.
        1 is highly recommended - less can chop off important things, and more can
        connect unwanted things like eyeballs and jowls.
    closerad: float
        A radius in mm used to close gaps when needed.  Effectively at least 2 * maxscale.
    whiskradinvox: float
        Scale relative to the largest voxel dimension of data.
    Dt: float
        The nominal diffusivity of tissue in reciprocal units of bvals.
        Depends on temperature.
    DCSF: float
        The nominal diffusivity of CSF in reciprocal units of bvals.
        Depends on temperature, and can be artificially depressed in scans where
        the b0 CSF is brighter than the maximum int value.
    isFLAIR: None or bool
        Whether or not data is a FLAIR diffusion acquisition.  If None, try to
        determine it from CSF/tissue brightness ratio by doing a (slow)
        preliminary segmentation.
    svc : Optional, None, str, or dict
        The classifier as a directory or filename, or a dict.
        If None, classifier_io.choose_default_clf() will look for one in onnx
        format if you have onnx installed, or as a pickle otherwise.
        (Pickles are liable to break when libraries change. Otherwise
         there is no difference between the versions.)
    t1wtiv : Optional, str, Path, or None
        Filename of a .nii to load the undistorted (e.g. T1w) intracranial
        mask from. It must be in the same space as data, and is only used as a
        blurry prior.

    Outputs
    -------
    mask: array-like
        3D array which is 1 where there should be brain, and 0 elsewhere.
    tiv: array-like
        3D array which is 1 inside the braincase and 0 outside.
    """
    if svc is None:
        svc = choose_default_clf()
    #ic(svc)

    aff = ecnii.affine
    # for d in range(3):
    #     # Accept up to pi/4 obliqueness.
    #     if aff[d, d] < 0.70711 * np.linalg.norm(aff[d, :3]):
    #         # The problem seems to be in get_data(), not nib.save()
    #         print """
    #         Uh Oh: The input data is flipped or rotated by >= pi/4, which would
    #         cause inconsistencies in the output. Flip the input around first with
    #         fslreorient2std or removeFlipsRots.
    #         """
    #         return None

    if data is None:
        data = ecnii.get_fdata()
    if len(data.shape) != 4:
        raise ValueError("the input must be 4 dimensional")

    if len(bvals) != data.shape[-1]:
        raise ValueError("the length of bvals, %d, does not match the number of data volumes, %d" %
                         (len(bvals), data.shape[-1]))
    # b0 = utils.calc_average_s0(data, bvals, relbthresh, estimator=np.median)
    scales = utils.voxel_sizes(aff)
    maxscale = max(scales)
    #ic(scales)

    flairness = FLAIRity(data, aff, bvals, relbthresh, maxscale, Dt, DCSF, nmed, medrad,
                         verbose=verbose, closerad=closerad, forced_flairity=isFLAIR)
    flair_msg = _print_flair_msg(isFLAIR, flairness, verbose)

    if not flairness.flairity:
        #ic("Starting feature_vector_classify")
        mask, csfmask, other, submsg = feature_vector_classify(data, aff, bvals, clf=svc, t1wtiv=t1wtiv)
        #ic("Finished feature_vector_classify")
        tiv = mask + csfmask + other
    else:
        mask, tiv = get_brain_and_tiv_from_FLAIR_dMRI(dilate_before_chopping, maxscale, flairness, aff, verbose, dilate,
                                                      nmed, medrad, whiskradinvox, trim_whiskers, closerad)
        submsg = ""
    exttext  = "Mask made " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    exttext += """
    by get_dmri_brain_and_tiv(data, %s, brfn, tivfn,
                              bvals=%s,
                              relbthresh=%f,
                              medrad=%f,
                              nmed=%d,
                              verbose=%s,
                              dilate=%s,
                              dilate_before_chopping=%s,
                              closerad=%f,
                              whiskradinvox=%f,
                              Dt=%f,
                              DCSF=%f,
                              trim_whiskers=%s,
                              svc=%s)
    """ % (ecnii.get_filename(), bvals, relbthresh, medrad, nmed,
           verbose, dilate, dilate_before_chopping, closerad,
           whiskradinvox, Dt, DCSF, trim_whiskers, svc)
    exttext += "on %s,\n" % socket.gethostname()
    exttext += "using python %s,\n" % sys.version
    exttext += get_version_info()
    exttext += flair_msg + "\n" + submsg

    if brfn:
        if flairness.flairity:
            mask[tiv > 0] = 1  # Recover dark voxels in the right place
            mask[flairness.csfmask > 0] = 0

            # Remove blips and whiskers
            ball = utils.make_structural_sphere(aff, maxscale)
            mask = utils.binary_opening(mask, ball)
            mask = utils.remove_disconnected_components(mask, aff, 0)

            mask = utils.binary_dilation(mask)  # Recover partial CSF voxels.
            mask[tiv == 0] = 0                  # but clamp to the TIV.
        save_mask(mask, aff, brfn, exttext)
    if tivfn:
        save_mask(tiv, aff, tivfn, exttext)
    if verbose:
        # Useful if running in the background on a terminal.
        print("get_dmri_brain_and_tiv finished")
    return mask, tiv


def make_mean_adje(data, bvals, relbthresh=0.04, s0=None, Dt=0.00210, Dcsf=0.00305,
                   blankval=0, clamp=30, regscale=0.07):
    """
    Makes an average over volume number of data / s0
    * weighted by the expected squared CNR between tissue and CSF and
    * divided by the expected value of data / s0 for tissue, so data
      with different b can be averaged together, and the expected value
      for tissue is 1.

    Parameters
    ----------
    data: (nx, ny, nz, nv) array
    bvals: (nv,) array
    relbthresh: float
        The cutpoint between b0s and DWIs, as a fraction of max(bvals).
        Only used if s0 is None.
    s0: None or (nx, ny, nz) array
        The average of the b=0 volumes.  Will be made if None.
    Dt: float
        The intrinsic diffusivity of periaxonal water parallel to the axons,
        in reciprocal units of bvals.
    Dcsf: float
        The CSF diffusiivity in reciprocal units of bvals.
    blankval: float
        Set the output to this wherever s0 is 0.
    clamp: float
        Typically some voxels will have ratios of small numbers / tiny numbers,
        and be ridiculously and pointlessly large.  Since they can easily exceed
        the data range supported by fslview, it is best to clip them at +- some
        value well away from 1.  Set to None if you really do not want to clamp.
    regscale: float
        Avoid division by 0 by adding this * median(brain brightness) to the signal
        in quadrature.

    Output
    ------
    madje: (nx, ny, nz) array
        Specially weighted average over volume number of data / s0
    brightness_scale: float
        The median of s0[(madje * s0) > otsu(madje * s0)]
    """
    if s0 is None:
        s0 = utils.calc_average_s0(data, bvals, relbthresh)
    noise = np.median(s0[s0 < otsu(s0)])
    rbl = (bvals[bvals > 0] * Dt)**0.5
    et = np.ones(len(bvals))
    et[bvals > 0] = 0.5 * np.pi**0.5 * scipy.special.erf(rbl) / rbl
    ecsf = np.exp(-bvals * Dcsf)
    w = et * (et - ecsf)**2

    # np.average sometimes gives a memory error, so don't use it.
    madje = np.zeros(data.shape[:-1])
    w /= w.sum()
    for v, etv in enumerate(et):
        vw = w[v] / etv
        if vw > 0:
            db2 = data[..., v]**2 - noise**2
            db2[db2 < 0] = 0
            madje += vw * db2**0.5
    #madje = np.average(dataoet, axis=-1, weights=w)

    thresh = otsu(madje)
    brightness_scale = np.median(s0[madje > thresh])
    regularizer = regscale * brightness_scale
    madje *= ((1.0 + (noise / brightness_scale)**2) / (s0**2 + regularizer**2))**0.5

    madje[s0 == 0] = blankval
    madje[np.isnan(madje)] = blankval
    if clamp is not None:
        madje = np.clip(madje, -clamp, clamp, madje)
    return madje, brightness_scale


def seal_edges(src, mask, aff, dilation):
    """
    Treating the TIV as the biggest hole inside the gradient defined edge
    is great, except when there is a "leak" to the outside; then it is
    disastrous. Such leaks can happen when the FOV is placed badly, or when
    eddy zeros the top and bottom slices.

    If mask touches an edge, that edge is closed using dilation and then
    2D hole filled. Otherwise, that edge is completely filled with ones
    (the eddy case).

    Parameters
    ----------
    src : 3D array, nominally float
       The source image that mask was made from. An edge is only completely
       filled with ones if it is all zeroes in *src* (the eddy case). src is
       used just as a check of where the input data for mask is valid.
    mask : 3D bool array
        The mask to be copied and sealed on its top and bottom,
        left and right, and front and back sides.
    aff : 4x4 affine array
    dilation : float
        The dilation radius, in aff's units, used to close mask in 2D
        before hole filling its edges.

    Returns
    -------
    smask : 3D bool array
        Edge sealed mask
    """
    selector = [slice(None)] * 3
    smask = mask.copy()
    nfilled = 0

    for ax in range(2, -1, -1):    # Deliberately start with z because we might have to stop before x.
        disk = utils.make_disk(aff, ax, dilation)

        end = mask.shape[ax] - 1
        for ind in (0, end):
            slicer = selector[:]   # Py2 doesn't have list.copy()
            slicer[ax] = ind
            slicer = tuple(slicer)
            if (src[slicer] == 0).all():
                if nfilled < 5:
                    smask[slicer] = True       # Suspiciously zeroed (by eddy?)
                nfilled += 1
            else:
                smask[slicer], _ = utils.fill_holes(smask[slicer], aff, verbose=False, inplace=True,
                                                    ball=disk)
    if nfilled == 6:
        warnings.warn("All of the edges were zeros - edge sealing on the 0th axis was disabled to prevent filling in the entire box.",
                      RuntimeWarning, stacklevel=2)
    return smask
    

def grad_based_tiv_from_pd(pd, aff, sigma, norm, compscale, ball, relthresh=0.5,
                           ncomponents=1, is_phantom=False, pd_bias=0.5):
    """
    Make a TIV mask which is mostly based on finding the edge of a proton
    density approximation.

    Parameters
    ----------
    pd : 3D array of floats
        The proton density approximation
    aff : 4 x 4 array of floats
        The affine matrix
    sigma : float
        Blurring radius for gaussian_gradient_magnitude, in aff's units.
    norm : 3D array of floats
        Blurred pd or similar to account for brightness shading
    compscale : float
        Dilation radius in aff's units for filling in edges
    relthresh : float
        Factor to lower the Otsu(grad) threshold by.
    ncomponents : int
        Keep the ncomponents largest components
    is_phantom : bool
    pd_bias : float
        Bias the gradient by pd_bias * relthresh * Otsu(grad) * edge(thresholded pd)
        to fill in susceptibility blurred holes and deemphasize edges that are
        distant from the brain.
    """
    # Use mode='constant' (with implied cval=0) so that the TIV is capped.
    grad = ndi.gaussian_gradient_magnitude(pd, sigma, mode='constant') / norm

    thresh = relthresh * otsu(grad)

    # Sometimes susceptibility blurs so much that it leaves holes in the
    # gradient based TIV edge, causing hole filling below to fill the wrong
    # hole. We know that thresholding pd is approximately correct, so let it help.
    pdmask = np.zeros(pd.shape, dtype=np.uint8)
    pdmask[pd > otsu(pd)] = 1
    dpdmask = utils.binary_dilation(pdmask, ball)
    #
    # Deemphasize edges far away from pd, because there might be large voids
    # inside the head that *aren't* the TIV and/or leaks to the air, and we
    # need the largest hole step below to work.
    bgrad = grad.copy()
    bgrad[dpdmask == 0] -= pd_bias * thresh
    #
    # Emphasize the part right outside pd.
    dpdmask[pdmask > 0] = 0
    bgrad[dpdmask > 0] += pd_bias * thresh
    
    gradmask = np.zeros(pd.shape, dtype=np.uint8)
    gradmask[bgrad > thresh] = 1

    segradmask = seal_edges(pd, gradmask, aff, 3.0 * compscale)
    fgmask, _ = utils.fill_holes(segradmask, aff, 3.0 * compscale, verbose=False)
    if is_phantom:
        # Remove the "outside" part of gradmask.
        fgmask = utils.binary_erosion(fgmask, ball)
    else:
        # We need to remove the eyeballs, so remove all of segradmask.
        fgmask[segradmask > 0] = 0
    fgmask = utils.binary_opening(fgmask, ball)
    utils.remove_disconnected_components(fgmask, inplace=True, nkeep=ncomponents,
                                         weight=pd)
    fgmask = utils.binary_dilation(fgmask, ball)
    fgmask = utils.binary_closing(fgmask, ball)
    sefgmask = seal_edges(pd, fgmask, aff, 3.0 * compscale)
    edges = sefgmask.copy()
    edges[fgmask > 0] = 0
    fgmask, success = utils.fill_holes(sefgmask, aff, verbose=False)
    fgmask[edges > 0] = 0

    if fgmask.sum() == 0:
        msg =  "Warning: grad_based_tiv_from_pd returned an empty mask. There was probably\n"
        msg += "a hole due to susceptibility blurring out the edge. Falling back to a mask\n"
        msg += "made by thresholding a proton density approximation.\n"
        print(msg)
        fgmask[pd > otsu(pd)] = 1

    return fgmask, grad


def mgac_improve_mask(mask, grad, pd, aff, voxscale, thresh=None, npedvox=3, pedax=1,
                      niters=10, balloon=0, alpha=100.0):
    """
    Improve mask using geodesic active contours implemented with morphological
    operators. It can be used to segment objects with visible but noisy,
    cluttered, broken borders.

    Parameters
    ----------
    mask: 3D array
        A reasonably good approximation of the wanted mask.
    grad: 3D array
        |grad(pd)|
    pd: 3D array
        Grayscale image to be masked
    aff: 4x4 array
        Affine transform from voxel to world coordinates.
    voxscale: float
        The largest extent of a voxel, in aff's units (e.g. mm).
    thresh: None or float
        Voxels brighter than this in pd and neighboring the intermediate mask
        will be included. If None, median(pd[mask > 0]) will be used.
    npedvox: int
        Expected thickness of pileup regions, in voxels.
    pedax: int
        Axis of the phase encoding direction; 0: x, 1: y, 2: z.
    niters: int
        Number of morphological_geodesic_active_contour iterations to use.
    balloon: float
        Balloon force to guide the contour in non-informative areas of the
        image, i.e., areas where the gradient of the image is too small to push
        the contour towards a border. A negative value will shrink the contour,
        while a positive value will expand the contour in these areas. Setting
        this to zero will disable the balloon force.
        Attractive as it sounds, it does not have much effect, at least with
        balloon = 1. I could not find what the scale should be.
    alpha: float
        Controls how deep the troughs of 1.0 / np.sqrt(1.0 + alpha * grad) are,
        with higher values making it nominally more sensitive to grad.

    Returns
    -------
    mgac: 3D array
        mask after improving it with morphological_geodesic_active_contour(),
        inclusion of neighboring signal pileups, and hole filling.
    """
    gimage = 1.0 / np.sqrt(1.0 + alpha * grad)
    mgac = morphological_geodesic_active_contour(gimage, niters, init_level_set=mask, balloon=balloon)

    # mgac might exclude regions of signal pileup near the edge. Add bright
    # regions that could be EPI distorted parts of the TIV.
    pedshape = [1, 1, 1]
    pedshape[pedax] = 1 + npedvox
    pedline = np.ones(pedshape)
    dmgac = utils.binary_dilation(mgac, pedline)
    if thresh is None:
        thresh = np.median(pd[mgac > 0])
    dmgac[pd < thresh] = 0
    dmgac[mgac > 0] = 1
    dmgac = utils.remove_disconnected_components(dmgac)
    dmgac, _ = utils.fill_holes(dmgac, aff, 2 * voxscale)
    return dmgac


def make_grad_based_TIV(s0, madje, aff, softener=0.2, dr=2.0, relthresh=0.5,
                        ncomponents=1, is_phantom=False, pd=None):
    """
    The edge of the TIV is usually apparent to the eye even when a bad bias
    field precludes using a constant intensity threshold to pick out the TIV,
    and the edge can (with care) be normalized to mostly remove the effect of
    the bias field.

    Parameters
    ----------
    s0: array
        The b ~ 0 brightness, already divided by brightness_scale
    madje: array
        Weighted average of the b > 0 volumes divided by s0.
    softener: float
        Prevents division by 0.  The smaller the number the more the effect of
        the bias field is removed, but at the cost of amplifying noise.
    dr: float, in aff's units.
        The nominal change in position to use for calculating the gradient.
        It will be automatically adjusted to account for the actual voxel
        size.
    ncomponents: None or int > 0
        The number of separate regions to keep, sorted by s0-weighted size.
        If None it will be determined by Otsu thresholding.
    is_phantom: bool
    pd : Optional, None or array
       An approximation of a proton (specifically, hydrogen) density image.
       If None, one will be calculated using s0 and madje.
    """
    if pd is None:
        # Approximate a proton density image by making the CSF and tissue more
        # isointense.
        pd = s0 * (1 + 2 * madje)

    voxdims = utils.voxel_sizes(aff)

    # Make dr a suitable compromise between the nominal scale and the actual
    # voxel size.  Remember that bigger voxels have more partial volume
    # dilution, so some stretching is needed for them.
    maxscale = max(voxdims)
    compscale = np.sqrt(0.5 * (dr**2 + maxscale**2))
    ball = utils.make_structural_sphere(aff, 2.0 * compscale)
    sigma = compscale / voxdims

    norm = ndi.gaussian_filter(pd, sigma, mode='nearest')
    intensity_scale = np.std(pd)
    norm = np.sqrt(norm**2 + (softener * intensity_scale)**2)

    fgmask, grad = grad_based_tiv_from_pd(pd, aff, sigma, norm, compscale, ball,
                                          relthresh, ncomponents, is_phantom)

    # Signal pileup from EPI distortion should not count against the TIV, so clamp it
    # using the estimate so far of normal brain PD.
    med_in_mask = np.median(pd[fgmask > 0])
    np.clip(pd, 0, med_in_mask, pd)
    fgmask, grad = grad_based_tiv_from_pd(pd, aff, sigma, norm, compscale, ball,
                                          relthresh, ncomponents, is_phantom)

    return mgac_improve_mask(fgmask, grad, pd, aff, maxscale, med_in_mask)


def make_feature_vectors(data, aff, bvals, relbthresh=0.04, smoothrad=10.0, s0=None,
                         Dt=0.0021, Dcsf=0.00305, blankval=0, clamp=30,
                         normslop=0.4, logclamp=-10, use_grad=True):
    """
    Make 4D vectors for segmenting data.

    Parameters
    ----------
    data: (nx, ny, nz, nv) array
        Non-FLAIR diffusion data
    aff: (4, 4) array
        Affine matrix for data
    bvals: (nv,) array
    relbthresh: float
       The cutpoint between b0s and DWIs, as a fraction of max(bvals).
       Only used if s0 is None.
    smoothrad: float
        The smoothing radius in mm.
        Since it is a parameter of training data, do not change it!
    s0: None or (nx, ny, nz) array
        The average of the b=0 volumes.  Will be made if None.
    Dt: float
        The intrinsic diffusivity of periaxonal water parallel to the axons,
        in reciprocal units of bvals.
    Dcsf: float
        The CSF diffusivity in reciprocal units of bvals.
    blankval: float
        Set the output to this wherever s0 is 0.
    clamp: float
        Typically some voxels will have ratios of small numbers / tiny numbers,
        and be ridiculously and pointlessly large.  Since they can easily exceed
        the data range supported by fslview, it is best to clip them at +- some
        value well away from 1.  Set to None if you really do not want to clamp.
    normslop: float
        s0 needs to be normalized before it can be used with an SVM, so it is
        divided by median(s0[np.abs(madje - 1) < normslop]), where madje comes
        from make_mean_adje().
        It should be large enough to include most brain voxels, but small enough
        to exclude most non-brain voxels.
        Since it is a parameter of training data, do not change it!
    logclamp: float
        Feature vector values < this (in log10 space) will be set to this.
    use_grad: bool
        The edge of the s0 + 2 madje approximate PD image makes a fairly good
        boundary for the TIV, except on the inferior side, so blur it to make
        a prior.

    Output
    ------
    fvecs: (nx, ny, nz, 4) array
        The classification quantities to use for SVM segmentation.
        fvecs[..., 0]: s0 / median(s0[np.abs(madje - 1) < normslop])
        fvecs[..., 1]: "neighborhood s0", i.e. fvecs[..., 0] smoothed with a top
                       hat of radius smoothrad.
        fvecs[..., 2]: madje from make_mean_adje().
        fvecs[..., 3]: "neighborhood madje", i.e. fvecs[..., 2] smoothed with a top
                       hat of radius smoothrad.
    posterity: str
        Information about how fvecs was made.
    """
    if s0 is None:
        s0 = utils.calc_average_s0(data, bvals, relbthresh)
    if use_grad:
        nfeatures = 5
    else:
        nfeatures = 4                                   # l2amp looks helpful, but empirically it isn't.

    fvecs = np.empty(data.shape[:3] + (nfeatures,))
    fvecs[..., 2], brightness_scale = make_mean_adje(data, bvals, s0=s0, Dt=Dt, Dcsf=Dcsf,
                                                     blankval=blankval, clamp=clamp)

    fvecs[..., 0] = s0 / brightness_scale

    ball = utils.make_structural_sphere(aff, smoothrad)
    median_filter(fvecs[..., 2], footprint=ball, output=fvecs[..., 3], mode='nearest')
    median_filter(fvecs[..., 0], footprint=ball, output=fvecs[..., 1], mode='nearest')

    if use_grad:
        # Before blurring the grad based TIV, close it to avoid demphasizing
        # susceptibility horns.
        gbtiv = make_grad_based_TIV(fvecs[..., 0], fvecs[..., 2], aff)
        sigma = smoothrad / utils.voxel_sizes(aff)
        gbtiv = utils.binary_closing(gbtiv, ball)
        fvecs[..., 4] = ndi.gaussian_filter(gbtiv.astype(float), sigma, mode='nearest')

    # if nfeatures > 4:
    #     l2amp = sfa.calc_l2_amp(data, bvals, bvecs, s0=s0, nonorm=True)
    #     fvecs[..., 4] = l2amp
    #     #ml2amp = ndi.filters.gaussian_filter(l2amp, 0.5 * smoothrad, mode='nearest')
    #     #adev = ndi.filters.gaussian_filter(np.abs(l2amp - ml2amp), 0.5 * smoothrad, mode='nearest')
    #     #fvecs[..., 5] = np.sqrt(adev + ml2amp)
    #     median_filter(np.sqrt(l2amp), footprint=ball, output=fvecs[..., 5],
    #     mode='nearest')

    tlogclamp = 10**logclamp
    fvecs[..., :4] = np.log10(np.clip(fvecs[..., :4], tlogclamp, 1.0 / tlogclamp, fvecs[..., :4]))

    posterity  = "Feature vectors made with:\n"
    posterity += "\trelbthresh = %f\n" % relbthresh
    posterity += "\tsmoothrad = %f mm\n" % smoothrad
    posterity += "\tDt = %f\n" % Dt
    posterity += "\tDcsf = %f\n" % Dcsf
    posterity += "\tblankval = %f\n" % blankval
    posterity += "\tclamp = %f\n" % clamp
    posterity += "\tnormslop = %f\n" % normslop
    posterity += "\tlogclamp = %f\n" % logclamp
    posterity += "\tuse_grad = %s\n" % use_grad

    return fvecs, posterity


def classify_fvf(fvecs, clf, airthresh=0.5, t1_will_be_used=False):
    """
    Segment a feature vector field using a scikit-learn classifier.

    Parameters
    ----------
    fvecs: (nx, ny, nz, nfeatures) array
        The feature vector field
    clf: sklearn.ensemble.*
        A trained classifier

    Output
    ------
    segmentation: (nx, ny, nz) array
        The segmentation with classes as integers
    probs: (nx, ny, nz, nclasses) array
        The probabilistic segmentation
    """
    fvecsr = np.reshape(fvecs, (np.prod(fvecs.shape[:3]), fvecs.shape[-1]))

    # One bit of fiat here: if there is no signal, we MUST call that voxel
    # air because it is crucial for the mask to protect downstream
    # processing from logs of or division by 0.  The 1st stage machine
    # learning classifier is pretty good at that, but the 2nd isn't.
    #
    # fvecs[..., 0] is a clamped log of the signal at b=0.  Technically this
    # function does not know where the clamp is, but usually it is -10.  Use
    # the minimum of fvecs[..., 0] as a suggestion, but override it in the case
    # of cropped input where there is no air.
    forcedair = fvecsr[:, 0] <= min(-8, 0.8 * np.min(fvecs[..., 0]))

    if hasattr(clf, 'predict_proba'):
        probsr = clf.predict_proba(fvecsr)

        probsr[forcedair, 0] = 1
        probsr[forcedair, 1:] = 0

        probs = np.reshape(probsr, fvecs.shape[:3] + (probsr.shape[-1],)).copy()

        # Since there are 4 types to segment to (air, brain, csf, and other,
        # 0-3) but one of the questions is more important than the others (is
        # it air or not?), we have a classic election problem.
        #
        # Consider a voxel where the probabilities are [0.4, 0.2, 0.3, 0.1].
        # clf.predict() would choose air, but for TIV purposes CSF is the right
        # choice.
        #
        # Make the TIV come out right by internally zeroing out air where its
        # probability is < airthresh.  (In practice this does not affect a lot
        # of voxels.)
        probsr[:, 0][probsr[:, 0] < airthresh] = 0
        mask = clf.classes_.take(np.argmax(probsr, axis=1), axis=0)
    else:
        mask = clf.predict(fvecsr)
        mask[forcedair] = 0
        probs = None

    seg = np.reshape(mask, fvecs.shape[:3])

    if t1_will_be_used:
        # Other that is a voxel or so thick is probably other, i.e. tentorium +
        # partial volume stuff, but large clumps of other are probably actually
        # dark brain, IF the T1 TIV can be relied on to get rid of other far
        # outside the brain.
        other = utils.cond_to_mask(seg, 3)
        obrain = utils.binary_erosion(other)
        seg[obrain == 1] = 1
        probs[obrain == 1, 1] += probs[obrain == 1, 3]
        probs[obrain == 1, 3] = 0

    return seg, probs


def fwhm_to_voxel_sigma(fwhm, aff, fwhm_per_sigma=0.4246609):
    """
    Convert FWHM (as a scalar or vector) in mm to a vector of scales in voxels
    suitable for use as sigma in ndi.filters.gaussian_filter.

    Parameters
    ----------
    fwhm: float or sequence of floats in aff's scale (should be mm)
    aff: affine matrix for the image
    fwhm_per_sigma: how many FWHMs are in a standard deviation,
        = 1.0 / (8 * np.log(2))**0.5 for the normal distribution.
        (~= 0.4246609)
    Output
    ------
    sigma: (3,) array
        Standard deviations in voxel coordinates
    """
    return fwhm_per_sigma * np.asarray(fwhm) / utils.voxel_sizes(aff)


def boost_bgbtiv(probs, bgbtiv, relboost=0.5, absboost=0.5):
    """
    The trained classifiers have deliberately limited locational knowledge,
    and we know better than them that csf or other outside the gbtiv is probably
    a distraction. On the other hand, we don't want to completely trust the gbtiv.
    This returns a copy of probs where air has been upweighted and everything else
    downweighted outside gbtiv, by an amount controlled by bgbtiv and boost.

    Parameters
    ----------
    probs : [nx, ny, nz, 4] np.array
        A classifier's probabilities for air, brain, csf, and other.
    bgbtiv : [nx, ny, nz] array
        Blurred gradient based TIV
    relboost : float in [0, 1]
        A relatively gentle boost. Definite tissue/air will still be definite tissue/air afterwards.
    absboost : float in [0, 1]
        The no-messing-around boost. What bgbtiv says goes.
        The output will be boost * boosted + (1 - boost) * probs
    """
    boosted = probs.copy()
    boosted[..., 0] *= 1 - bgbtiv
    for t in range(1, probs.shape[-1]):
        boosted[..., t] *= bgbtiv
    norm = boosted.sum(axis=3)
    for t in range(probs.shape[-1]):
        boosted[..., t] /= norm
    boosted = relboost * boosted + (1 - relboost) * probs
    absboosted = probs.copy()
    airprob = probs[..., 0]
    absboosted[..., 0] = 1 - bgbtiv
    norm = bgbtiv[airprob < 1] / (1 - airprob[airprob < 1])
    for t in range(1, probs.shape[-1]):
        absboosted[airprob < 1, t] *= norm
    return absboost * absboosted + (1 - absboost) * boosted


def _note_progress(mask, label):
    msg = "sum(% 50s):\t%6d" % (label, mask.sum())
    return msg + "\n"


def renormalize_probs(probs):
    """
    Return a version of probs with the last axis summing to 1.
    """
    sprobs = probs.sum(axis=-1)
    out = probs.astype(float)
    for category in range(probs.shape[-1]):
        out[..., category] /= sprobs
    return out


def probabilistic_classify(fvecs, aff, clf, smoothrad=10.0,
                           t1wtiv=None, t1fwhm=(2.0, 10.0, 2.0)):
    """
    Do a probabilistic segmentation.

    Parameters
    ----------
    fvecs: (nx, ny, nz, nfeatures) array
        The feature vector field
    aff: (4, 4) array
        Affine matrix for fvecs
    clf: dict
        The classifier
    smoothrad: float
        The smoothing radius in mm
        This legacy parameter is used in training the classifier as well, so
        it should match for training and classification.  Recent classifiers
        record their smoothrad and will override anything you set here.
    t1wtiv: (nx, ny, nz) array
        Optional intracranial volume mask from an undistorted (e.g. T1w) image,
        but in the same space as the dMRI. Because of EPI distortion it is only
        used as a suggestion.
    t1fwhm: sequence of 3 floats
        How much to blur t1wtiv by in x, y, and z mm. Blur it more in the phase
        encoding direction.

    Output
    ------
    seg: (nx, ny, nz) int array
        The non-probabilistic segmentation
    probs: (nx, ny, nz, nclasses) array
        The estimated probability of each class.
    posterity: str
        Info on how probs was made.
    """
    lsvmmask, probs = classify_fvf(fvecs, clf['1st stage'], t1_will_be_used=t1wtiv is not None)

    sigma = fwhm_to_voxel_sigma(smoothrad, aff)
    #ndi.filters.gaussian_filter(1 - probs[..., 0], sigma=sigma, output=fvecs[..., 4],
    #                            mode='nearest')
    #fvecs[..., 4] *= fvecs[..., 4]

    probs = boost_bgbtiv(probs, fvecs[..., 4])

    augvecs = np.empty(fvecs.shape[:3] + (fvecs.shape[3] + probs.shape[-1],))
    augvecs[..., :fvecs.shape[-1]] = fvecs
    for v in range(probs.shape[-1]):
        ndi.filters.gaussian_filter(probs[..., v], sigma=sigma,
                                    output=augvecs[..., v + fvecs.shape[-1]],
                                    mode='nearest')

    # Reboost because blurring caused classes to leak over the gbtiv.
    augvecs[..., fvecs.shape[-1]:] = boost_bgbtiv(augvecs[..., fvecs.shape[-1]:], fvecs[..., 4])

    # augvecs = np.empty(fvecs.shape[:3] + (12,))
    # augvecs[..., :4] = fvecs
    # sigma = fwhm_to_voxel_sigma(smoothrad, aff)
    # for v in xrange(4):
    #     ndi.filters.gaussian_filter(probs[..., v], sigma=sigma,
    #                                 output=augvecs[..., v + 4], mode='nearest')
    #     ndi.filters.gaussian_filter(probs[..., v], sigma=2 * sigma,
    #                                 output=augvecs[..., v + 8], mode='nearest')
    # em10 = np.exp(-10)
    # augvecs[..., 4:][augvecs[..., 4:] < em10] = em10
    # augvecs[..., 4:] = 1 + np.log(augvecs[..., 4:])
    lsvmmask, probs = classify_fvf(augvecs, clf['2nd stage'], t1_will_be_used=t1wtiv is not None)
    posterity = "\nClassified with a 2nd RFC stage.\n"

    # Turn off absboost to avoid injecting blurred air into the gbtiv.
    probs = boost_bgbtiv(probs, fvecs[..., 4], absboost=0)
    lsvmmask = probs.argmax(axis=3)

    if t1wtiv is not None:
        # Update probs with t1wtiv blurred along the phase encoding direction.
        posterity += "\nUsing %s.\n" % t1wtiv
        t1sigma = fwhm_to_voxel_sigma(t1fwhm, aff)
        t1tiv = ndi.filters.gaussian_filter(nib.load(t1wtiv).get_fdata().astype(float),
                                            sigma=t1sigma, mode='nearest')
        probs[..., 0] *= 1 - t1tiv
        for t in range(1, 4):
            probs[..., t] *= t1tiv

        # Deal with t1tiv's absolutes or practical absolutes. (Don't use ==, it doesn't work here.)
        probs[t1tiv < 0.01, 0] = 1
        probs[t1tiv < 0.01, 1:] = 0
        probs[t1tiv > 0.99, 0] = 0

        sprobs = probs.sum(axis=3)
        probs[probs[..., 0] < 0.5 * sprobs, 0] = 0  # Coalesce the anyone-but-air vote.
        sprobs = probs.sum(axis=3)

        # Set otherwise irreconcilable disagreements to other (not other,
        # because other is more fragile in morphological filter steps later).
        probs[sprobs < 0.01, 3] = 1
        probs[sprobs < 0.01, :3] = 0

        probs = renormalize_probs(probs)

        probsr = np.reshape(probs, (np.prod(probs.shape[:3]), probs.shape[-1]))
        mask = clf['2nd stage'].classes_.take(np.argmax(probsr, axis=1), axis=0)
        tempmask = np.reshape(mask, probs.shape[:3])

        # Voxels which used to be air but were brought back by t1tiv are likely
        # to be other in a large enough lump that it would be removed by the
        # morphological filtering later.  Reclassify them as brain to protect them.
        mask = utils.cond_to_mask(tempmask, 3)
        mask[lsvmmask > 0] = 0  # Leave old ones alone.
        lsvmmask = tempmask.copy()
        lsvmmask[mask > 0] = 1

    brain = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    csf = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    brain[lsvmmask == 1] = 1
    csf[lsvmmask == 2] = 1
    posterity += _note_progress(brain, "brain before removing disconnected components")
    posterity += _note_progress(csf, "csf")

    # The brain has to be connected, even if somewhat tenuously.  We can firm
    # that up with dilation by 1 voxel.
    scales = utils.voxel_sizes(aff)
    maxscale = max(scales)
    ball = utils.make_structural_sphere(aff, maxscale)
    tiv = utils.binary_dilation(brain, ball)
    tiv = utils.remove_disconnected_components(tiv, inplace=False)
    brain[tiv == 0] = 0
    posterity += _note_progress(brain, "brain after removing disconnected components")

    # "Other" is necessary to fill in partial volume voxels and dark regions
    # like the globus pallidus.  However, it is the least reliable class coming
    # out of the RFC.  Other in direct contact with brain is much more reliable
    # than distant other, and large chunks of air or other that has been hole
    # filled (after a closing operation) can be reclassified as brain.
    other = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    other[lsvmmask == 3] = 1

    tiv = brain + csf  # + other
    # Close by a bit at first.
    ball = utils.make_structural_sphere(aff, 1.5 * maxscale)
    closed = utils.binary_closing(tiv, ball)
    # fill holes
    #holes, success = utils.fill_holes(closed, aff, verbose=False)
    holes, success = utils.fill_axial_holes(closed)
    holes[tiv > 0] = 0
    # Reclassify holes as brain, csf, or other.
    posterity += _note_progress(holes, "holes in brain + csf")
    if np.any(holes) and probs is not None:
        brainholes = holes.copy()
        #csfholes = holes.copy()
        # Big chunks of other in holes are brain.
        brainholes[probs[..., 1] + probs[..., 2] > probs[..., 3]] = 0
        #csfholes[brainholes > 0] = 0

        brainholes = utils.binary_opening(brainholes, ball)
        brainholes = utils.binary_dilation(brainholes, ball)
        brainholes[holes == 0] = 0

        #posterity += _note_progress(brain, "brain before reclassifying holes")
        brain[brainholes > 0] = 1
        posterity += _note_progress(brain, "brain after reclassifying holes")
        posterity += _note_progress(other, "other before reclassifying holes")
        other[holes > 0] = 1
        other[brainholes > 0] = 0
        posterity += _note_progress(other, "other after reclassifying holes")
#        posterity += _note_progress(csf[brain > 0], "csf[brain > 0]")
#        posterity += _note_progress(csf, "csf")

    # rm cruft
    bits = utils.binary_opening(brain + csf + other, ball)
    tiv = utils.remove_disconnected_components(bits, inplace=False)
    # # Now close by a lot (slow)
    # #ball = utils.make_structural_sphere(aff, smoothrad)
    # closed = utils.binary_closing(tiv, ball)
    # # fill holes
    # #holes, success = utils.fill_holes(closed, aff, verbose=False)
    # holes, success = utils.fill_axial_holes(closed)
    # holes[tiv > 0] = 0
    # tiv[holes > 0] = 1
    # # rm cruft
    # bits = utils.binary_opening(tiv, ball)
    # tiv = utils.remove_disconnected_components(bits, inplace=False)

    tiv = utils.binary_dilation(tiv, ball)
    tiv[lsvmmask == 0] = 0
    tiv[holes > 0] = 1
    bits[tiv > 0] = 0
    posterity += _note_progress(bits, "bits")
    if bits.sum() > 0:
        ball = utils.make_structural_sphere(aff, maxscale)
        bits = utils.binary_dilation(bits, ball)
        tiv[bits > 0] = 0
    csf[tiv == 0] = 0  # csf that isn't connected to itself or brain isn't CSF.
    posterity += _note_progress(csf, "csf after trimming by the dilated TIV")
    other[tiv == 0] = 0  # other that isn't connected to itself or brain isn't CSF.
    posterity += _note_progress(other, "other")

    # CSF far away from brain is probably eyeball, so be more aggressive in
    # removing dangly CSF (but watch out for severe atrophy).
    ball = utils.make_structural_sphere(aff, 4 * maxscale)
    #tiv[holes > 0] = 1
    bits = utils.binary_opening(brain + csf + other, ball)  # Do NOT connect with other.
    tiv = utils.remove_disconnected_components(bits, inplace=False)
    bits[tiv > 0] = 0
    if np.any(bits):
        #ball = utils.make_structural_sphere(aff, 2 * maxscale)
        bits = utils.binary_dilation(bits, ball)
        #tiv[bits > 0] = 0
        csf[bits > 0] = 0
        other[bits > 0] = 0
        posterity += _note_progress(csf, "csf after removing bits")
        posterity += _note_progress(other, "other after removing bits")

    # Now there is a good chance that tiv excludes the eyeballs, we want to
    # remove isolated whiskers of other, but keep large clumps of other near
    # the brain + csf that is probably biased down brain.
    optiv = tiv.copy()
    #optiv[other == 1] = 1
    if fvecs.shape[-1] > 4:
        optiv[fvecs[..., 4] > 0.75] = 1  # Use the grad-based TIV
        optiv[fvecs[..., 4] < 0.25] = 0
    ball = utils.make_structural_sphere(aff, 1.5 * maxscale)
    optiv = utils.binary_opening(optiv, ball)
    utils.remove_disconnected_components(optiv, inplace=True)
    optiv[tiv > 0] = 1
    optiv = utils.binary_closing(optiv, ball)
    optiv, success = utils.fill_holes(optiv, aff, verbose=False)
    other[optiv == 0] = 0
    posterity += _note_progress(other, "other after trimming by the dilated TIV")

    # # Restore voxels at the edge of the TIV with prob(air) <= ...
    # # 0.75 is too much.
    # # 0.625 helps a little bit, but is it significant?
    # ball = utils.make_structural_sphere(aff, maxscale)
    # tiv = brain + csf + other
    # edge = utils.binary_dilation(tiv, ball)
    # edge[tiv > 0] = 0   # Now it's the edge just outside the tiv.
    # probs[..., 0][probs[..., 0] < 0.625] = 0
    # probs = probs.reshape((np.prod(tiv.shape), probs.shape[-1]))

    # clf2 = clf.get('2nd stage', clf['1st stage'])
    # seg = clf2.classes_.take(np.argmax(probs, axis=1), axis=0)
    # seg = seg.reshape(tiv.shape)
    # seg[edge == 0] = 0
    # brain[seg == 1] = 1
    # posterity += _note_progress(brain, "brain after growing")
    # csf[seg == 2]   = 1
    # posterity += _note_progress(csf, "csf after growing")
    # other[seg == 3] = 1
    # posterity += _note_progress(other, "other after growing")

    # Some other voxels that were isolated from other other voxels, but
    # surrounded by brain and csf may have been mistakenly eliminated.
    # Restore them.
    tiv = brain + csf + other
    holes, success = utils.fill_holes(tiv, aff, verbose=False)
    holes[tiv > 0] = 0  # Convert holes from a filled mask to just the holes.
    other[holes > 0] = 1

    # Final cleanup
    tiv = utils.remove_disconnected_components(tiv + holes, inplace=False)
    brain[tiv == 0] = 0
    posterity += _note_progress(brain, "brain after removing disconnections")
    csf[tiv == 0] = 0
    posterity += _note_progress(csf, "csf after removing disconnections")
    other[tiv == 0] = 0
    posterity += _note_progress(other, "other after removing disconnections")

    lsvmmask = utils.cond_to_mask(brain, 1)
    lsvmmask[csf   == 1] = 2
    lsvmmask[other == 1] = 3

    if '3rd stage' in clf:
        # Now reclassify using the blurred segmentations as morphological priors.
        for v in range(4):
            unsmoothed = np.zeros(lsvmmask.shape)
            unsmoothed[lsvmmask == v] = 1.0
            ndi.filters.gaussian_filter(unsmoothed, sigma=sigma,
                                        output=augvecs[..., v + fvecs.shape[-1]], mode='nearest')
        lsvmmask, probs = classify_fvf(augvecs, clf['3rd stage'], t1wtiv is not None)
    return lsvmmask, probs, posterity


def correct_bias(fvecs, s0, clf, aff, smoothrad, t1wtiv):
    """
    Get the probabilities from fvecs and clf (and t1wtiv and t1fwhm),
    estimate the average brightness for each tissue class,
    predict the voxel brightnesses unbiased by the coil sensitivity map,
    calculate ln(s0 / prediction), blur it, and exponentiate that to get the bias field,
    use it to correct fvecs.
    """
    _lsvmmask, probs = classify_fvf(fvecs, clf['1st stage'], t1_will_be_used=t1wtiv is not None)
    nclasses = probs.shape[-1]

    mean_brightnesses = np.linalg.lstsq(probs.reshape((np.prod(probs.shape[:-1]), probs.shape[-1])),
                                        s0.flat, rcond=None)[0]
    pred = np.zeros_like(s0)
    bfloor = 0.5 * min(mean_brightnesses)
    for v in range(nclasses):
        pred += mean_brightnesses[v] * probs[..., v]
    logbias = 0.5 * np.log((s0**2 + bfloor**2) / (pred**2 + bfloor**2))
    del pred
    
    # There is a lot of stuff in probs[..., 0] that isn't air, and that throws
    # off the bias estimate. We don't need bias correction in the air, so just
    # attenuate it.
    logbias *= 1.0 - probs[..., 0]
    
    ndi.filters.gaussian_filter(logbias, sigma=smoothrad, output=logbias, mode='nearest')
    bias = np.exp(logbias)

    logbias /= np.log(10)
    for v in (0, 1):
        fvecs[..., v] -= logbias
    del logbias
    fveclog = "Corrected for bias: the rms factor was %f." % np.std(bias)

    if fvecs.shape[-1] > 4:
        # Remake the grad-based TIV, using a better pd approximation that isn't
        # just bias corrected, but does away with the assumption that CSF is 2x
        # as bright as tissue. To make the new PD we will multiply bias
        # corrected s0 by sum(prob[..., v] / mean_brightnesses[v]), to simulate
        # what s0 would look like if all the tissue classes had the same
        # intrinsic brightness.  But because the tissue weighted *brightness*
        # needs to be weighted by both tissue fraction (p) and intrinsic tissue
        # brightness (T), the "p/T" factor is really sum(pT) / sum(pT**2).
        bcs0 = s0 / bias
        pd = np.zeros_like(s0)
        pd_denom = np.zeros_like(s0)
        #
        # Call the stuff outside the TIV other.
        mean_brightnesses[0] = mean_brightnesses[3]
        #
        for v in range(nclasses):
            pd += probs[..., v] * mean_brightnesses[v]
            pd_denom += probs[..., v] * mean_brightnesses[v]**2
        pd = bcs0 * pd / pd_denom
        del bcs0
        del pd_denom
        
        # Before blurring the grad based TIV, close it to avoid demphasizing
        # susceptibility horns.
        gbtiv = make_grad_based_TIV(None, None, aff, pd=pd)
        sigma = smoothrad / utils.voxel_sizes(aff)
        ball = utils.make_structural_sphere(aff, smoothrad)
        gbtiv = utils.binary_closing(gbtiv, ball)
        fvecs[..., 4] = ndi.gaussian_filter(gbtiv.astype(float), sigma, mode='nearest')
    return fvecs, fveclog, bias


def feature_vector_classify(data, aff, bvals=None, clf='RFC_classifier.pickle',
                            relbthresh=0.04, smoothrad=10.0, s0=None, Dt=0.0021,
                            Dcsf=0.00305, blankval=0, clamp=30,
                            normslop=0.2, logclamp=-10, fvecs=None,
                            t1wtiv=None, t1fwhm=(2.0, 10.0, 2.0),
                            bias_correct=True):
    """
    Make at least one feature vector field from data, and use for segmentation
    with a trained sklearn classifier.

    Parameters
    ----------
    data: (nx, ny, nz, nv) array
        Non-FLAIR diffusion data
    aff: (4, 4) array
        Affine matrix for data
    bvals: None or (nv,) array
        The diffusion weightings for each volume in data.  Only needed if fvecs
        is not provided.
    clf: str or dict
        The (classifier, which must have been already trained to segment air,
        brain, CSF, and other tissue as 0, 1, 2, and 3 from
        make_feature_vectors(same parameters).
        If a str, load_classifier(clf) will be used.
    relbthresh: float
        The cutpoint between b0s and DWIs, as a fraction of max(bvals).
        Only used if s0 is None.
    smoothrad: float
        The smoothing radius in mm
        This legacy parameter is used in training the classifier as well, so
        it should match for training and classification.  Recent classifiers
        record their smoothrad and will override anything you set here.
    s0: None or (nx, ny, nz) array
        The average of the b=0 volumes.  Will be made if None.
    Dt: float
        The intrinsic diffusivity of periaxonal water parallel to the axons,
        in reciprocal units of bvals.
    Dcsf: float
        The CSF diffusiivity in reciprocal units of bvals.
    blankval: float
        Set the output to this wherever s0 is 0.
    clamp: float
        Typically some voxels will have ratios of small numbers / tiny numbers,
        and be ridiculously and pointlessly large.  Since they can easily exceed
        the data range supported by fslview, it is best to clip them at +- some
        value well away from 1.  Set to None if you really do not want to clamp.
    normslop: float
        s0 needs to be normalized before it can be used with an SVM, so it is
        divided by median(s0[np.abs(madje - 1) < normslop]), where madje comes
        from make_mean_adje().
        It should be large enough to include most brain voxels, but small enough
        to exclude most non-brain voxels.
        Since it is a parameter of training data, do not change it!
    logclamp: float
        Feature vector values < this (in log10 space) will be set to this.
    fvecs: str or None
        Optional filename of a .nii to load the (1st stage) feature vectors from.
        If not given, they will be made from data.
    t1wtiv: str or None
        Optional filename of a .nii to load the undistorted (e.g. T1w) intracranial
        mask from. It must be in the same space as data, and is only used as a
        suggestion.
    t1fwhm: sequence of 3 floats in mm
        FWHM of the Gaussian kernel to blur t1wtiv with to soften the suggestion.
        Blur a bit to account for head motion, and more in the phase encoding
        direction to account for EPI distortion.

    Output
    ------
    brain: (nx, ny, nz) array of type np.uint8
        1 where it thinks there is brain tissue and 0 elsewhere.
    csf: (nx, ny, nz) array of type np.uint8
        1 where it thinks there is CSF and 0 elsewhere.
        brain + csf, maybe with hole filling and/or a little closing, should be very TIVish.
    holes: (nx, ny, nz) array of type np.uint8
        1 where there were holes, and 0 elsewhere.
    posterity: str
        A description of how the classification was done, i.e. the classifier parameters.
    """
    posterity = ''

    # Hurl early if we can't get a classifier.
    clf, posterity = load_classifier(clf)
    #ic(clf)
    if '3rd stage' in clf:
        nstages = 3
    else:
        nstages = 2
    posterity += "The classifier is a %d stage random forest.\n" % nstages
    posterity += clf['log'] + "\n"
    if not (hasattr(clf['1st stage'], 'predict_proba') or hasattr(clf['1st stage'], 'predict')):
        raise ValueError("The classifier is invalid")
    if hasattr(clf['1st stage'], 'intercept_'):
        posterity += "Using support vector classifier:\n%s\n\n" % clf['1st stage']
        posterity += "Classifier intercept:         %s\n" % clf['1st stage'].intercept_
        posterity += "Classifier intercept scaling: %s\n" % clf['1st stage'].intercept_scaling
        posterity += "Classifier coefficients:\n%s\n\n" % clf['1st stage'].coef_

    if isinstance(fvecs, six.string_types):
        fvnii = nib.load(fvecs)
        fvecs = fvnii.get_fdata()
        for ext in fvnii.header.extensions:
            posterity += str(ext)
    else:
        fvecs, fveclog = make_feature_vectors(data, aff, bvals, relbthresh, smoothrad,
                                              s0, Dt, Dcsf, blankval, clamp, normslop,
                                              logclamp=logclamp)

    if bias_correct:
        if s0 is None:
            s0 = utils.calc_average_s0(data, bvals, relbthresh)
        fvecs, fveclog, bias = correct_bias(fvecs, s0, clf, aff, smoothrad, t1wtiv=t1wtiv)
    lsvmmask, probs, clog = probabilistic_classify(fvecs, aff, clf,
                                                   smoothrad=smoothrad,
                                                   t1wtiv=t1wtiv,
                                                   t1fwhm=t1fwhm)
    posterity += clog

    brain = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    csf = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    other = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    brain[lsvmmask == 1] = 1
    csf[lsvmmask   == 2] = 1
    other[lsvmmask == 3] = 1

    posterity += _note_progress(brain, "brain at the end")
    posterity += _note_progress(csf, "csf at the end")
    posterity += _note_progress(other, "other at the end")

    return brain, csf, other, posterity
