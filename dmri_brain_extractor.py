import datetime
import nibabel as nib
import numpy as np
import os
import scipy.ndimage as ndi
from scipy.ndimage.filters import median_filter, convolve
import scipy.special
import scipy.stats as stats
import sklearn.externals.joblib as joblib

try:
    from skimage.filter import threshold_otsu as otsu
except:
    from dipy.segment.threshold import otsu

from skimage.morphology import reconstruction

import ADIR.numerical_utils as nu

def save_mask(arr, aff, outfn, exttext='', outtype=np.uint8):
    """
    Saves a numpy array to a nii.
    """
    mnii = nib.Nifti1Image(arr.astype(outtype), aff)
    if exttext:
        mnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                                 exttext))
    nib.save(mnii, outfn)    

def extract_dmri_brain(data, ecnii, tivfn, bvals, bthresh=300.0, holefill=True,
                       percentage=75, medrad=1, nmed=2, verbose=True,
                       maxtissdiffusivity=0.001616, dilate=True,
                       dilate_before_chopping=1, closerad=3.7,
                       whiskradinvox=4, Dt=0.0007, DCSF=0.003, brfn='',
                       isFLAIR=None, svc='RFC_classifier.pickle',
                       trim_whiskers=True):
    """
    Make a brain or TIV mask from 4D diffusion data.

    It selects voxels based on a minimum threshold (not air), maximum
    diffusivity (not CSF), and location relative to other brain voxels
    (e.g. hole filling).

    Parameters
    ----------
    data: array-like
        The 4D data.
    ecnii: array-like
        The opened nii instance of the data (data = ecnii.get_data())
    tivfn: string or None
        Filename for the TIV-style image.  If blank, no file will be
        written.
    bvals: array  (NOT a list!)
        The diffusion strengths for each volume in data, nominally in
        s/mm**2.  (Adjust bthresh, Dt, and DCSF if using different units.)
    bthresh: float
        Threshold b value between b0s and diffusion volumes.
    holefill: Boolean
        Whether or not to fill holes, producing a more TIV-like result.
        
    percentage: 
        No longer used.
    medrad: float
        Radius of the median filter relative to the largest voxel size.
    nmed: int
        Number of times to run the median filter
    verbose: Boolean
        Make it chattier.
    maxtissdiffusivity: 
        No longer used.
    dilate: Bool or float
        If >= 1, dilate with this radius (relative to the largest voxel size).
        If True, use medrad * nmed.
        If between 0 and 1, dilate by a voxel.
    dilate_before_chopping: float
        If >= 1, dilate _the_copy_used_for_finding_the_largest_connected_component_
        with this * the largest voxel size before removing disconnected components.
        1 is highly recommended - less can chop off important things, and more can
        connect unwanted things like eyeballs and jowls.
    closerad: float
    whiskradinvox: float
        Scale relative to the largest voxel dimension of data.
    Dt: float
        The nominal diffusivity of tissue in reciprocal units of bvals. 
        Depends on temperature.
    DCSF: float
        The nominal diffusivity of CSF in reciprocal units of bvals.
        Depends on temperature, and can be artificially depressed in scans where
        the b0 CSF is brighter than the maximum int value.
    brfn: str
        Filename for the (tightish) brain mask.
        If not given no image will be saved to disk.

    Outputs
    -------
    mask: array-like
        3D array which is true where there should be brain, and 0 elsewhere.
    """
    brain, tiv = get_dmri_brain_and_tiv(data, ecnii, brfn, tivfn,
                                        bvals, bthresh, medrad, nmed, verbose,
                                        dilate, dilate_before_chopping,
                                        closerad, whiskradinvox, Dt, DCSF,
                                        isFLAIR=isFLAIR, trim_whiskers=trim_whiskers,
                                        svc=svc)
    if holefill:
        return tiv
    else:
        return brain

def get_dmri_brain_and_tiv(data, ecnii, brfn, tivfn, bvals, bthresh=300.0,
                           medrad=1, nmed=2, verbose=True, dilate=True,
                           dilate_before_chopping=1, closerad=3.7,
                           whiskradinvox=4, Dt=0.0007, DCSF=0.003,
                           isFLAIR=None, trim_whiskers=True,
                           svc='RFC_classifier.pickle'):
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
        The opened nii instance of the data (data = ecnii.get_data())
    brfn: string
        Filename for the brain mask.  If blank, no file will be
        written.
    tivfn: string
        Filename for the TIV mask.  If blank, no file will be
        written.
    bvals: array  (NOT a list!)
        The diffusion strengths for each volume in data, nominally in
        s/mm**2.  (Adjust bthresh, Dt, and DCSF if using different units.)
    bthresh: float
        Threshold b value between b0s and diffusion volumes.
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
    dilate_before_chopping: float
        If >= 1, dilate _the_copy_used_for_finding_the_largest_connected_component_
        with this * the largest voxel size before removing disconnected components.
        1 is highly recommended - less can chop off important things, and more can
        connect unwanted things like eyeballs and jowls.
    closerad: float
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
        determine it from the InversionTime in the DICOM extension (if present) of
        the nii header and/or the CSF/tissue brightness ratio.

    Outputs
    -------
    mask: array-like
        3D array which is 1 where there should be brain, and 0 elsewhere.
    tiv: array-like
        3D array which is 1 inside the braincase and 0 outside.
    """
    aff = ecnii.get_affine()
    for d in xrange(3):
        # Accept up to pi/4 obliqueness.
        if aff[d, d] < 0.70711 * np.linalg.norm(aff[d, :3]):
            # The problem seems to be in get_data(), not nib.save()
            print """
            Uh Oh: The input data is flipped or rotated by >= pi/4, which would
            cause inconsistencies in the output. Flip the input around first with
            fslreorient2std or removeFlipsRots.
            """
            return None
    
    if data is None:
        data = ecnii.get_data()
    assert len(data.shape) == 4

    b = np.asarray(bvals)
    b0 = np.median(data[..., bvals <= bthresh], axis=-1)
    scales = utils.voxel_sizes(aff)
    maxscale = max(scales)
    if nmed > 0 and medrad >= 1:
        # (Spatially) median filter the b0 just for consistency.
        ball = utils.make_structural_sphere(aff, medrad * maxscale)
        for i in xrange(nmed):
            b0 = median_filter(b0, footprint=ball)
    embDt = np.exp(-b * Dt)
    embDCSF = np.exp(-b * DCSF)
    w = (embDt * (1.0 - embDCSF))**2
    tisssig = np.zeros(b0.shape)
    embb0s = {}
    sw = 0.0
    for v, emb in enumerate(embDCSF):
        if emb not in embb0s:
            embb0s[emb] = emb * b0
        tisssig += w[v] * (data[..., v] - embb0s[emb])
        sw += w[v]
    del embb0s
    tisssig /= sw
    
    # Don't use median_otsu because it assumes isotropic voxels.
    if nmed > 0:
        if medrad >= 1:
            ball = utils.make_structural_sphere(aff, medrad * maxscale)
            if verbose:
                print "Median filtering %d times with radius %f." % (nmed,
                                                                     medrad * maxscale)
            for i in xrange(nmed):
                tisssig = median_filter(tisssig, footprint=ball)
        elif medrad > 0:
            print "Warning: not median filtering since medrad < 1."

    if verbose:
        print "Getting the Otsu threshold."
    thresh = otsu(tisssig)
    mask = np.zeros(tisssig.shape, np.bool)
    mask[tisssig >= thresh] = 1

    b0 = np.median(data[..., bvals <= bthresh], axis=-1)
    ball = utils.make_structural_sphere(aff, max(10.0, maxscale))
    csfmask = utils.binary_closing(mask, ball)
    csfmask, success = fill_holes(csfmask, aff, closerad * maxscale, verbose)
    csfmask = utils.binary_opening(csfmask, ball)
    csfmask[mask > 0] = 0
    csfmed = np.median(b0[csfmask > 0])
    b0tiss = b0[mask > 0]

    if isFLAIR is None:
        # Now we have an approximate brain, and we know it is surrounded by CSF
        # (in vivo) or solution (ex vivo), which we'll call CSF.  Figure out
        # whether CSF is brighter or darker than tissue in the b0.
        tissmed = np.median(b0tiss)
        if tissmed > 2.0 * csfmed:
            isFLAIR = True
            flair_msg = "This appears to be"
        else:
            isFLAIR = False
            flair_msg = "This is not"
        flair_msg += " a FLAIR acquisition"
    else:
        flair_msg = "Treating this as a "
        if not isFLAIR:
            flair_msg += "non-"
        flair_msg += "FLAIR scan as instructed."
    if verbose:
        print flair_msg

    purecsf = csfmask.copy()
    if not isFLAIR:
        mask, csfmask, other, holes, submsg = feature_vector_classify(data, aff, bvals, clf=svc)
        tiv = mask + csfmask + other + holes
    else:
        if dilate_before_chopping >= 1:
            dilrad = dilate_before_chopping * maxscale
        else:
            dilrad = 0
        mask = utils.remove_disconnected_components(mask, aff, dilrad, verbose=verbose)

        if dilate:
            # Dilate once with a big ball instead of multiple times with a small
            # one since the former will better approximate a sphere.
            if dilate is True:
                dilate = nmed * medrad
            if dilate >= 1:
                dilrad = dilate * maxscale
                if verbose:
                    print "Dilating with radius %f." % dilrad
                ball = utils.make_structural_sphere(aff, dilrad)
                mask = utils.binary_dilation(mask, ball)
            else:
                mask = utils.binary_dilation(mask)

        # In principle the brain is a hole in the CSF (depending on how you cut off
        # the spine), but in practice this is just a waste of time - it's better to
        # include dark brain voxels by accepting nonCSF voxels that are mostly
        # surrounded by known brain tissue.
        # csfholes, success = fill_holes(csfmask, aff, closerad * maxscale, verbose)
        # csfholes[csfmask > 0] = 0
        # mask[csfholes > 0] = 1
        # mask, success = fill_holes(mask, aff, closerad * maxscale, verbose)

        # Trim any whiskers caused by susceptibility problems.
        whiskrad = whiskradinvox * maxscale
        ball = utils.make_structural_sphere(aff, whiskrad)

        omask = utils.binary_closing(mask, ball) # Fill in sulci
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
            print "Removing %d whisker voxels" % sum(mask[whiskers > 0])
        mask[whiskers > 0] = 0
        mask = utils.remove_disconnected_components(mask, aff, dilrad, verbose=verbose)

        # Now, to get very dark voxels (putamen), close, and then remove CSF.
        tiv = utils.binary_closing(mask, ball)

    if trim_whiskers:
        # More whisker removal
        omask = utils.binary_dilation(tiv, ball)
        omask, success = fill_holes(omask, aff, closerad * maxscale, verbose)
        csfmask[omask == 0] = 0
        tiv[csfmask > 0] = 1
        tiv, success = fill_holes(tiv, aff, 0, verbose)
        #ball = utils.make_structural_sphere(aff, dilrad)
        tiv = utils.binary_erosion(tiv, ball)
        tiv = utils.remove_disconnected_components(tiv, aff, 0, verbose=verbose)
        tiv = utils.binary_dilation(tiv, ball)
        tiv[omask == 0] = 0
        #tiv = utils.binary_closing(tiv, ball)
        submsg = ''

    exttext  = "Mask made " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
    exttext += """
    by get_dmri_brain_and_tiv(data, %s, brfn, tivfn,
                              bvals=%s,
                              bthresh=%f,
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
    """ % (ecnii.get_filename(), bvals, bthresh, medrad, nmed,
           verbose, dilate, dilate_before_chopping, closerad,
           whiskradinvox, Dt, DCSF, trim_whiskers, svc)
    exttext += flair_msg + "\n" + submsg
    exttext = posterity_section(exttext)
    if brfn:
        mask[tiv > 0] = 1  # Recover dark voxels in the right place
        mask[csfmask > 0] = 0

        if isFLAIR:
            # Remove blips and whiskers
            ball = utils.make_structural_sphere(aff, maxscale)
            mask = utils.binary_opening(mask, ball)
            mask = utils.remove_disconnected_components(mask, aff, 0)

            mask = utils.binary_dilation(mask) # Recover partial CSF voxels.
            mask[tiv == 0] = 0              # but clamp to the TIV.        
        save_mask(mask, aff, brfn, exttext)
    if tivfn:
        save_mask(tiv, aff, tivfn, exttext)
    if verbose:
        print "Done"   # Useful if running in the background on a terminal.
    return mask, tiv

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
            print "Filling holes"
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
            ball = utils.make_structural_sphere(aff, dilrad)
            dmask = utils.binary_closing(mask, ball)
        else:
            dmask = mask.copy()
        seed = dmask.copy()
        seed[1:-1, 1:-1, 1:-1] = 1
        hmask = reconstruction(seed, dmask, method='erosion')

        if dilrad > 0:
            # Remove dmask's dilation and leave just the holes, 
            hmask = utils.binary_erosion(hmask, ball)
            #hmask[dmask > 0] = 0
            # but replace dilation that was part of a hole.
            #hmask = utils.binary_dilation(hmask, ball)

        mask[hmask > 0] = 1
        errval = 0
    except ImportError as e:
        if verbose:
            print "Could not fill holes because skimage.morphology.reconstruction was not found"
        errval = e
    except Exception as e:
        if verbose:
            print "Problem trying to fill holes:", e
            print "...continuing anyway..."
        errval = e
    return mask, errval

def bcut_from_rel(bvals, relbthresh=0.02):
    """
    Given a sequence of b values and a relative b threshhold between
    "undiffusion weighted" and "diffusion weighted", return the absolute b
    corresponding to the threshhold.
    """
    minb = np.min(bvals)
    maxb = np.max(bvals)
    return minb + relbthresh * (maxb - minb)

def calc_average_s0(data, bvals, relbthresh=0.02, bcut=None):
    if bcut is None:
        bcut = bcut_from_rel(bvals, relbthresh)
    return data[..., bvals <= bcut].mean(axis=-1)

def make_mean_adje(data, bvals, bthresh=0.02, s0=None, Dt=0.00210, Dcsf=0.00305,
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
    bthresh: float
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

    Output
    ------
    madje: (nx, ny, nz) array
        Specially weighted average over volume number of data / s0
    brightness_scale: float
        The median of s0[(madje * s0) > otsu(madje * s0)]
    """
    if s0 is None:
        s0 = calc_average_s0(data, bvals, bthresh)
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
            madje += vw * data[..., v]
    #madje = np.average(dataoet, axis=-1, weights=w)

    thresh = otsu(madje)
    brightness_scale = np.median(s0[madje > thresh])
    regularizer = regscale * brightness_scale
    madje /= (s0**2 + regularizer**2)**0.5
    
    madje[s0 == 0] = blankval
    madje[np.isnan(madje)] = blankval
    if clamp is not None:
        madje = np.clip(madje, -clamp, clamp, madje)
    return madje, brightness_scale

def make_feature_vectors(data, aff, bvals, bthresh=0.02, smoothrad=10.0, s0=None,
                         Dt=0.0021, Dcsf=0.00305, blankval=0, clamp=30,
                         normslop=0.4):
    """
    Make 4D vectors for segmenting data with a SVM.  Technically only the
    vectors closest to the cutplane are the support vectors, but by making all
    the classification vectors, the support vectors will be included.

    Parameters
    ----------
    data: (nx, ny, nz, nv) array
        Non-FLAIR diffusion data
    aff: (4, 4) array
        Affine matrix for data
    bvals: (nv,) array
    bthresh: float
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

    Output
    ------
    svecs: (nx, ny, nz, 4) array
        The classification quantities to use for SVM segmentation.
        svecs[..., 0]: s0 / median(s0[np.abs(madje - 1) < normslop])
        svecs[..., 1]: "neighborhood s0", i.e. svecs[..., 0] smoothed with a top
                       hat of radius smoothrad.
        svecs[..., 2]: madje from make_mean_adje().
        svecs[..., 3]: "neighborhood madje", i.e. svecs[..., 2] smoothed with a top
                       hat of radius smoothrad.
    """
    if s0 is None:
        s0 = calc_average_s0(data, bvals, bthresh)
    nfeatures = 4                                   # l2amp looks helpful, but empirically it isn't.
    svecs = np.empty(data.shape[:3] + (nfeatures,))
    svecs[..., 2], brightness_scale = make_mean_adje(data, bvals, s0=s0, Dt=Dt, Dcsf=Dcsf,
                                                     blankval=blankval, clamp=clamp)

    svecs[..., 0] = s0 / brightness_scale
    
    ball = utils.make_structural_sphere(aff, smoothrad)    
    median_filter(svecs[..., 2], footprint=ball, output=svecs[..., 3], mode='nearest')
    median_filter(svecs[..., 0], footprint=ball, output=svecs[..., 1], mode='nearest')

    # if nfeatures > 4:
    #     l2amp = sfa.calc_l2_amp(data, bvals, bvecs, s0=s0, nonorm=True)
    #     svecs[..., 4] = l2amp
    #     #ml2amp = ndi.filters.gaussian_filter(l2amp, 0.5 * smoothrad, mode='nearest')
    #     #adev = ndi.filters.gaussian_filter(np.abs(l2amp - ml2amp), 0.5 * smoothrad, mode='nearest')
    #     #svecs[..., 5] = np.sqrt(adev + ml2amp)
    #     median_filter(np.sqrt(l2amp), footprint=ball, output=svecs[..., 5], mode='nearest')
    return svecs

def classify_fvf(svecs, clf, airthresh=0.5, t1_will_be_used=False):
    """
    Segment a feature vector field using a scikit-learn classifier.

    Parameters
    ----------
    svecs: (nx, ny, nz, nfeatures) array
        The feature vector field
    clf: sklearn.ensemble.*
        A trained classifier

    Output
    ------
    segmentation: (nx, ny, nz) array
        The segmentation
    """
    svecsr = np.reshape(svecs, (np.prod(svecs.shape[:3]), svecs.shape[-1]))
    if hasattr(clf, 'predict_proba'):
        probsr = clf.predict_proba(svecsr)
        probs = np.reshape(probsr, svecs.shape[:3] + (probsr.shape[-1],)).copy()

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
        mask = clf.predict(svecsr)
        probs = None

    seg = np.reshape(mask, svecs.shape[:3])

    if t1_will_be_used:
        # Other that is a voxel or so thick is probably other, i.e. tentorium +
        # partial volume stuff, but large clumps of other are probably actually
        # dark brain, IF the T1 TIV can be relied on to get rid of other far
        # outside the brain.
        other = np.zeros_like(seg)
        other[seg == 3] = 1
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
    scales = utils.voxel_sizes(aff)
    return fwhm_per_sigma * np.asarray(fwhm) / scales

def fill_axial_holes(arr):
    mask = arr.copy()
    for z in xrange(arr.shape[2]):
        seed = arr[..., z].copy()
        seed[1:-1, 1:-1] = 1
        hmask = reconstruction(seed, mask[..., z], method='erosion')
        mask[hmask > 0, z] = 1
    return mask, 0

def feature_vector_classify(data, aff, bvals=None, clf='RFC_classifier.joblib_dump',
                            bthresh=0.02, smoothrad=13.5, s0=None, Dt=0.00300,
                            Dcsf=0.00305, blankval=0, clamp=30,
                            normslop=0.2, logclamp=-10, fvecs=None,
                            t1wtiv=None, t1fwhm=[2.0, 10.0, 2.0]):
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
    clf: str or sklearn classifier
        The (1st stage) feature vector classifier, which must have been already
        trained to segment air, brain tissue, CSF, and other tissue as 0, 1, 2,
        and 3 from np.log10(make_feature_vectors(same parameters)).
        If a str, joblib.load(clf) will be used, looking in . or the same
        directory as this file if necessary.
    bthresh: float
        The cutpoint between b0s and DWIs, as a fraction of max(bvals).
        Only used if s0 is None.        
    smoothrad: float
        The smoothing radius in mm
        Since it is a parameter of training data, do not change it!
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
    clf2: None, str, or sklearn classifier
        An optional 2nd stage classifier.  If this is given, the feature vectors
        will be augmented with the 1st stage class probabiilties convolved with
        a 3D Gaussian kernel of FWMH smoothrad.  clf must have a .predict_proba()
        method (sklearn.ensemble.forest.RandomForestClassifier does, and it is
        just the fraction of trees voting for each class), and clf2 must have
        been trained with augmented feature vectors (fvecs followed by the
        convolved 1st stage probabilities).

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
    clffn = None
    clf2 = None
    if isinstance(clf, str):
        clffn = clf
        if not os.path.isfile(clffn):
            try:
                clffn = os.path.join(os.path.dirname(__file__), clffn)
            except:
                raise ValueError("Could not find %s" % clffn)
        clf = joblib.load(clffn)
        posterity += "Classifier loaded from %s.\n" % clffn        
    if isinstance(clf, dict):
        posterity += "The classifier is a two stage random forest.\n"
        posterity += clf['log'] + "\n"
        clf2 = clf['2nd stage']
        clf = clf['1st stage']
    if not hasattr(clf, 'predict'):
        raise ValueError(clffn + " does not contain a valid classifier")
    if hasattr(clf, 'intercept_'):
        posterity += "Using support vector classifier:\n%s\n\n" % clf
        posterity += "Classifier intercept:         %s\n" % clf.intercept_
        posterity += "Classifier intercept scaling: %s\n" % clf.intercept_scaling
        posterity += "Classifier coefficients:\n%s\n\n" % clf.coef_

    if isinstance(fvecs, str):
        fvecs = nib.load(fvecs).get_data()
    else:
        fvecs = make_feature_vectors(data, aff, bvals, bthresh, smoothrad, s0,
                                      Dt, Dcsf, blankval, clamp, normslop)
        tlogclamp = 10**logclamp
        fvecs[fvecs < tlogclamp] = tlogclamp
        fvecs = np.log10(fvecs)
        posterity += "Logarithmic support vectors made with:\n"
        posterity += "\tbthresh = %f\n" % bthresh
        posterity += "\tsmoothrad = %f mm\n" % smoothrad
        posterity += "\tDt = %f\n" % Dt
        posterity += "\tDcsf = %f\n" % Dcsf
        posterity += "\tblankval = %f\n" % blankval
        posterity += "\tclamp = %f\n" % clamp
        posterity += "\tnormslop = %f\n" % normslop
        posterity += "\tlogclamp = %f\n" % logclamp
    lsvmmask, probs = classify_fvf(fvecs, clf, t1wtiv is not None)
        
    if clf2 is not None:
        svecs2 = np.empty(fvecs.shape[:3] + (fvecs.shape[3] + probs.shape[-1],))
        svecs2[..., :fvecs.shape[-1]] = fvecs
        sigma = fwhm_to_voxel_sigma(smoothrad, aff)
        for v in xrange(probs.shape[-1]):
            ndi.filters.gaussian_filter(probs[..., v], sigma=sigma,
                                        output=svecs2[..., v + fvecs.shape[-1]], mode='nearest')
        # svecs2 = np.empty(fvecs.shape[:3] + (12,))
        # svecs2[..., :4] = fvecs
        # sigma = fwhm_to_voxel_sigma(smoothrad, aff)
        # for v in xrange(4):
        #     ndi.filters.gaussian_filter(probs[..., v], sigma=sigma,
        #                                 output=svecs2[..., v + 4], mode='nearest')
        #     ndi.filters.gaussian_filter(probs[..., v], sigma=2 * sigma,
        #                                 output=svecs2[..., v + 8], mode='nearest')
        # em10 = np.exp(-10)
        # svecs2[..., 4:][svecs2[..., 4:] < em10] = em10
        # svecs2[..., 4:] = 1 + np.log(svecs2[..., 4:])
        lsvmmask, probs = classify_fvf(svecs2, clf2, t1wtiv is not None)
        posterity += "\nClassified with a 2nd RFC stage.\n"

    if t1wtiv is not None:
        # Update probs with t1wtiv blurred along the phase encoding direction.
        posterity += "\nUsing %s.\n" % t1wtiv
        t1sigma = fwhm_to_voxel_sigma(t1fwhm, aff)
        t1tiv = ndi.filters.gaussian_filter(nib.load(t1wtiv).get_data().astype(np.float),
                                            sigma=t1sigma, mode='nearest')
        probs[..., 0] *= 1 - t1tiv
        for t in xrange(1, 4):
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

        probsr = np.reshape(probs, (np.prod(probs.shape[:3]), probs.shape[-1]))
        mask = clf.classes_.take(np.argmax(probsr, axis=1), axis=0)
        tempmask = np.reshape(mask, probs.shape[:3])

        # Voxels which used to be air but were brought back by t1tiv are likely
        # to be other in a large enough lump that it would be removed by the
        # morphological filtering later.  Reclassify them as brain to protect them.
        mask = np.zeros_like(tempmask)
        mask[tempmask == 3] = 1
        mask[lsvmmask > 0] = 0  # Leave old ones alone.
        lsvmmask = tempmask.copy()
        lsvmmask[mask > 0] = 1
        
    def squawk(mask, label):
        # return  # convert squawk to a no-op.
        msg = "sum(% 50s):\t%6d" % (label, mask.sum())
        #print msg
        return msg + "\n"

    brain = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    csf = np.zeros(fvecs.shape[:3], dtype=np.uint8)
    brain[lsvmmask == 1] = 1
    csf[lsvmmask == 2] = 1
    posterity += squawk(brain, "brain before removing disconnected components")
    posterity += squawk(csf, "csf")
    
    # The brain has to be connected, even if somewhat tenuously.  We can firm
    # that up with dilation by 1 voxel.
    scales = utils.voxel_sizes(aff)
    maxscale = max(scales)
    ball = utils.make_structural_sphere(aff, maxscale)
    tiv = utils.binary_dilation(brain, ball)
    tiv = utils.remove_disconnected_components(tiv, inplace=False)
    brain[tiv == 0] = 0
    posterity += squawk(brain, "brain after removing disconnected components")
    
    # "Other" is necessary to fill in partial volume voxels and dark regions
    # like the globus pallidus.  However, it is the least reliable class coming
    # out of the RFC.  Other in direct contact with brain is much more reliable
    # than distant other, and other that has been hole filled (after a closing
    # operation) can be reclassified as brain.
    other = np.zeros_like(tiv)
    other[lsvmmask == 3] = 1
    tiv = brain + csf# + other
    # Close by a bit at first.
    ball = utils.make_structural_sphere(aff, 1.5 * maxscale)
    closed = utils.binary_closing(tiv, ball)
    # fill holes
    #holes, success = fill_holes(closed, aff, verbose=False)
    holes, success = fill_axial_holes(closed)
    #tiv = brain + csf
    holes[tiv > 0] = 0
    tiv[holes > 0] = 1
    # rm cruft
    bits = utils.binary_opening(tiv, ball)
    tiv = utils.remove_disconnected_components(bits, inplace=False)
    # # Now close by a lot (slow)
    # #ball = utils.make_structural_sphere(aff, smoothrad)
    # closed = utils.binary_closing(tiv, ball)
    # # fill holes
    # #holes, success = fill_holes(closed, aff, verbose=False)
    # holes, success = fill_axial_holes(closed)
    # holes[tiv > 0] = 0
    # tiv[holes > 0] = 1
    # # rm cruft
    # bits = utils.binary_opening(tiv, ball)
    # tiv = utils.remove_disconnected_components(bits, inplace=False)

    tiv = utils.binary_dilation(tiv, ball)
    tiv[lsvmmask == 0] = 0
    tiv[holes > 0] = 1
    bits[tiv > 0] = 0
    posterity += squawk(bits, "bits")
    if bits.sum() > 0:
        ball = utils.make_structural_sphere(aff, maxscale)
        bits = utils.binary_dilation(bits, ball)
        tiv[bits > 0] = 0
    csf[tiv == 0] = 0  # csf that isn't connected to itself or brain isn't CSF.
    posterity += squawk(csf, "csf after trimming by the dilated TIV")
    posterity += squawk(other, "other")
    other[tiv == 0] = 0
    posterity += squawk(other, "other after trimming by the dilated TIV")
    #tiv[other > 0] = 1
    bits = utils.binary_opening(tiv, ball)
    tiv = utils.remove_disconnected_components(bits, inplace=False)
    edge = utils.binary_dilation(tiv, ball)
    edge[tiv > 0] = 0
    edge[lsvmmask == 0] = 0
    tiv += edge

    # altered_tiv = utils.binary_dilation(tiv, ball)
    # altered_tiv = utils.binary_dilation(altered_tiv, ball)
    # bits[tiv > 0] = 0
    # altered_tiv[bits > 0] = 0
    # brain[altered_tiv == 0] = 0
    # posterity += squawk(brain, "brain after trimming by the dilated TIV")
    # csf[altered_tiv == 0] = 0
    # posterity += squawk(csf, "csf after trimming by the dilated TIV")
    # posterity += squawk(other, "other before trimming by the dilated TIV")
    # other[altered_tiv == 0] = 0
    # posterity += squawk(other, "other after trimming by the dilated TIV")
        
    # Reclassify holes as either brain or csf.
    # closed = utils.binary_closing(brain + csf, ball)
    # holes, success = fill_holes(closed, aff, verbose=False)
    # holes[closed > 0] = 0  # Convert holes from a filled mask to just the holes.
    posterity += squawk(holes, "holes in brain + csf")
    if np.any(holes) and probs is not None:
        brainholes = holes.copy()
        #csfholes = holes.copy()
        brainholes[probs[..., 1] < probs[..., 2]] = 0
        #csfholes[brainholes > 0] = 0
        posterity += squawk(brain, "brain before reclassifying holes")
        brain[brainholes > 0] = 1
        posterity += squawk(other, "other before reclassifying holes")
        other[brainholes > 0] = 0

        # Other that is in a hole and in a big chunk is probably dark brain.
        otherbrain = other.copy()
        otherbrain[holes == 0] = 0
        otherbrain = utils.binary_opening(otherbrain, ball)
        brain[otherbrain > 0] = 1
        other[otherbrain > 0] = 0
        
        posterity += squawk(brain, "brain after reclassifying holes")
        posterity += squawk(other, "other after reclassifying holes")
        posterity += squawk(csf[brain > 0], "csf[brain > 0]")
        posterity += squawk(csf, "csf")

    if probs is not None:
        # Restore voxels at the edge of the TIV with prob(air) <= ...
        # 0.75 is too much.
        # 0.625 helps a little bit, but is it significant?
        ball = utils.make_structural_sphere(aff, maxscale)
        tiv = brain + csf + other
        altered_tiv = utils.binary_dilation(tiv, ball)
        #altered_tiv[tiv > 0] = 0   # Now it's the edge just outside the tiv.
        probs[..., 0][probs[..., 0] < 0.625] = 0
        probs = probs.reshape((np.prod(tiv.shape), probs.shape[-1]))
        if clf2 is None:
            clf2 = clf
        seg = clf2.classes_.take(np.argmax(probs, axis=1), axis=0)
        seg = seg.reshape(tiv.shape)
        seg[altered_tiv == 0] = 0 
        seg[tiv > 0] = 0            # Now it's the edge just outside the tiv.
        brain[seg == 1] = 1
        posterity += squawk(brain, "brain after growing")
        csf[seg == 2]   = 1
        posterity += squawk(csf, "csf after growing")
        other[seg == 3] = 1
        posterity += squawk(other, "other after growing")
        tiv = brain + csf + other
    
    # Remove other that is > 2 voxels from brain + csf.
    ball = utils.make_structural_sphere(aff, 2 * maxscale)
    altered_tiv = utils.binary_dilation(brain + csf, ball)
    other[altered_tiv == 0] = 0
    posterity += squawk(other, "other after trimming by the dilated TIV")
    
    # altered_tiv = utils.binary_closing(brain + csf + other, ball)
    # holes, success = fill_holes(altered_tiv, aff, verbose=False)
    # holes = utils.binary_opening(holes, ball)
    # holes = utils.remove_disconnected_components(holes, inplace=True)    
    # holes[brain > 0] = 0
    # holes[csf > 0] = 0
    # holes[other > 0] = 0

    # CSF far away from brain is probably eyeball, so be more aggressive in
    # removing dangly CSF (but watch out for severe atrophy).
    ball = utils.make_structural_sphere(aff, 4 * maxscale)
    #tiv[holes > 0] = 1
    bits = utils.binary_opening(tiv, ball)
    tiv = utils.remove_disconnected_components(bits, inplace=False)
    bits[tiv > 0] = 0
    ball = utils.make_structural_sphere(aff, 2 * maxscale)
    if np.any(bits):
        bits = utils.binary_dilation(bits, ball)    
        tiv[bits > 0] = 0
    #tiv[brain > 0] = 1
    #tiv = utils.binary_closing(tiv, ball)
    brain[tiv == 0] = 0
    csf[tiv == 0] = 0
    other[tiv == 0] = 0
    # tiv = brain + csf + other
    # tiv[holes > 0] = 1

    posterity += squawk(brain, "brain at the end")
    posterity += squawk(csf, "csf at the end")
    posterity += squawk(other, "other at the end")
    
    # This doesn't work well at all.
    #brainpother = brain.copy()
    #brainpother[lsvmmask == 3] = 1
    #brainpother = utils.remove_disconnected_components(brainpother)
    #brain[brainpother == 1] = 1

    # # Cut off pial snippets, i.e. "brain" or other outside CSF.
    # No! Don't do that, because of partial voluming.  They could arguably 
    # be reclassified as CSF, but ultimately they are CSF + skull + pial surface,
    # so they are not pure CSF - I am leaving them as other for now.
    # edge = tiv.copy()
    # altered_tiv = utils.binary_erosion(tiv, ball)
    # edge[altered_tiv > 0] = 0
    # dcsf = utils.binary_dilation(csf, ball)
    # dcsf[csf > 0] = 0
    # dcsf[edge == 0] = 0
    # brain[dcsf > 0] = 0
    # other[dcsf > 0] = 0
    # shrunk_brain = brain + other
    # shrunk_brain[holes > 0] = 1
    # shrunk_brain[edge > 0] = 0
    # shrunk_brain = utils.binary_dilation(shrunk_brain, ball)
    # shrunk_brain[csf > 0] = 0
    # altered_tiv[shrunk_brain > 0] = 1
    # ball = utils.make_structural_sphere(aff, maxscale)
    # opened = utils.binary_opening(altered_tiv, ball)
    # #main_opened = utils.remove_disconnected_components(opened)
    # #bits = 
    # #shrunk_brain[altered_tiv == 0] = 0
    # opened = utils.binary_dilation(opened, ball)
    # opened[csf > 0] = 0 
    # brain[lsvmmask == 1] = 1
    # brain[opened == 0] = 0
    # other[opened == 0] = 0
    # tiv = brain + csf + other
    # tiv[holes > 0] = 1
    # tiv = utils.binary_opening(tiv, ball)
    # tiv = utils.remove_disconnected_components(tiv, inplace=True)
    # tiv = utils.binary_dilation(tiv, ball)
    # brain[tiv == 0] = 0
    # csf[tiv == 0] = 0
    # other[tiv == 0] = 0
    # holes[tiv == 0] = 0
    return brain, csf, other, holes, posterity    
