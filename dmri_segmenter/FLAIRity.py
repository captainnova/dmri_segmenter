import numpy as np
from scipy.ndimage.filters import median_filter
try:
    from skimage.filter import threshold_otsu as otsu
except:
    from dipy.segment.threshold import otsu

import utils

class FLAIRity(object):
    """
    Determines, using the voxel values, and holds whether a dMRI had its CSF
    suppressed by FLAIR.

    It can be told to skip the determination and just hold the FLAIRity.
    """
    def __init__(self, data, aff, bvals, relbthresh, maxscale, Dt, DCSF, nmed,
                 medrad, verbose=True, closerad=3.7, forced_flairity=None):
        self.data = data
        self.aff = aff
        self.bvals = bvals
        self.relbthresh = relbthresh
        self.maxscale = maxscale
        self.Dt = Dt
        self.DCSF = DCSF
        self.nmed = nmed
        self.medrad = medrad
        self.verbose = verbose
        self.closerad = closerad

        # Initialize self-caching properties
        self._flairity = forced_flairity
        self._mask = None
        self._csfmask = None

        
    @property
    def flairity(self):
        if self._flairity is None:
            self._flairity = self.guessFLAIRity()
        return self._flairity


    @property
    def mask(self):
        if self._mask is None:
            self.guessFLAIRity()  # Sets _mask as a side-effect.
        return self._mask

    @property
    def csfmask(self):
        if self._csfmask is None:
            self.guessFLAIRity()  # Sets _csfmask as a side-effect.
        return self._csfmask


    def guessFLAIRity(self):
        """
        Guess at whether data's contrast suppressed free water or not, i.e. return
        True for FLAIR DTI and False otherwise.  It operates by doing a rough
        segmentation of brain tissue and CSF, and checking whether the tissue is
        brighter than CSF for b ~ 0.

        Sets self.mask as a side-effect.
        """
        b = np.asarray(self.bvals)
        b0 = utils.calc_average_s0(self.data, b, self.relbthresh, estimator=np.median)
        scales = utils.voxel_sizes(self.aff)
        maxscale = max(scales)
        embDt = np.exp(-b * self.Dt)
        embDCSF = np.exp(-b * self.DCSF)
        w = (embDt * (1.0 - embDCSF))**2
        tisssig = np.zeros(b0.shape)
        embb0s = {}
        sw = 0.0
        for v, emb in enumerate(embDCSF):
            if emb not in embb0s:
                embb0s[emb] = emb * b0
            tisssig += w[v] * (self.data[..., v] - embb0s[emb])
            sw += w[v]
        del embb0s
        tisssig /= sw

        # Don't use median_otsu because it assumes isotropic voxels.
        if self.nmed > 0:
            if self.medrad >= 1:
                ball = utils.make_structural_sphere(self.aff, self.medrad * self.maxscale)
                if self.verbose:
                    print "Median filtering %d times with radius %f." % (self.nmed,
                                                                         self.medrad * self.maxscale)
                for i in xrange(self.nmed):
                    tisssig = median_filter(tisssig, footprint=ball)
            elif self.medrad > 0:
                print "Warning: not median filtering since medrad < 1."

        if self.verbose:
            print "Getting the Otsu threshold."
        thresh = otsu(tisssig)
        self._mask = np.zeros(tisssig.shape, np.bool)
        self._mask[tisssig >= thresh] = 1

        ball = utils.make_structural_sphere(self.aff, max(10.0, self.maxscale))
        self._csfmask = utils.binary_closing(self._mask, ball)
        gaprad = max(self.closerad, 2 * self.maxscale)
        self._csfmask, success = utils.fill_holes(self._csfmask, self.aff, gaprad, self.verbose)
        self._csfmask = utils.binary_opening(self._csfmask, ball)
        self._csfmask[self._mask > 0] = 0
        csfmed = np.median(b0[self._csfmask > 0])
        b0tiss = b0[self._mask > 0]

        # Now we have an approximate brain, and we know it is surrounded by CSF
        # (in vivo) or solution (ex vivo), which we'll call CSF.  Figure out
        # whether CSF is brighter or darker than tissue in the b0.
        tissmed = np.median(b0tiss)
        return tissmed > 2.0 * csfmed
