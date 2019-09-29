from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
import datetime
import dipy.io
import numpy as np
import nibabel as nib
import os
import scipy.ndimage as ndi
#import sys
import sklearn                  # just to get .__version__
from sklearn import ensemble
#import sklearn.externals.joblib as joblib

try:
    from skimage.filter import threshold_otsu as otsu
except Exception:
    from dipy.segment.threshold import otsu

from . import brine
from . import dmri_brain_extractor as dbe
from . import utils

trainees = {'RFC 20000 Siemens 0_0': 'rfc_20000_Siemens_0_0.pickle',
            'RFC 100000 Siemens 0_0': 'rfc_100000_Siemens_0_0.pickle',
            'RFC GE 2.0mm': 'rfc_GE2.pickle',
            'RFC GE': 'rfc_GE.pickle',
            'RFC GE and Philips': 'rfc_GE_and_Philips.pickle',
            'RFC GE and Siemens': 'rfc_GE_and_Siemens.pickle',
            'RFC Philips 2.0mm': 'rfc_Philips2.pickle',
            'RFC Philips': 'rfc_Philips.pickle',
            'RFC Philips and Siemens': 'rfc_Philips_and_Siemens.pickle',
            'RFC Siemens 2.0mm': 'rfc_Siemens_seq0.pickle',
            'RFC Siemens': 'rfc_Siemens.pickle',
            'RFC all': 'rfc_all.pickle'}


def get_bvals(dwfn, bvalpat='*bval*'):
    """
    Get the diffusion strengths for each volume in dwfn.
    It assumes there is exactly 1 b values file matching either
    os.path.splitext(dwfn)[0] + bvalpat or just
    os.path.dirname(dwfn) + bvalpat.

    Parameters
    ----------
    dwfn: str
        Filename of a 4D diffusion MRI .nii.
    bvalpat: str
        glob pattern for finding the b values of dwfn, written
        as an ASCII file in FSL/dipy format.
        There must be exactly 1 match.
    """
    try:
        bvalfn = utils.get_1_file_or_hurl(os.path.splitext(dwfn)[0] + bvalpat)
    except Exception:
        bvalfn = utils.get_1_file_or_hurl(os.path.join(os.path.dirname(dwfn), bvalpat))
    bvals, _ = dipy.io.read_bvals_bvecs(bvalfn, None)
    return bvals


def make_fvecs(dwfn, bthresh=0.02, smoothrad=10.0, s0=None, Dt=0.0021,
               Dcsf=0.00305, blankval=0, clamp=30, normslop=0.2,
               logclamp=-10, outlabel='fvecs'):
    bvals = get_bvals(dwfn)
    dwnii = nib.load(dwfn)
    aff = dwnii.affine
    data = dwnii.get_data()

    fvecs, posterity = dbe.make_feature_vectors(data, aff, bvals, smoothrad=smoothrad)

    outfn = dwfn.replace('.nii', '_%s.nii' % outlabel)
    outnii = nib.Nifti1Image(fvecs, aff)
    outnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                               posterity.encode('utf-8')))
    nib.save(outnii, outfn)
    return outfn


def edited_to_segmentation(tivfn, brfn, s0fn):
    outfn = tivfn.replace('.nii', '_segmentation.nii')
    brnii = nib.load(brfn)
    brain = brnii.get_data()
    tiv = nib.load(tivfn).get_data()
    seg = np.zeros_like(tiv)
    csf = tiv.copy()
    csf[brain > 0] = 0
    s0 = nib.load(s0fn).get_data()
    s0[s0 < 0.01] = 0.01
    csf_other_thresh = np.exp(otsu(np.log(s0[csf > 0])))
    other = csf.copy()
    csf[s0 < csf_other_thresh] = 0
    other[csf > 0] = 0
    seg[brain > 0] = 1
    seg[csf > 0] = 2
    seg[other > 0] = 3
    nib.save(nib.Nifti1Image(seg, brnii.affine), outfn)
    return outfn


def gather_svm_samples(svecs, tmask, maxperclass=100000,
                       tmasktype=np.int8, verbose=False):
    """
    Gather samples for each segmentation class.

    Parameters
    ----------
    svecs: (nx, ny, nz, len(feature vector)) array
        The feature vector field
    tmask: (nx, ny, nz) array of ints
        The segmentation labels (classes) for each voxel.
    maxperclass: int
        The maximum number of samples for each class.
        100000 works well for RandomForestClassifier, which works better than
        *SVM_CV or AdaBoost anyway.  (no tuning was done for AdaBoost, though.)
    tmasktype: type
        The type that tmask will be internally cast to.

    Output
    ------
    samps: (nsamples, len(feature vector)) array
        A subset of svecs
    targets: (nsamples,) array of ints
        The corresponding segmentation classes
    """
    nvox = np.prod(tmask.shape)

    if verbose:
        print("svecs.shape: %s" % svecs.shape)

    sfsvecs = svecs.reshape((nvox, svecs.shape[-1]))   # Reshaped feature vectors
    ftargs = tmask.reshape((nvox,)).astype(tmasktype)  # Flattened segmentations
    mint = np.min(ftargs)                              # Minimum segmentation class
    maxt = np.max(ftargs)                              # Maximum segmentation class
    samps = np.empty((0, svecs.shape[-1]))             # Make a stub to append to.
    targets = []                                       # Segmentation class for each sample
    for t in range(mint, maxt + 1):                   # for each class,
        tsamps = sfsvecs[ftargs == t]                  # feature vectors matching class
        ntsamps = len(tsamps)
        if ntsamps > maxperclass:
            rows = np.random.randint(0, ntsamps, maxperclass)
            tsamps = tsamps[rows]
            ntsamps = maxperclass
        samps = np.vstack((samps, tsamps))        # Append tsamps to samps
        targets += [t] * ntsamps                  # Annotate them
    return samps, np.array(targets)


def make_segmentation(fvecsfn, fvcfn, custom_label=False, outfn=None,
                      useT1=False):
    fnii = nib.load(fvecsfn)
    aff = fnii.affine
    fvecs = fnii.get_data()

    if useT1:
        t1wtiv = fvecsfn.replace('dtb_eddy_fvecs.nii',
                                 'bdp/dtb_eddy_T1wTIV.nii')
    else:
        t1wtiv = None

    # os.path.abspath is idempotent.
    clf = brine.debrine(os.path.abspath(fvcfn))

    seg, probs, posterity = dbe.probabilistic_classify(fvecs, aff, clf,
                                                       t1wtiv=t1wtiv)

    if outfn is None:
        if custom_label:
            outfn = fvecsfn.replace('_fvecs',
                                    '_' + os.path.basename(fvcfn).replace('.pickle', ''))
        else:
            outfn = fvecsfn.replace('_fvecs.nii', '_rfcseg.nii')
    outdir, outbase = os.path.split(outfn)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    outnii = nib.Nifti1Image(seg, aff)
    outnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                               posterity.encode('utf-8')))
    nib.save(outnii, outfn)
    return outfn


def gather_error_samples(svecs, trial, gold, maxperclass=5000,
                         tmasktype=np.int8):
    """
    Like gather_svm_samples, but favors samples that were incorrectly
    classified in trial.

    Parameters
    ----------
    svecs: (nx, ny, nz, len(feature vector)) array
        The feature vector field
    trial: (nx, ny, nz) array of ints
        The segmentation labels (classes) for each voxel from a possibly
        imperfect segmentation.
    gold: (nx, ny, nz) array of ints
        The segmentation labels (classes) for each voxel from a supposedly
        imperfect segmentation.
    maxperclass: int
        The maximum number of samples for each class.

    Output
    ------
    samps: (nsamples, len(feature vector)) array
        A subset of svecs
    targets: (nsamples,) array of ints
        The corresponding segmentation classes
    """
    mint = np.min(gold)
    maxt = np.max(gold)
    nvox = np.prod(gold.shape)
    sfsvecs = svecs.reshape((nvox, svecs.shape[-1]))  # Reshaped feature vectors
    flattrial = trial.reshape((nvox,)).astype(tmasktype)
    flatgold = gold.reshape((nvox,)).astype(tmasktype)
    errmask = flattrial != flatgold
    svecs_err = sfsvecs[errmask]
    svecs_ok = sfsvecs[~errmask]
    trial_err = flattrial[errmask]
    #trial_ok = flattrial[~errmask]
    gold_err = flatgold[errmask]
    gold_ok = flatgold[~errmask]
    samps = np.empty((0, svecs.shape[-1]))            # Make a stub to append to.
    targets = []                                      # Segmentation class for each sample

    # I wanted to use collections.OrderedDict for notes and collections.Counter
    # for filling it, but both were introduced in python 2.7 and we're still
    # using 2.6.
    notes = {}
    for t in range(mint, maxt + 1):
        tsamps = svecs_err[gold_err == t]
        ntsamps = len(tsamps)
        ttargs_err = trial_err[gold_err == t]
        notes[t] = {"# in class": sum(flatgold == t),
                    "# of errors": ntsamps,
                    "available errors": dict([(k, sum(ttargs_err == k)) for k in range(mint, maxt + 1)
                                              if k != t])}
        notes[t]['number sampled'] = notes[t]["available errors"].copy()  # Only if ntsamps == maxperclass
        if ntsamps > maxperclass:
            rows = np.random.randint(0, ntsamps, maxperclass)
            tsamps = tsamps[rows]
            ntsamps = maxperclass
            ttargs_err = ttargs_err[rows]
            for k in notes[t]['number sampled']:
                notes[t]['number sampled'][k] = sum(ttargs_err == k)
        elif ntsamps < maxperclass:                                      # Supplement with nonerrors.
            n_wanted = maxperclass - ntsamps
            samps_ok = svecs_ok[gold_ok == t]
            n_ok = len(samps_ok)
            if n_ok > n_wanted:
                rows = np.random.randint(0, n_ok, n_wanted)
                samps_ok = samps_ok[rows]
            tsamps = np.vstack((tsamps, samps_ok))
            notes[t]['number sampled'][t] = min(n_ok, n_wanted)
        samps = np.vstack((samps, tsamps))        # Append tsamps to samps
        ntsamps = len(tsamps)
        targets += [t] * ntsamps                  # Annotate them
        notes[t]['total sampled'] = ntsamps

    return samps, np.array(targets), notes


def train(srclist, label, maxperclass=100000, class_weight="balanced_subsample",
          smoothrad=10.0, srclist_is_srcdirs=False, fvfn='dtb_eddy_fvecs.nii',
          rT1TIVfn=None, t1fwhm=[2.0, 10.0, 2.0], n_estimators=10,
          max_features='auto',  # 'auto' = sqrt(n_features)
          max_depth=24, min_samples_split=2, min_samples_leaf=1,
          n_jobs=None, srcroot='training', segfn='dmri_segment_edited.nii',
          min_weight_fraction_leaf=5e-5, nstages=2):
    """
    Parameters
    ----------
    srclist: str or list of strs
        Source *directories* with both feature vector and segmented .niis.
        If a str, it is taken as the name of a file listing the
        directories one per line.
    label: str
        The classifier parameters will be written as a modified pickle to
        RFC_<label>.pickle
    maxperclass: int
        The maximum number of samples per class.
    class_weight: str
        See ensemble.RandomForestClassifier, but note that the number of voxels
        in each class is typically fairly imbalanced.
    smoothrad: float
        FWHM in mm of the Gaussian kernel to use for making smoothed class
        probabilities to append to the feature vectors used in the 2nd and
        3rd stage classifications.
    srclist_is_srcdirs: bool
        Iff True, do not prepend srcroot to the directories in srclist.
    fvfn: str
        Name of the feature vectors image in each directory of srclist.
    rT1TIVfn: None or str
        Name of the T1 TIV registered to diffusion space in each directory
        of srclist, or None to not use T1 TIVs.
    t1fwhm: sequence of 3 floats
        FWHM in mm of the Gaussian kernel to blur rT1TIVfn by to make it
        a prior.  It should typically be much larger in the phase encoding
        direction because of EPI distortion.
    n_estimators: int
        The number of trees in the forest.  10 works well for allowing
        probabilities for the 2nd stage to be calculated using the
        parliament of trees from the 1st stage.
    max_features: int, float, string or None, optional (default="auto"="sqrt")
        The number of features to consider when looking for the best split.
        Note that random forest classifiers work in part by having different
        trees consider different features.
        See ensemble.RandomForestClassifier
    max_depth: int or None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        See ensemble.RandomForestClassifier
    min_samples_split: int
        The minimum number of samples required to split an internal node.
    min_samples_leaf: int
        The minimum number of samples in newly created leaves.
    n_jobs: None or int
        The number of parallel jobs to use.  If None it will be determined
        using utils.suggest_number_of_processors().
    srcroot: str
        Directory holding the directories in srclist.
    segfn: str
        Name of the segmented image to use for training in each directory of
        srclist.
    nstages: int
        How many stages to train.

    WARNING! This assumes that srclist is short, since it holds all of
             srclist's svecs in memory.

            If srclist is long the algorithm should be changed to only
            hold 1 svec image at a time, even though that means
            rereading them.
    """
    if isinstance(srclist, str):
        with open(srclist) as f:
            srclist = [line.strip() for line in f]
    if len(srclist) > 24:  # Even 24 might be too large
        raise ValueError("""
    This function is not currently written for long srclists.
    See the help.
    """)
    if not n_jobs:
        n_jobs = utils.suggest_number_of_processors()
    res = {'src_properties': []}
    svecslist = []
    tmasklist = []
    afflist = []
    samplist = []
    targlist = []
    volslist = []
    utils.instaprint("Beginning the 1st stage")
    for src in srclist:
        if not srclist_is_srcdirs:
            vols = os.path.join(srcroot, src)
        else:
            vols = src
        volslist.append(vols)
        snii = nib.load(os.path.join(vols, fvfn))
        afflist.append(snii.affine)
        svecs = snii.get_data()
        svecslist.append(svecs)
        tmasklist.append(nib.load(os.path.join(vols, segfn)).get_data())
        samps, targets = gather_svm_samples(svecslist[-1], tmasklist[-1],
                                            maxperclass=maxperclass)
        nclasses = max(targets) + 1
        res['src_properties'].append([])
        for c in range(nclasses):
            sieve = (targets == c)
            n = sum(sieve)
            m = samps[sieve, 0].mean()  # 0 for s0.
            res['src_properties'][-1].append({'n': n, 's0 level': m})
        samplist.append(samps)
        targlist.append(targets)

    res['s0 brain level'] = np.average([t[1]['s0 level']     # 1 for brain
                                        for t in res['src_properties']],
                                       weights=[t[1]['n']    # 1 for brain
                                                for t in res['src_properties']])
    catsamps = None
    for i in range(len(srclist)):
        targets = targlist[i]

        # # Recalibrate s0 to bring the samples to a common brightness level.
        # # There is still potentially a difference in the CSF/brain brightness
        # # ratio from scan to scan if TE varies.
        # delta = props[1]['s0 level'] - res['s0 brain level']
        # props = res['src_properties'][i]
        # samplist[i][:, :1] -= delta
        # svecslist[i][..., :1] -= delta

        if catsamps is None:
            catsamps = samplist[i]
            cattargs = targets
        else:
            catsamps = np.vstack((catsamps, samplist[i]))
            cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.

    res['n_features'] = catsamps.shape[-1]
    res['n_classes'] = max(cattargs) + 1
    res['smoothrad'] = smoothrad

    res['1st stage'] = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                       class_weight=class_weight,
                                                       max_features=max_features,
                                                       max_depth=max_depth,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                       n_jobs=n_jobs)
    res['1st stage'].fit(catsamps, cattargs)
    res['log']  = "RandomForestClassifiers trained from\n\t"
    res['log'] += "\n\t".join(srclist) + "\n"
    res['log'] += "with maxperclass = %d each.\n" % maxperclass

    if nstages > 1:
        # Train the 2nd stage
        catsamps = None
        cattargs = None
        utils.instaprint("Beginning the 2nd stage")
        for svecs, tmask, aff, vols in zip(svecslist, tmasklist, afflist, volslist):
            svecsr = np.reshape(svecs, (np.prod(svecs.shape[:3]), svecs.shape[3]))
            probs = res['1st stage'].predict_proba(svecsr)
            probs = np.reshape(probs, svecs.shape[:3] + (probs.shape[-1],))

            if rT1TIVfn:
                t1sigma = dbe.fwhm_to_voxel_sigma(t1fwhm, afflist[-1])
                t1wtiv = os.path.join(vols, rT1TIVfn)
                svecs2 = np.empty(svecs.shape[:3] + (svecs.shape[3] + probs.shape[-1] + 1,))
                ndi.filters.gaussian_filter(nib.load(t1wtiv).get_data(), sigma=t1sigma,
                                            output=svecs2[..., -1], mode='nearest')
            else:
                svecs2 = np.empty(svecs.shape[:3] + (svecs.shape[3] + probs.shape[-1],))
            svecs2[..., :svecs.shape[3]] = svecs
            sigma = dbe.fwhm_to_voxel_sigma(smoothrad, aff)
            for v in range(probs.shape[-1]):
                ndi.filters.gaussian_filter(probs[..., v], sigma=sigma,
                                            output=svecs2[..., v + svecs.shape[3]],
                                            mode='nearest')
            # svecs2 = np.empty(svecs.shape[:3] + (12,))
            # svecs2[..., :4] = svecs
            # sigma = dbe.fwhm_to_voxel_sigma(smoothrad, aff)
            # for v in xrange(4):
            #     ndi.filters.gaussian_filter(probs[..., v], sigma=sigma, output=svecs2[..., v + 4],
            #                                 mode='nearest')
            #     ndi.filters.gaussian_filter(probs[..., v], sigma=2 * sigma, output=svecs2[..., v + 8],
            #                                 mode='nearest')
            # em10 = np.exp(-10)
            # svecs2[..., 4:][svecs2[..., 4:] < em10] = em10
            # svecs2[..., 4:] = 1 + np.log(svecs2[..., 4:])
            samps, targets = gather_svm_samples(svecs2, tmask, maxperclass=maxperclass)
            if catsamps is None:
                catsamps = samps
                cattargs = targets
            else:
                catsamps = np.vstack((catsamps, samps))
                cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.
        res['2nd stage'] = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                           class_weight=class_weight,
                                                           max_features=max_features,
                                                           max_depth=max_depth,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf,
                                                           n_jobs=n_jobs)
        res['2nd stage'].fit(catsamps, cattargs)

    if nstages > 2:
        # Train the 3rd stage
        catsamps = None
        cattargs = None
        utils.instaprint("Beginning the 3rd stage")
        for svecs, tmask, aff, vols in zip(svecslist, tmasklist, afflist, volslist):
            seg, probs, clog = dbe.probabilistic_classify(svecs, aff, res,
                                                          t1wtiv=rT1TIVfn, t1fwhm=t1fwhm)

            augvecs = np.empty(svecs.shape[:3] + (svecs.shape[3] + probs.shape[-1],))
            augvecs[..., :svecs.shape[-1]] = svecs

            # probs from the 2nd stage is better than probs from the 1st stage, but
            # less useful than seg, which has benefitted from morphological mojo.
            # Eschew probs in favor of a smoothed seg.
            sigma = dbe.fwhm_to_voxel_sigma(smoothrad, aff)
            for v in range(probs.shape[-1]):
                unsmoothed = np.zeros(seg.shape)
                unsmoothed[seg == v] = 1.0
                ndi.filters.gaussian_filter(unsmoothed, sigma=sigma,
                                            output=augvecs[..., v + svecs.shape[-1]], mode='nearest')

            samps, targets = gather_svm_samples(augvecs, tmask, maxperclass=maxperclass)
            if catsamps is None:
                catsamps = samps
                cattargs = targets
            else:
                catsamps = np.vstack((catsamps, samps))
                cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.
        res['3rd stage'] = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                           class_weight=class_weight,
                                                           max_features=max_features,
                                                           max_depth=max_depth,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf,
                                                           n_jobs=n_jobs)
        res['3rd stage'].fit(catsamps, cattargs)

    blabel = os.path.basename(label)
    if blabel[:4] != 'RFC_':
        label = label.replace(blabel, 'RFC_' + blabel)

    #if not os.path.isdir(label):
    #    os.makedirs(label)
    pfn = label + '.pickle'
    logfn = label + '_training.log'
    res['log'] += "and pickled to %s.\n" % pfn
    #joblib.dump(res, pfn, compress=compress)
    brine.brine(res, pfn)

    res['log'] += "\nClassifier attributes:\n"
    for k, v in res.items():
        if k != 'log':
            res['log'] += "\t%s:\n\t\t%s\n" % (k, v)
    res['log'] += "\nTrained %s on %s using sklearn %s.\n" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                              os.uname()[1], sklearn.__version__)
    with open(logfn, 'w') as lf:
        lf.write(res['log'])
    return res, pfn, logfn
