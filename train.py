import dipy.io
import numpy as np
import nibabel as nib
import os
import scipy.ndimage as ndi
import sys
from sklearn import ensemble
import sklearn.externals.joblib as joblib

try:
    from skimage.filter import threshold_otsu as otsu
except:
    from dipy.segment.threshold import otsu

import dmri_brain_extractor as dbe
import utils

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
    It assumes there is a b values file matching
    os.path.splitext(dwfn)[0] + bvalpat.

    Parameters
    ----------
    dwfn: str
        Filename of a 4D diffusion MRI .nii.
    bvalpat: str
        glob pattern for finding the b values of dwfn, written
        as an ASCII file in FSL/dipy format.
        There must be exactly 1 match.
    """
    bvalfn = utils.get_1_file_or_hurl(os.path.splitext(dwfn)[0] + bvalpat)
    bvals, _ = dipy.io.read_bvals_bvecs(bvalfn, None)
    return bvals

def make_fvecs(dwfn, bthresh=0.02, smoothrad=10.0, s0=None, Dt=0.0021,
                Dcsf=0.00305, blankval=0, clamp=30, normslop=0.2,
                logclamp=-10, outlabel='fvecs'):
    bvals = get_bvals(dwfn)
    dwnii = nib.load(dwfn)
    aff = dwnii.affine
    data = dwnii.get_data()
    
    fvecs = dbe.make_feature_vectors(data, aff, bvals, bvecs, smoothrad=smoothrad)
    tlogclamp = 10**logclamp
    fvecs[fvecs < tlogclamp] = tlogclamp
    fvecs = np.log10(fvecs)
    posterity  = "Logarithmic support vectors made with:\n"
    posterity += "\tbthresh = %f\n" % bthresh
    posterity += "\tsmoothrad = %f mm\n" % smoothrad
    posterity += "\tDt = %f\n" % Dt
    posterity += "\tDcsf = %f\n" % Dcsf
    posterity += "\tblankval = %f\n" % blankval
    posterity += "\tclamp = %f\n" % clamp
    posterity += "\tnormslop = %f\n" % normslop
    posterity += "\tlogclamp = %f\n" % logclamp
    
    outfn = dwfn.replace('.nii', '_%s.nii' % outlabel)
    outnii = nib.Nifti1Image(fvecs, aff)
    outnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                                     posterity))
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
    tmasktype: type
        The type that tmask will be internally cast to.

    Output
    ------
    samps: (nsamples, len(feature vector)) array
        A subset of svecs
    targets: (nsamples,) array of ints
        The corresponding segmentation classes
    """
    # 100000 works well for RandomForestClassifier, which works better than
    # *SVM_CV or AdaBoost anyway.  (no tuning was done for AdaBoost, though.)
    nvox = np.prod(tmask.shape)

    if verbose:
        print "svecs.shape:", svecs.shape
    
    sfsvecs = svecs.reshape((nvox, svecs.shape[-1]))  # Reshaped feature vectors
    ftargs = tmask.reshape((nvox,)).astype(tmasktype) # Flattened segmentations
    mint = np.min(ftargs)                             # Minimum segmentation class
    maxt = np.max(ftargs)                             # Maximum segmentation class
    samps = np.empty((0, svecs.shape[-1]))            # Make a stub to append to.
    targets = []                                      # Segmentation class for each sample
    for t in xrange(mint, maxt + 1):                  # for each class,
        tsamps = sfsvecs[ftargs == t]                 # feature vectors matching class
        ntsamps = len(tsamps)
        if ntsamps > maxperclass:
            rows = np.random.randint(0, ntsamps, maxperclass)
            tsamps = tsamps[rows]
            ntsamps = maxperclass
        samps = np.vstack((samps, tsamps))        # Append tsamps to samps
        targets += [t] * ntsamps                  # Annotate them 
    return samps, np.array(targets)

def make_segmentation(fvecsfn, fvcfn, custom_label=False, outfn=None, useT1=False):
    aff = nib.load(fvecsfn).get_affine()

    if useT1:
        t1wtiv = fvecsfn.replace('dtb_eddy_fvecs.nii', 'bdp/dtb_eddy_T1wTIV.nii')
    else:
        t1wtiv = None
    
    # os.path.abspath is idempotent.
    brain, csf, other, holes, posterity = dbe.feature_vector_classify(None, aff, None,
                                                                      clf=os.path.abspath(fvcfn),
                                                                      fvecs=fvecsfn,
                                                                      t1wtiv=t1wtiv)

    seg = np.zeros_like(brain)
    seg[brain > 0] = 1
    seg[csf > 0] = 2
    seg[other > 0] = 3

    if outfn is None:
        if custom_label:
            outfn = fvecsfn.replace('_fvecs', '_' + fvcfn.replace('.joblib_dump', ''))
        else:
            outfn = fvecsfn.replace('_fvecs.nii', '_rfcseg.nii')
    outdir, outbase = os.path.split(outfn)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    outnii = nib.Nifti1Image(seg, aff)
    outnii.header.extensions.append(nib.nifti1.Nifti1Extension('comment',
                                                               posterity))
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
    for t in xrange(mint, maxt + 1):
        tsamps = svecs_err[gold_err == t]
        ntsamps = len(tsamps)
        ttargs_err = trial_err[gold_err == t]
        notes[t] = {"# in class": sum(flatgold == t),
                    "# of errors": ntsamps,
                    "available errors": dict([(k, sum(ttargs_err == k)) for k in xrange(mint, maxt + 1)
                                              if k != t])}
        notes[t]['number sampled'] = notes[t]["available errors"].copy() # Only if ntsamps == maxperclass
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

def train_from_multiple(srclist, label, maxperclass=100000, class_weight="balanced_subsample",
                        compress=0):
    """
    Parameters
    ----------
    compress: int from 0 to 9, optional
        Compression level for the data. 0 is no compression.
        We use a compressed filesystem at ADIR and do not want
        explicit compression.
        Higher means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        For more details see
        http://gael-varoquaux.info/programming/new_low-overhead_persistence_in_joblib_for_big_data.html .
    """
    for i, src in enumerate(srclist):
        vols = 'training/' + src
        svecs = nib.load(os.path.join(vols, 'dtb_eddy_fvecs.nii')).get_data()
        tmask = nib.load(os.path.join(vols, 'dtb_eddy_T1wTIV_edited_segmentation.nii')).get_data()
        samps, targets = dbe.gather_svm_samples(svecs, tmask, maxperclass=maxperclass)
        if i == 0:
            catsamps = samps
            cattargs = targets
        else:
            catsamps = np.vstack((catsamps, samps))
            cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.
    clf = ensemble.RandomForestClassifier(class_weight=class_weight)
    clf.fit(catsamps, cattargs)
    if label[:4] != 'RFC_':
        label = 'RFC_' + label
    pfn = label + '.joblib_dump'
    joblib.dump(clf, pfn)
    log  = "RandomForestClassifier trained from\n\t"
    log += "\n\t".join(srclist) + "\n"
    log += "with maxperclass = %d\n" % maxperclass
    log += "and pickled to %s.\n" % pfn
    logfn = 'training_' + label + '.log'
    with open(logfn, 'w') as lf:
        lf.write(log)
    return pfn, logfn

def train_both_stages_from_multiple(srclist, label, maxperclass=100000,
                                    class_weight="balanced_subsample",
                                    smoothrad=10.0, srclist_is_srcdirs=False,
                                    fvecs_fn='dtb_eddy_fvecs.nii',
                                    useT1=False, t1fwhm=[2.0, 10.0, 2.0],
                                    n_estimators=10,
                                    max_features='auto', # 'auto' = sqrt(n_features)
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    n_jobs=None):
    """
    Parameters
    ----------
    n_jobs: None or int
        The number of parallel jobs to use.  If None it will be determined
        using utils.suggest_number_of_processors().

    WARNING! This assumes that srclist is short, since it holds all of
             srclist's svecs in memory.

            If srclist is long the algorithm should be changed to only
            hold 1 svec image at a time, even though that means
            rereading them.
    """
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
    for src in srclist:
        if not srclist_is_srcdirs:
            vols = 'training/' + src
        else:
            vols = src
        volslist.append(vols)
        snii = nib.load(os.path.join(vols, fvecs_fn))
        afflist.append(snii.get_affine())
        svecs = snii.get_data()
        svecslist.append(svecs)
        tmasklist.append(nib.load(os.path.join(vols,
                                               'dtb_eddy_T1wTIV_edited_segmentation.nii')).get_data())
        samps, targets = dbe.gather_svm_samples(svecslist[-1], tmasklist[-1], maxperclass=maxperclass)
        nclasses = max(targets) + 1
        res['src_properties'].append([])
        for c in xrange(nclasses):
            sieve = (targets == c)
            n = sum(sieve)
            m = samps[sieve, 1].mean()  # 1 for brain
            res['src_properties'][-1].append({'n': n, 's0 level': m})
        samplist.append(samps)
        targlist.append(targets)

    # Recalibrate s0 to bring the samples to a common brightness level.  There
    # is still potentially a difference in the CSF/brain brightness ratio from
    # scan to scan if TE varies.
    res['s0 brain level'] = np.average([t[1]['s0 level'] for t in res['src_properties']],   # 1 for brain
                                       weights=[t[1]['n'] for t in res['src_properties']])  # 1 for brain
    catsamps = None
    for i in xrange(len(srclist)):
        targets = targlist[i]
        props = res['src_properties'][i]
        delta = props[1]['s0 level'] - res['s0 brain level']
        samplist[i][:, :1] -= delta
        svecslist[i][..., :1] -= delta
        
        if catsamps is None:
            catsamps = samplist[i]
            cattargs = targets
        else:
            catsamps = np.vstack((catsamps, samplist[i]))
            cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.

    res['n_features'] = catsamps.shape[-1]
    res['n_classes'] = max(cattargs) + 1
            
    res['1st stage'] = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                       class_weight=class_weight,
                                                       max_features=max_features,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       n_jobs=n_jobs)
    res['1st stage'].fit(catsamps, cattargs)
    res['log']  = "RandomForestClassifiers trained from\n\t"
    res['log'] += "\n\t".join(srclist) + "\n"
    res['log'] += "with maxperclass = %d each.\n" % maxperclass
    catsamps = None
    cattargs = None
    for svecs, tmask, aff, vols in zip(svecslist, tmasklist, afflist, volslist):
        svecsr = np.reshape(svecs, (np.prod(svecs.shape[:3]), svecs.shape[3]))
        probs = res['1st stage'].predict_proba(svecsr)
        probs = np.reshape(probs, svecs.shape[:3] + (probs.shape[-1],))
        #if morpho1to2:
            
        if useT1:
            t1sigma = dbe.fwhm_to_voxel_sigma(t1fwhm, afflist[-1])
            t1wtiv = os.path.join(vols, 'bdp/dtb_eddy_T1wTIV.nii')
            svecs2 = np.empty(svecs.shape[:3] + (svecs.shape[3] + probs.shape[-1] + 1,))
            ndi.filters.gaussian_filter(nib.load(t1wtiv).get_data(), sigma=t1sigma,
                                        output=svecs2[..., -1], mode='nearest')
        else:
            svecs2 = np.empty(svecs.shape[:3] + (svecs.shape[3] + probs.shape[-1],))
        svecs2[..., :svecs.shape[3]] = svecs
        sigma = dbe.fwhm_to_voxel_sigma(smoothrad, aff)
        for v in xrange(probs.shape[-1]):
            ndi.filters.gaussian_filter(probs[..., v], sigma=sigma, output=svecs2[..., v + svecs.shape[3]],
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
        samps, targets = dbe.gather_svm_samples(svecs2, tmask, maxperclass=maxperclass)
        if catsamps is None:
            catsamps = samps
            cattargs = targets
        else:
            catsamps = np.vstack((catsamps, samps))
            cattargs = np.concatenate((cattargs, targets))  # vstack doesn't work with 1d arrays.
    res['2nd stage'] = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                       class_weight=class_weight,
                                                       max_features=max_features,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       n_jobs=n_jobs)
    res['2nd stage'].fit(catsamps, cattargs)

    blabel = os.path.basename(label)
    if blabel[:4] != 'RFC_':
        label = label.replace(blabel, 'RFC_' + blabel)
    pfn = label + '.joblib_dump'
    logfn = label + '_training.log'
    res['log'] += "and serialized (pickled) to %s.\n" % pfn
    joblib.dump(res, pfn)
    with open(logfn, 'w') as lf:
        lf.write(res['log'])
    return res, pfn, logfn

