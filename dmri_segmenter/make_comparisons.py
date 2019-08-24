# Functions used to compare dmri_brain_extractor to bet, dwi2mask, etc. for the paper.
# Except for jaccard_index() they are unlikely to be used again.
import numpy as np
import nibabel as nib
import os
import re
import shutil

import brine


def _cond_to_mask(seg, cond):
    mask = np.zeros_like(seg)
    mask[seg == cond] = 1
    return mask


def dice(a, b):
    """a and b must be binary!"""
    return np.sum(np.asarray(a) * np.asarray(b)) * 2.0 / (np.sum(a) + np.sum(b))


def jaccard_index(a, b):
    """a and b must be binary!"""
    u = np.asarray(a).copy()
    u[b > 0] = 1
    return float(np.sum(np.asarray(a) * np.asarray(b))) / u.sum()


def relative_error(trial, gold):
    t = np.asarray(trial, dtype=np.int8)    # Make sure it's signed!
    g = np.asarray(gold, dtype=np.int8)

    # This formula is incorrect for multivalued segmentations, since it weights
    # other-to-air errors 3x as much as e.g. air-to-brain.
    # err = np.abs(t - g).sum()
    # float(err) / g.sum()

    err = np.zeros(g.shape, np.bool)
    err[t != g] = True
    err = err.sum()

    return float(err) / len(g[g > 0])


def arr_from_file_or_arr(foa):
    if isinstance(foa, str):
        return nib.load(foa).get_data()
    else:
        return foa


def get_re_dc_and_ji(afn, bfn):
    a = arr_from_file_or_arr(afn)
    b = arr_from_file_or_arr(bfn)
    return relative_error(a, b), dice(a, b), jaccard_index(a, b)


def make_comparisons(trial='bet', gold_standard='dtb_eddy_T1wTIV_edited', branch='test'):
    outfn = "%s_vs_%s_in_%s.csv" % (trial.replace('/', '_'), gold_standard, branch)
    with open(outfn, 'w') as outf:
        outf.write("Manufacturer,Subject,Jaccard Index,Dice Coefficient,Relative Error\n")
        for manuf in ['ge', 'philips', 'siemens']:
            for subj in ['0_0', '0_1', '1_0', '1_1']:
                niidir = os.path.join(branch, manuf, subj)
                afn = os.path.join(niidir, trial + '.nii')
                bfn = os.path.join(niidir, gold_standard + '.nii')
                relerr, dc, ji = get_re_dc_and_ji(afn, bfn)
                outf.write("%s,%s,%f,%f,%f\n" % (manuf, subj, ji, dc, relerr))
    return outfn


def make_all_untrained_comparisons(trials=['bet', 'bet_bc_mask', 'dwi2mask', 'dwi2mask_bc',
                                           'bdp/dtb_eddy_T1wTIV'],
                                   gold_standard='dtb_eddy_T1wTIV_edited',
                                   gold_standard_root='gold_standards'):
    """
    Run in crossvalidation/.
    """
    outfns = {}
    outfhs = {}
    try:
        for trial in trials:
            outfns[trial] = "all_%s_vs_%s.csv" % (os.path.split(trial)[-1], gold_standard)
            outfhs[trial] = open(outfns[trial], 'w')
            outfhs[trial].write(
                "Manufacturer,Sequence,Subject,Jaccard Index,Dice Coefficient,Relative Error\n")
        for manuf in ['ge', 'philips', 'siemens']:
            for seq in '01':
                for subj in '0123':
                    subdir = os.path.join(gold_standard_root, manuf, seq, subj)
                    gfn = os.path.join(subdir, gold_standard + '.nii')
                    gold = nib.load(gfn).get_data()
                    for trial in trials:
                        tfn = os.path.join(subdir, trial + '.nii')
                        relerr, dc, ji = get_re_dc_and_ji(tfn, gold)
                        outfhs[trial].write("%s,%s,%s,%f,%f,%f\n" % (manuf, seq, subj, ji, dc, relerr))
    finally:
        for outf in outfhs.itervalues():
            outf.close()
    return outfns


def make_cv_untrained_comparisons(trials=['bet', 'bet_bc_mask',
                                          'dwi2mask', 'dwi2mask_bc',
                                          'dtb_eddy_T1wTIV'], gold_standard='dtb_eddy_T1wTIV_edited'):
    """
    Select entries from crossvalidation/all_*_vs_dtb_eddy_T1wTIV_edited.csv
    matching the scans used to crossvalidate <label>.

    Run in crossvalidation/<label>/.
    """
    # Find the crossvalidation scans by reading ./qmake_segmentations.sh
    segs = []
    with open('qmake_segmentations.sh') as f:
        for line in f:
            parts = line.split()
            for p in parts:
                if p[-7:] == 'seg.nii':
                    segs.append(p)  # 5/philips/1/3/seg.nii = trial/manuf/seq/subj/seg.nii

    # Predict an index from (manuf, seq, subj) to row number in one of
    # crossvalidation/all_*_vs_dtb_eddy_T1wTIV_edited.csv.
                rownum = 1     # 0th row is the header.
    rownums = {}
    for manuf in ['ge', 'philips', 'siemens']:
        for seq in '01':
            for subj in '0123':
                rownums[manuf, seq, subj] = rownum
                rownum += 1

    # Precalculate the info that will be in common between the trials.
    indexed_rows = []
    for seg in segs:
        trial, manuf, seq, subj, junk = seg.split('/')
        indexed_rows.append((rownums[manuf, seq, subj], manuf, seq, trial, subj))

    outfns = []
    for trial in trials:
        with open('../all_%s_vs_dtb_eddy_T1wTIV_edited.csv' % trial) as allf:
            allrows = [line.strip().split(',') for line in allf]
        os.mkdir(trial)
        outfns.append(trial + "/tiv_vs_dtb_eddy_T1wTIV_edited_segmentation.csv")
        with open(outfns[-1], 'w') as outf:
            outf.write(
                "Manufacturer,Sequence,trial,Subject,Jaccard Index,Dice Coefficient,Relative Error\n")
            batch = []
            for ir in indexed_rows:
                manuf, seq, trial_num, subj = ir[1:]
                if [manuf, seq, subj] != allrows[ir[0]][:3]:
                    raise ValueError("%s: mismatch for %s,%s" % (trial, ir, allrows[ir[0]][:3]))    
                outf.write("%s,%s,%s,%s,%s\n" % (manuf, seq, trial_num, subj, ','.join(allrows[ir[0]][3:])))
                batch.append(map(float, allrows[ir[0]][3:]))
                if len(batch) == 12:
                    means = np.mean(batch, axis=0)
                    outf.write("Average,Both,%s,trial_set,%f,%f,%f\n" % (trial_num, means[0],
                                                                         means[1], means[2]))
                    batch = []
    return outfns


def compare_segmentations(trial='dtb_eddy_rfcseg.nii', gold='dtb_eddy_T1wTIV_edited_segmentation.nii',
                          branch='test'):
    results = {}
    tseg = nib.load(trial).get_data()
    gseg = nib.load(gold).get_data()
    brains = [_cond_to_mask(tseg, 1), _cond_to_mask(gseg, 1)]
    csfs   = [_cond_to_mask(tseg, 2), _cond_to_mask(gseg, 2)]
    others = [_cond_to_mask(tseg, 3), _cond_to_mask(gseg, 3)]
    results['brain'] = (jaccard_index(brains[0], brains[1]), dice(brains[0], brains[1]),
                        relative_error(brains[0], brains[1]))
    results['csf'] = (jaccard_index(csfs[0], csfs[1]), dice(csfs[0], csfs[1]), relative_error(csfs[0],
                                                                                              csfs[1]))
    results['other'] = (jaccard_index(others[0], others[1]), dice(others[0], others[1]),
                        relative_error(others[0], others[1]))

    for i in [0, 1]:
        brains[i] += others[i]
    results['brain + other'] = (jaccard_index(brains[0], brains[1]), dice(brains[0], brains[1]),
                                relative_error(brains[0], brains[1]))

    for i in [0, 1]:
        brains[i] += csfs[i]
    results['tiv'] = (jaccard_index(brains[0], brains[1]), dice(brains[0], brains[1]),
                      relative_error(brains[0], brains[1]))

    return results


def make_segmentation_comparisons(trial='dtb_eddy_rfcseg',
                                  gold_standard='dtb_eddy_T1wTIV_edited_segmentation',
                                  branch='test', label=''):
    if not label:
        label = trial
    outfns = {}
    outfhs = {}
    try:
        for segtype in ['brain', 'csf', 'other', 'brain + other', 'tiv']:
            outfns[segtype] = "%s_vs_%s_%s_in_%s.csv" % (label.replace('/', '_'), gold_standard,
                                                         segtype.replace(' + ', '_p_'), branch)
            outfhs[segtype] = open(outfns[segtype], 'w')
            outfhs[segtype].write("Manufacturer,Subject,Jaccard Index,Dice Coefficient,Relative Error\n")
        for manuf in ['ge', 'philips', 'siemens']:
            for subj in ['0_0', '0_1', '1_0', '1_1']:
                niidir = os.path.join(branch, manuf, subj)
                afn = os.path.join(niidir, trial + '.nii')
                bfn = os.path.join(niidir, gold_standard + '.nii')
                results = compare_segmentations(afn, bfn)
                for segtype, result in results.iteritems():
                    outfhs[segtype].write("%s,%s,%f,%f,%f\n" % (manuf, subj, result[0], result[1],
                                                                result[2]))
    finally:
        for outf in outfhs.itervalues():
            outf.close()
    return outfns


def make_cv_comparisons(label, testim='seg.nii', gold_standard_im='dtb_eddy_T1wTIV_edited_segmentation.nii',
                        gold_standard_root='gold_standards', trial_pat=r'[0-9]+'):
    """
    Run in crossvalidation/.
    """
    outfns = {}
    outfhs = {}
    trial_pat = re.compile(trial_pat)
    try:
        for segtype in ['brain', 'csf', 'other', 'brain + other', 'tiv']:
            outfns[segtype] = "%s/%s_vs_%s.csv" % (label.replace(' ', '_'),
                                                   segtype.replace(' + ', '_p_'),
                                                   gold_standard_im[:-4])
            outfhs[segtype] = open(outfns[segtype], 'w')
            outfhs[segtype].write(
                "Manufacturer,Sequence,trial,Subject,Jaccard Index,Dice Coefficient,Relative Error\n")
        items = os.listdir(label)
        for trial in items:
            if trial_pat.match(trial):
                trial_root = os.path.join(label, trial)
                print "looking at", trial_root
                trial_stats = {}
                if os.path.isdir(trial_root):
                    for manuf in ['ge', 'philips', 'siemens']:
                        for seq in '01':
                            mseq = os.path.join(manuf, seq)
                            subjs = os.listdir(os.path.join(trial_root, mseq))
                            for subj in subjs:
                                testfn = os.path.join(trial_root, mseq, subj, testim)
                                goldfn = os.path.join(gold_standard_root, mseq, subj, gold_standard_im)
                                results = compare_segmentations(testfn, goldfn)
                                for segtype, result in results.iteritems():
                                    outfhs[segtype].write("%s,%s,%s,%s,%f,%f,%f\n" % (manuf, seq, trial,
                                                                                      subj, result[0],
                                                                                      result[1], result[2]))
                                    if segtype not in trial_stats:
                                        trial_stats[segtype] = []
                                    trial_stats[segtype].append(result)
                # Add the mean across (manuf, seq, subj) for trial.
                for segtype, rl in trial_stats.iteritems():
                    means = np.array(rl).mean(axis=0)
                    outfhs[segtype].write("Average,Both,%s,trial_set,%f,%f,%f\n" % (trial, means[0],
                                                                                    means[1], means[2]))
    finally:
        for outf in outfhs.itervalues():
            outf.close()
    return outfns


def symlink_remote(src, dst):
    """
    os.symlink (and ln -s) produce broken symlinks if dst is not in the current directory,
    and or if src relative to the current directory, but not dst's directory.

    This tries to be more robust and DWIM.
    """
    startdir = os.path.abspath('.')
    dstdir, dstbase = os.path.split(dst)
    if not os.path.isabs(src):
        nsteps = len(dstdir.split('/'))
        for d in xrange(nsteps):
            src = '../' + src
    if not os.path.isdir(dstdir):
        os.makedirs(dstdir)
    try:
        os.chdir(dstdir)
        os.symlink(src, dstbase)
    finally:
        os.chdir(startdir)


def cleanup_cv(label, scorefn='tiv_vs_dtb_eddy_T1wTIV_edited_segmentation.csv', trial_pat=r'[0-9]+'):
    """
    Find the best and worst segmentation for label, and rm the rest.

    Run in crossvalidation/.
    """
    extremes = {'best': None,
                'worst': None}
    bestscore = 2.0
    worstscore = -1
    # best_mean_score = 2.0
    # best_trial = None
    csvfn = os.path.join(label, scorefn)
    with open(csvfn) as f:
        for line in f:
            parts = line.split(',')
            if parts[0].lower() in ['ge', 'philips', 'siemens']:
                score = float(parts[-1])
                if score < bestscore:
                    bestscore = score
                    extremes['best'] = parts[:4]
                if score > worstscore:
                    worstscore = score
                    extremes['worst'] = parts[:4]
            elif parts[0] == 'Average':
                score = float(parts[-1])
                if score < bestscore:
                    bestscore = score
                    # best_trial = parts[2]
            else:
                pass  # header line

    # Keep the best and worst segmentation for label
    for k, parts in extremes.iteritems():
        outdir = os.path.join(label, k)
        os.mkdir(outdir)
        manuf, seq, trial, subj = parts
        infn = os.path.join(label, trial, manuf, seq, subj, 'seg.nii')
        outfn = os.path.join(outdir, '_'.join(('seg', 'trial', trial, manuf,
                                               'seq', seq, 'subj', subj)) + '.nii')
        os.rename(infn, outfn)
        gold = os.path.join('gold_standards', manuf, seq, subj)
        symlink_remote(gold, os.path.join(outdir, 'gold'))

    # Keep the best classifier!
    # (With "best" being defined here as lowest mean_over_manuf_and_seq(relative_error).)
    os.rename(os.path.join(label, trial, 'RFC_classifier.pickle'),
              os.path.join(label, 'RFC_classifier.pickle'))
    with open(os.path.join(label, 'RFC_classifier.mean_relerr_over_manuf_and_seq'), 'w') as f:
        f.write("%f\n" % bestscore)

    # rm the rest.
    items = os.listdir(label)
    trial_pat = re.compile(trial_pat)
    for trial in items:
        if trial_pat.match(trial):
            shutil.rmtree(os.path.join(label, trial))
    return bestscore, worstscore, extremes


def mask_from_possible_filename(m, binarize=True, thresh=0):
    if isinstance(m, str):
        m = nib.load(m).get_data()
    if binarize:
        m = m > thresh
    return m


def make_3way_comparison_image(mask1fn, mask2fn, goldfn):
    mask1 = mask_from_possible_filename(mask1fn)
    aff = nib.load(mask1fn).get_affine()
    mask2 = mask_from_possible_filename(mask2fn)
    gold = mask_from_possible_filename(goldfn)
    m1label = os.path.splitext(os.path.split(mask1fn)[-1])[0]
    m2label = os.path.splitext(os.path.split(mask2fn)[-1])[0]
    glabel = os.path.splitext(os.path.split(goldfn)[-1])[0]
    m1m2olab = "%s and %s only" % (m1label, m2label)
    m1golab = "%s and %s only" % (m1label, glabel)
    m2golab = "%s and %s only" % (m2label, glabel)
    res = {m1label + " only": mask1.copy(),
           m2label + " only": mask2.copy(),
           m1m2olab: mask1 * mask2,
           m1golab: mask1 * gold,
           m2golab: mask2 * gold,
           glabel + " only": gold.copy()}
    res[m1label + " only"][mask2 > 0] = False
    res[m1label + " only"][gold > 0] = False

    res[m2label + " only"][mask1 > 0] = False
    res[m2label + " only"][gold > 0] = False

    res[m1m2olab][gold > 0] = False
    res[m1golab][mask2 > 0] = False
    res[m2golab][mask1 > 0] = False

    res[glabel + " only"][mask1 > 0] = False
    res[glabel + " only"][mask2 > 0] = False

    for k, v in res.iteritems():
        save_mask(v, aff, k.replace(' ', '_') + '.nii')

    sums = dict([(k, int(v.sum())) for k, v in res.iteritems()])
    sums[glabel] = int(gold.sum())

    relerr = {m1label: (mask1 != gold).sum() / float(sums[glabel]),
              m2label: (mask2 != gold).sum() / float(sums[glabel])}
    return {"images": res, "sums": sums, "relative errors": relerr}

# Cross-validation samples, because for comparing one method to another, there
# doesn't have to be just one training and one test set - multiple training and
# test sets can be made by picking and choosing from each, as long as
# * the results don't overlap, and
# * the results have the same size (so we don't get size effects).
#
#            label   use           tt        subj   tt        subj
cv_samples = {'a': {'training': (('training', 0), ('training', 1)),
                    'test':     (('test',     0), ('test',     1))},
              'b': {'training': (('training', 0), ('test',     0)),
                    'test':     (('training', 1), ('test',     1))},
              'c': {'training': (('training', 0), ('test',     1)),
                    'test':     (('training', 1), ('test',     0))},
              'd': {'training': (('training', 1), ('test',     0)),
                    'test':     (('training', 0), ('test',     1))},
              'e': {'training': (('training', 1), ('test',     1)),
                    'test':     (('training', 0), ('test',     0))},
              'f': {'training': (('test',     0), ('test',     1)),
                    'test':     (('training', 0), ('training', 1))}}


def setup_gold_standards_for_cross_validation(base='gold_standards'):
    """Run in crossvalidation"""
    root = os.path.abspath('..')
    os.makedirs(base)
    for m in ['ge', 'philips', 'siemens']:
        basem = os.path.join(base, m)
        os.makedirs(basem)
        for seqi, seq in enumerate('01'):
            basemseq = os.path.join(basem, seq)
            os.makedirs(basemseq)
            i = 0
            for tti, tt in enumerate(['training', 'test']):
                for subj in '01':
                    os.symlink(os.path.join(root, tt, m, seq + '_' + subj),
                               os.path.join(basemseq, "%s" % i))
                    i += 1


def setup_trials(label, train_manufs, train_seqs, ntrials=10, base='gold_standards',
                 lsvecs_fn='dtb_eddy_lsvecs.nii', manufs=['ge', 'philips', 'siemens'],
                 seqs='01', maxperclass=100000, useT1=False):
    """
    Run in crossvalidation/.

    For classifiers that don't need training, i.e. dwi2mask, just set
    train_manufs and train_seqs to [].
    """
    label = label.replace(' ', '_')
    os.makedirs(label)
    test_lists = []
    qtrainfn = 'qtrain_%s.sh' % label
    qsegfn = os.path.join(label, 'qmake_segmentations.sh')
    with open(qtrainfn, 'w') as qtf:
        for trial in xrange(ntrials):
            train_list = []
            test_lists.append([])
            ltrial = os.path.join(label, str(trial))
            os.makedirs(ltrial)
            for manuf in manufs:
                for seq in seqs:
                    deal = np.random.choice(4, 4, False)
                    if manuf in train_manufs and seq in train_seqs:
                        train_list += [os.path.join(base, manuf, str(seq), str(subj))
                                       for subj in deal[:2]]
                    test_lists[-1] += [os.path.join(manuf, str(seq), str(subj))
                                       for subj in deal[2:]]
            if train_list:
                # print "Training for trial %d" % trial
                # res, pfn, logfn = train.train_both_stages_from_multiple(train_list,
                #                                                         os.path.join(ltrial, 'classifier'),
                #                                                         srclist_is_srcdirs=True,
                #                                                         lsvecs_fn=lsvecs_fn,
                #                                                         maxperclass=maxperclass,
                #                                                         useT1=useT1)
                tlfn = os.path.join(ltrial, "train_list.pickle")
                classifier = os.path.join(ltrial, 'classifier')
                brine.brine(train_list, tlfn)
                cmdline = "../train_from_multiple "
                if useT1:
                    cmdline += "-t "
                cmdline += "%s %s %s %d\n" % (tlfn, classifier, lsvecs_fn, maxperclass)
                qtf.write(cmdline)
    with open(qsegfn, 'w') as f:
        for trial, test_list in enumerate(test_lists):
            for test_dir in test_list:
                f.write("../../make_segmentation -o {trial}/{test_dir}/seg.nii ../{base}/{test_dir}/{lsvecs_fn} {trial}/RFC_classifier.pickle\n".format(**locals()))
    brine.brine(test_lists, os.path.join(label, 'test_lists.pickle'))
    print "First\nqsw_a %s\nthen\nqsw_a %s" % (qtrainfn, qsegfn)


def qmake_segmentation_to_test_lists():
    """
    Needed because the 1st run of setup_trials didn't save test_lists except
    implicitly in qmake_segmentation.sh.

    Run in the label directory.
    """
    tld = {}
    with open("qmake_segmentations.sh") as f:
        for line in f:
            if line and line[0] not in "#\n":
                segpath = line[27:].split()[0]
                segparts = segpath.split('/')
                trial = int(segparts[0])
                if trial not in tld:
                    tld[trial] = []
                tld[trial].append('/'.join(segparts[1:4]))

    # Check that all of tld's values have the same length (12):
    #for k, v in tld.iteritems():
    #    print k, len(v)

    test_lists = [tld[k] for k in xrange(len(tld))]
    brine.brine(test_lists, 'test_lists.pickle')
    return test_lists
