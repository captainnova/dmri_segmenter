import dmri_segmenter.dmri_brain_extractor as dbe
import dmri_segmenter.make_comparisons as mc
import nibabel as nib
#import numpy as np
import os


def test_get_version_info():
    vinfo = dbe.get_version_info()
    assert "\ncommit " in vinfo
    assert "\nDate:" in vinfo
    assert "version" in vinfo


def test_feature_vector_classify(fakedata, tmpdir):
    outdir = str(tmpdir)
    t1tivfn = os.path.join(outdir, 't1tiv.nii')
    dbe.save_mask(fakedata.phantom, fakedata.aff, t1tivfn)
    assert os.path.isfile(t1tivfn)
    brain, csf, holes, posterity = dbe.feature_vector_classify(fakedata.data,
                                                               fakedata.aff,
                                                               fakedata.bvals,
                                                               smoothrad=4.0,
                                                               Dt=0.0014,
                                                               Dcsf=0.0021,
                                                               t1wtiv=t1tivfn,
                                                               t1fwhm=[0.5, 1.0, 0.75])
    assert brain.shape == fakedata.phantom.shape

    # Since fakedata is an approximation of a 20C water phantom, don't expect
    # dbe to do too well at classifying the different tissue types.
    tiv = brain + csf + holes

    ji = mc.jaccard_index(fakedata.phantom, tiv)
    assert ji > 0.9

    assert "Classifier loaded from " in posterity
    assert "\nThe classifier is a" in posterity
    assert "rained from" in posterity


def test_get_dmri_brain_and_tiv(fakedata, tmpdir):
    outdir = str(tmpdir)
    ecfn = os.path.join(outdir, 'ec.nii')
    nib.save(nib.nifti1.Nifti1Image(fakedata.data, fakedata.aff), ecfn)
    ecnii = nib.load(ecfn)
    brfn = os.path.join(outdir, 'br.nii')
    tivfn = os.path.join(outdir, 'tiv.nii')
    _, tiv = dbe.get_dmri_brain_and_tiv(fakedata.data, ecnii, brfn, tivfn,
                                        fakedata.bvals, isFLAIR=False)
    assert os.path.isfile(brfn)
    assert os.path.isfile(tivfn)
    assert (tiv[::10, 45, 4] == [0, 0, 1, 1, 1, 1, 1, 0]).all()
