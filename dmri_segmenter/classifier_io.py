from __future__ import absolute_import
from dipy.utils.optpkg import optional_package
import numpy as np
import os

onnxruntime, have_onnxruntime, setup_module = optional_package('onnxruntime')
skl2onnx, have_skl2onnx, setup_module = optional_package('skl2onnx')

try:
    from . import brine
except Exception:                         # I'm not sure this ever gets used anymore.
    import brine


def _append_mydir_to_path(path):
    try:
        mydir = os.path.dirname(__file__)
    except Exception:
        mydir = os.path.dirname(brine.__file__)
    return path + [mydir]


def find_thing_in_path(thing, path):
    """Return the first occurrence of thing in path, or raise a ValueError.

    Parameters
    ----------
    thing : str
        The thing to look for.
    path : list
        The directories to look for it in.

    Returns
    -------
    os.path.join(directory, thing), IFF it exists.
    """
    found = False
    for d in path:
        cand = os.path.join(d, thing)
        if os.path.exists(cand):
            found = cand
            break
    if not found:
        raise ValueError("%s not found in %s" % (thing, path))
    return found


if have_onnxruntime:
    from onnxruntime import InferenceSession
else:
    InferenceSession = object


class OnnxAsSkl(InferenceSession):
    """Wraps onnxruntime.InferenceSession so it has classes_ and predict_proba()
    members like sklearn's RandomForestClassifier.
    """
    @property
    def classes_(self):
        "Labels for the classes"
        if hasattr(self, '_classes_'):
            return self._classes_
        else:
            raise ValueError(".classes_ has not been set yet")

    @classes_.setter
    def classes_(self, val):
        """
        Parameters
        ----------
        val : sequence
        """
        self._classes_ = np.asarray(val)

    @classes_.deleter
    def classes_(self):
        del self._classes_

    def predict_proba(self, X, probability_label='output_probability',
                      xtype=np.float32):
        input_name = self.get_inputs()[0].name

        # AFAICT self.run's output is a list with only 1 element: a list of dicts.
        # It's run_options option does not look very useful, and the documentation
        # is very sparse.
        lod = self.run([probability_label], {input_name: X.astype(xtype)})[0]

        # Assumes all the dicts have the same keys, but they have to, meaning onnx
        # is oddly inefficient.
        if lod:
            self.classes_ = sorted(lod[0].keys())
        else:
            self.classes_ = []
        return np.array([[d[k] for k in self.classes_] for d in lod])


def load_classifier(src, srcpath=['.', '~/.dipy/dmri_segmenter/classifiers']):
    """
    Load a classifier dict from src

    Parameters
    ----------
    src : dict or str
        If a dict, simply return src.
        Else if a directory, attempt to load the classifier stages from
          src/1.onnx, src/2.onnx, etc..
        Else unpickle the dict from src. (Convenient, but there may be
          compatibility problems if you are using a different version of
          sklearn than src was pickled with.)
        Files and directories will be searched for in srcpath.
    srcpath : list, optional
        List of directories to search for src in.
        os.path.dirname(__file__) or os.path.dirname(brine.__file__) will
        be appended.

    Returns
    -------
    clf : a dict
        The classifier dict, with at least 'smoothrad', 'log',
        's0 brain level', '1st stage', 'n_classes', '2nd stage',
        'src_properties', and 'n_features' as keys.
    posterity : str
        Message saying where it got clf from.
    """
    if isinstance(src, dict):
        clf = src
        posterity = "Using already instantiated classifier dict.\n"
    else:
        loadfrom = find_thing_in_path(src, _append_mydir_to_path(srcpath))
        posterity = "Classifier loaded from %s.\n" % os.path.abspath(loadfrom)
        if os.path.isdir(loadfrom):
            if have_onnxruntime:
                clf = brine.debrine(os.path.join(loadfrom, 'dict.pickle'))
                stages = [k for k in clf if isinstance(k, str) and k.endswith('stage')]
                for s in stages:
                    clf[s] = OnnxAsSkl(os.path.join(loadfrom, clf[s]))
                    clf[s].classes_ = clf['classes_'][s]
            else:
                raise ValueError("onnxruntime must be installed to use a multifile classifier")
        else:
            clf = brine.debrine(loadfrom)
    return clf, posterity


def save_clf_dict_to_onnx(clfd, outdir, desc='', target_opset=12):
    """Save a classifier dict to a directory with the weights in ONNX format.

    Parameters
    ----------
    clfd : dict
        Top level classifier directory, including '1st stage', '2nd stage',
        etc. sklearn classifiers as keys/values.
    outdir : str
        The directory to save clfd as. Will be created or overwritten.
    desc : str, optional
        A descriptive comment that will be saved in the output.
    target_opset : int, optional
        The minimum version of onnx's operations set to aim for.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    stages = sorted([k for k in clfd if isinstance(k, str) and k.endswith('stage')])
    if not stages:
        raise ValueError("The classifier dict must have some keys ending in 'stage'")
    myd = {k: v for k, v in clfd.items() if k not in stages}
    myd['classes_'] = {}
    myd['n_features_'] = {}
    if desc:
        myd['Description'] = desc
    for i, s in enumerate(stages):
        myd[s] = "%d.onnx" % i
        clf = clfd[s]
        nfeatures = clf.n_features_
        myd['classes_'][s] = clf.classes_
        myd['n_features_'][s] = nfeatures
        input_type = [('nx%d_input' % nfeatures,
                       skl2onnx.common.data_types.FloatTensorType([None, nfeatures]))]
        onx = skl2onnx.convert_sklearn(clfd[s], initial_types=input_type,
                                       target_opset=target_opset)
        with open(os.path.join(outdir, myd[s]), 'wb') as f:
            f.write(onx.SerializeToString())
    brine.brine(myd, os.path.join(outdir, 'dict.pickle'))


def fetch_classifier(ctype=None, training_set='RFC_ADNI6',
                     srcurl='https://github.com/dipy',
                     destdir='~/.dipy/dmri_segmenter/classifiers', force=False):
    """Download (if necessary) a trained classifier.

    Parameters
    ----------
    ctype : , optional

    training_set : , optional


    Returns
    -------
    None (but will raise an Exception if it fails)
    """
    pass
