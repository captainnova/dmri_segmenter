from __future__ import absolute_import
from dipy.utils.optpkg import optional_package
import os

onnxruntime, have_onnxruntime, setup_module = optional_package('onnxruntime')

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
                pass  # TODO
            else:
                raise ValueError("onnxruntime must be installed to use a multifile classifier")
        else:
            clf = brine.debrine(loadfrom)
    return clf, posterity


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
