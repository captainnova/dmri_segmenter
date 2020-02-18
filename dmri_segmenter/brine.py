"""
pickle's interface is annoying in that it doesn't include
"if isinstance(f, str), open(f)" to DWIM.  brine does that, which is very
convenient for using it with readline in ipython (This way, you see the
file and the object on the same line).
"""
from future import standard_library
standard_library.install_aliases()
import pickle                              # noqa E402


def brine(obj, filename, protocol=-1):
    "pickle.dump(obj) into filename."
    # b needed (outside UNIX) for protocol=-1 (highest avail)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def debrine(filename, encoding='latin1'):
    """pickle.load(open(filename))

    Use encoding='latin1' to load datetime, numpy, or scikit objects pickled in
    Python 2 in Python 3. (Latin-1 works for any input as it maps the byte
    values 0-255 to the first 256 Unicode codepoints directly:
    https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3/28218598#28218598
    )
    """
    obj = None
    with open(filename, 'rb') as f:
        try:
            obj = pickle.load(f, encoding=encoding)  # Python 3
        except Exception:
            obj = pickle.load(f)   # Python 2
    return obj
