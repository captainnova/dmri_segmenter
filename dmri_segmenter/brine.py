"""
pickle's interface is annoying in that it doesn't include
"if isinstance(f, str), open(f)" to DWIM.  brine does that, which is very
convenient for using it with readline in ipython (This way, you see the
file and the object on the same line).
"""
try:
    import cPickle as pickle    # faster
except:
    import pickle               # subclassable

def brine(obj, filename, protocol=-1):
    "cPickle.dump(obj) into filename."
    with open(filename, 'wb') as f:   # b needed (outside UNIX) for protocol=-1 (highest avail)
        pickle.dump(obj, f, protocol)

def debrine(filename):
    "cPickle.load(open(filename))"
    obj = None
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj