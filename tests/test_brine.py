import dmri_segmenter.brine as brine
import numpy as np
import os


def test_brine(tmpdir):
    print "brine.__file__ =", brine.__file__
    arr = np.arange(5)
    pfn = os.path.join(str(tmpdir), 'arr.pickle')
    brine.brine(arr, pfn)
    reconstituted = brine.debrine(pfn)
    assert (arr == reconstituted).all()
