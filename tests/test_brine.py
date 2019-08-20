import brine
import numpy as np
import os


def test_brine(tmpdir):
    arr = np.arange(5)
    pfn = os.path.join(str(tmpdir), 'arr.pickle')
    brine.brine(arr, pfn)
    reconstituted = brine.debrine(pfn)
    assert (arr == reconstituted).all()
