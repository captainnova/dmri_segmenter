from dmri_segmenter.FLAIRity import FLAIRity


def test_FLAIRity(fakedata):
    guess = FLAIRity(fakedata.data, fakedata.aff, fakedata.bvals)
    assert guess.flairity == False   # noqa
