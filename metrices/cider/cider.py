from .cider_scorers import CiderScorer

class Cider:

    def __init__(self, gts=None, n=4, sigma=6.0):
        self._n = n
        self._sigma_a = sigma
        self._doc_frequency = None
        self._reflen_ = None
        if gts is not None:
            tmp_cider = CiderScorer(gts, n=self._n, sigma=self._sigma_a)
            self._doc_frequency = tmp_cider._doc_frequency
            self._reflen_ = tmp_cider._reflen_

    def computescore(self, gts, res):
        assert(gts.keys() == res.keys())
        cider_scorers = CiderScorer(gts, test=res, n=self._n, sigma=self._sigma_a, _doc_frequency=self._doc_frequency,
                                   _reflen_=self._reflen_)
        return cider_scorers.computescore()

    def __str__(self):
        return 'CIDEr'
