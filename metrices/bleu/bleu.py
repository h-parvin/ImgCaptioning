from .bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        self._n = n
        self._hypo_of_image = {}
        self.ref_of_image = {}

    def computescore(self, gts, res):

        assert(gts.keys() == res.keys())
        img_Ids = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in img_Ids:
            hypo = res[id]
            ref = gts[id]

            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.computescore(option='closest', verbose=0)

        return score, scores

    def __str__(self):
        return 'BLEU'
