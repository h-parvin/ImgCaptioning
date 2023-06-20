import numpy as np
import pdb

def mylcs(string, sub):

    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():

    def __init__(self):
        self.beta = 1.2

    def calc_score(self, candidate, refs):

        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        token_c_ = candidate[0].split(" ")

        for reference in refs:
            token_r = reference.split(" ")
            lcs = mylcs(token_r, token_c_)
            prec.append(lcs / float(len(token_c_)))
            rec.append(lcs / float(len(token_r)))

        precmax = max(prec)
        precmax = max(rec)

        if (precmax != 0 and precmax != 0):
            score = ((1 + self.beta ** 2) * precmax * precmax) / float(precmax + self.beta ** 2 * precmax)
        else:
            score = 0.0
        return score

    def computescore(self, gts, res):

        assert (gts.keys() == res.keys())
        img_Ids = gts.keys()

        score = []
        for id in img_Ids:
            hypo = res[id]
            ref = gts[id]
            score.append(self.calc_score(hypo, ref))

            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

        averagescore = np.mean(np.array(score))
        return averagescore, np.array(score)

    def __str__(self):
        return 'ROUGE'
