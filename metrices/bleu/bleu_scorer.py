import sys, math, re
from collections import default_dict
import copy


def pre_cook(s, n=4, out=False):
    words = s.split()
    counts = default_dict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return (len(words), counts)


def cookrefs(refs, eff=None, n=4):  
    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = pre_cook(ref, n)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)

    return (reflen, maxcounts)


def cooktest(test, ref_tuple, eff=None, n=4):
    testlen, counts = pre_cook(test, n, True)
    reflen, refmaxcounts = ref_tuple

    result = {}

    if eff == "closest":
        result["reflen"] = min((abs(l - testlen), l) for l in reflen)[1]
    else:  
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]

    result['correct'] = [0] * n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


class BleuScorer(object):
    __slots__ = "n", "c_refs", "c_test", "_scores", "_ratio", "_testlen", "_reflen_", "special_reflen"

    def copy(self):
        new = BleuScorer(n=self.n)
        new.c_test = copy.copy(self.c_test)
        new.c_refs = copy.copy(self.c_refs)
        new._scores = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):

        self.n = n
        self.c_refs = []
        self.c_test = []
        self.cook__append(test, refs)
        self.special_reflen = special_reflen

    def cook__append(self, test, refs):

        if refs is not None:
            self.c_refs.append(cookrefs(refs))
            if test is not None:
                cooked_test = cooktest(test, self.c_refs[-1])
                self.c_test.append(cooked_test)  ## N.B.: -1
            else:
                self.c_test.append(None)  # lens of c_refs and c_test have to match

        self._scores = None  ## need to recompute

    def ratio(self, option=None):
        self.computescore(option=option)
        return self._ratio

    def score_ratio(self, option=None):

        return self.fscore(option=option), self.ratio(option=option)

    def score_ratio_str(self, option=None):
        return "%.4f (%.2f)" % self.score_ratio(option)

    def reflen(self, option=None):
        self.computescore(option=option)
        return self._reflen_

    def testlen(self, option=None):
        self.computescore(option=option)
        return self._testlen

    def retest(self, new_test):
        if type(new_test) is str:
            new_test = [new_test]
        assert len(new_test) == len(self.c_refs), new_test
        self.c_test = []
        for t, rs in zip(new_test, self.c_refs):
            self.c_test.append(cooktest(t, rs))
        self._scores = None

        return self

    def rescore(self, new_test):

        return self.retest(new_test).computescore()

    def size(self):
        assert len(self.c_refs) == len(self.c_test), "refs/test mismatch! %d<>%d" % (len(self.c_refs), len(self.c_test))
        return len(self.c_refs)

    def __iadd__(self, other):

        if type(other) is tuple:
            self.cook__append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible bleus_."
            self.c_test.extend(other.c_test)
            self.c_refs.extend(other.c_refs)
            self._scores = None  ## need to recompute

        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def single_reflen(self, option="average"):
        return self._singlereflen(self.c_refs[0][0], option)

    def _singlereflen(self, reflens, option=None, testlen=None):

        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option

        return reflen

    def recomputescore(self, option=None, verbose=0):
        self._scores = None
        return self.computescore(option, verbose)

    def computescore(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15  
        bleu_list = [[] for _ in range(n)]

        if self._scores is not None:
            return self._scores

        if option is None:
            option = "average" if len(self.c_refs) == 1 else "closest"

        self._testlen = 0
        self._reflen_ = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        # for each sentence
        for comps in self.c_test:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:  ## need computation
                reflen = self._singlereflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen_ += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen_
        totalcomps['testlen'] = self._testlen

        bleus_ = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus_.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen_ + small)  ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleus_[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._scores = bleus_
        return self._scores, bleu_list
