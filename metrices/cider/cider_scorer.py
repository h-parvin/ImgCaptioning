import copy
from collections import default_dict
import numpy as np
import math

def pre_cook(s, n=4):
    words = s.split()
    counts = default_dict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cookrefs(refs, n=4): 

    return [pre_cook(ref, n) for ref in refs]

def cooktest(test, n=4):

    return pre_cook(test, n)

class CiderScorer(object):

    def __init__(self, refs, test=None, n=4, sigma=6.0, _doc_frequency=None, _reflen_=None):
        self.n = n
        self.sigma = sigma
        self.c_refs = []
        self.c_test = []
        self._doc_frequency = default_dict(float)
        self._reflen_ = None

        for k in refs.keys():
            self.c_refs.append(cookrefs(refs[k]))
            if test is not None:
                self.c_test.append(cooktest(test[k][0]))  ## N.B.: -1
            else:
                self.c_test.append(None)  

        if _doc_frequency is None and _reflen_ is None:
            self.compute_doc_freq()
            self._reflen_ = np.log(float(len(self.c_refs)))
        else:
            self._doc_frequency = _doc_frequency
            self._reflen_ = _reflen_

    def compute_doc_freq(self):

        for refs in self.c_refs:
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self._doc_frequency[ngram] += 1

    def computeCider(self):
        def counts2vec(cnts):

            vec = [default_dict(float) for _ in range(self.n)]
            length_ = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self._doc_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self._reflen_ - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length_ += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length_

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):

            delta = float(length_hyp - length_ref)
            val_ = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    val_[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val_[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val_[n]))
                val_[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val_

        scores = []
        for test, refs in zip(self.c_test, self.c_refs):
            vec, norm, length_ = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length_, length_ref)
            scoreavg = np.mean(score)
            scoreavg /= len(refs)
            scoreavg *= 10.0
            scores.append(scoreavg)
        return scores

    def computescore(self):
        score = self.computeCider()
        return np.mean(np.array(score)), np.array(score)