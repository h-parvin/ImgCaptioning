import six
from six.moves.urllib.request import urlretrieve
import torch
from tqdm import tqdm
import tarfile
from .utils import reporthook
logger = logging.getLogger(__name__)
from __future__ import unicode_literals
import array
from collections import default_dict
from functools import partial
import io
import logging
import os
import zipfile


class Vectors(object):

    def __init__(self, name, cache=None,
                 url=None, unk_init=None):

        cache = '.vector_cache' if cache is None else cache
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.cache(name, cache, url=url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(self.dim))  # self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None):
        if os.path.isfile(name):
            path = name
            path_pt = os.path.join(cache, os.path.basename(name)) + '.pt'
        else:
            path = os.path.join(cache, name)
            path_pt = path + '.pt'

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    with tarfile.open(dest, 'r:gz') as tar:
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tar, path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            itos, vectors, dim = [], array.array(str('d')), None

            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]

            except:
                logger.warning("Could not read {} as UTF8 file, "
                               "reading file as bytes and skipping "
                               "words with malformed UTF8.".format(path))
                with open(path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            logger.info("Loading vectors from {}".format(path))
            for line in tqdm(lines, total=len(lines)):

                entries = line.rstrip().split(b" " if binary_lines else " ")

                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue
                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)


class GloVe(Vectors):
    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)


class FastText(Vectors):

    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.vec'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


class CharN_Gram(Vectors):

    name = 'CharN_Gram.txt'
    url = ('http://www.logos.t.u-tokyo.ac.jp/trained_embeddings.tar.gz')

    def __init__(self, **kwargs):
        super(CharN_Gram, self).__init__(self.name, url=self.url, **kwargs)

    def __getitem__(self, token):
        vector = torch.Tensor(1, self.dim).zero_()
        if token == "<unk>":
            return self.unk_init(vector)
        # These literals need to be coerced to unicode for Python 2 compatibility
        # when we try to join them with read ngrams from the files.
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:(i + n)] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
                    num_vectors += 1
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector


def _default_unk_index():
    return 0


pretrained_aliases = {
    "CharN_Gram.100d": partial(CharN_Gram),
    "fasttext.en.300d": partial(FastText, language="en"),
    "fasttext.simple.300d": partial(FastText, language="simple"),
    "glove.42B.300d": partial(GloVe, name="42B", dim="300"),
    "glove.840B.300d": partial(GloVe, name="840B", dim="300"),
    "glove.twitter.27B.25d": partial(GloVe, name="twitter.27B", dim="25"),
    "glove.twitter.27B.50d": partial(GloVe, name="twitter.27B", dim="50"),
    "glove.twitter.27B.100d": partial(GloVe, name="twitter.27B", dim="100"),
    "glove.twitter.27B.200d": partial(GloVe, name="twitter.27B", dim="200"),
    "glove.6B.50d": partial(GloVe, name="6B", dim="50"),
    "glove.6B.100d": partial(GloVe, name="6B", dim="100"),
    "glove.6B.200d": partial(GloVe, name="6B", dim="200"),
    "glove.6B.300d": partial(GloVe, name="6B", dim="300")
}

class Vocab(object):

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None):
 
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = default_dict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_vectors(self, vectors, **kwargs):

        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):

                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):

        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])

