import os
import subprocess
import tempfile

class PTBTokenizer_(object):
    corenlpjar = 'stanford-corenlp-3.4.1.jar'
    punctuations_ = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    @classmethod
    def tokenize(cls, corpus):
        cmd = ['java', '-cp', cls.corenlpjar, \
                'edu.stanford.nlp.process.PTBTokenizer_', \
                '-preserveLines', '-lowerCase']

        if isinstance(corpus, list) or isinstance(corpus, tuple):
            if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
                corpus = {i:c for i, c in enumerate(corpus)}
            else:
                corpus = {i: [c, ] for i, c in enumerate(corpus)}

        tokenizedcorpus = {}
        image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
        sentences = '\n'.join([c.replace('\n', ' ') for k, v in corpus.items() for c in v])

        path_to_jar_dirname=os.path.dirname(os.path.abspath(__file__))
        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=path_to_jar_dirname)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        cmd.append(os.path.basename(tmp_file.name))
        p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        token_lines = p_tokenizer.communicate(input=sentences.rstrip())[0]
        token_lines = token_lines.decode()
        lines = token_lines.split('\n')
        # remove temp file
        os.remove(tmp_file.name)

        for k, line in zip(image_id, lines):
            if not k in tokenizedcorpus:
                tokenizedcorpus[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                          if w not in cls.punctuations_])
            tokenizedcorpus[k].append(tokenized_caption)

        return tokenizedcorpus