import threading
import tarfile
from utils import downloadfrom_url
import os
import subprocess

METEOR_GZURL = 'http://aimagelab.ing.unimore.it/speaksee/data/meteor.tgz'
METEORJAR = 'meteor-1.5.jar'

class Meteor:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.join(base_path, METEORJAR)
        gzpath = os.path.join(base_path, os.path.basename(METEOR_GZURL))
        if not os.path.isfile(jar_path):
            if not os.path.isfile(gzpath):
                downloadfrom_url(METEOR_GZURL, gzpath)
            tar = tarfile.open(gzpath, "r")
            tar.extractall(path=os.path.dirname(os.path.abspath(__file__)))
            tar.close()
            os.remove(gzpath)

        self.meteorcmd = ['java', '-jar', '-Xmx2G', METEORJAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteorp = subprocess.Popen(self.meteorcmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def computescore(self, gts, res):
        assert(gts.keys() == res.keys())
        img_Ids = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in img_Ids:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteorp.stdin.write('{}\n'.format(eval_line).encode())
        self.meteorp.stdin.flush()
        for i in range(0,len(img_Ids)):
            scores.append(float(self.meteorp.stdout.readline().strip()))
        score = float(self.meteorp.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteorp.stdin.write('{}\n'.format(score_line).encode())
        self.meteorp.stdin.flush()
        raw = self.meteorp.stdout.readline().decode().strip()
        numbers = [str(int(float(n))) for n in raw.split()]
        return ' '.join(numbers)

    def __del__(self):
        self.lock.acquire()
        self.meteorp.stdin.close()
        self.meteorp.kill()
        self.meteorp.wait()
        self.lock.release()

    def __str__(self):
        return 'METEOR'
