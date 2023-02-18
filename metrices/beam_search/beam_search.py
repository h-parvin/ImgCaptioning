import torch
import utils

class BeamSearch(object):
    def __init__(self, model, maxlen: int, eosidx: int, beamsize: int):
        self.model = model
        self.maxlen = maxlen
        self.eosidx = eosidx
        self.beamsize = beamsize
        self.BS = None
        self.device = None
        self.seqmask = None
        self.seqlogprob = None
        self.outs = None
        self.logprobs_ = None
        self.selectedwords = None
        self.alllog_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(s.view(*([self.BS, cur_beam_size] + shape[1:])), 1,
                             beam.expand(*([self.BS, self.beamsize] + shape[1:])))
            s = s.view(*([-1, ] + shape[1:]))
            return s

        return fn

    def _expand_visual(self, visual: utils.TensorOrSequence, cur_beam_size: int, selected_beam: torch.Tensor):
        if isinstance(visual, torch.Tensor):
            visual_shape = visual.shape
            visualexp_shape = (self.BS, cur_beam_size) + visual_shape[1:]
            visualred_shape = (self.BS * self.beamsize,) + visual_shape[1:]
            selectedbeam_red_size = (self.BS, self.beamsize) + tuple(1 for _ in range(len(visualexp_shape) - 2))
            selectedbeam_exp_size = (self.BS, self.beamsize) + visualexp_shape[2:]
            visualexp = visual.view(visualexp_shape)
            selectedbeam_exp = selected_beam.view(selectedbeam_red_size).expand(selectedbeam_exp_size)
            visual = torch.gather(visualexp, 1, selectedbeam_exp).view(visualred_shape)
        else:
            new_visual = []
            for im in visual:
                visual_shape = im.shape
                visualexp_shape = (self.BS, cur_beam_size) + visual_shape[1:]
                visualred_shape = (self.BS * self.beamsize,) + visual_shape[1:]
                selectedbeam_red_size = (self.BS, self.beamsize) + tuple(1 for _ in range(len(visualexp_shape) - 2))
                selectedbeam_exp_size = (self.BS, self.beamsize) + visualexp_shape[2:]
                visualexp = im.view(visualexp_shape)
                selectedbeam_exp = selected_beam.view(selectedbeam_red_size).expand(selectedbeam_exp_size)
                new_im = torch.gather(visualexp, 1, selectedbeam_exp).view(visualred_shape)
                new_visual.append(new_im)
            visual = tuple(new_visual)
        return visual

    def apply(self, visual: utils.TensorOrSequence, out_size=1, returnprobs=False, **kwargs):
        self.BS = utils.getbatch_size(visual)
        self.device = utils.get_device(visual)
        self.seqmask = torch.ones((self.BS, self.beamsize, 1), device=self.device)
        self.seqlogprob = torch.zeros((self.BS, 1, 1), device=self.device)
        self.logprobs_ = []
        self.selectedwords = None
        if returnprobs:
            self.alllog_probs = []

        outs = []
        with self.model.statefulness(self.BS):
            for t in range(self.maxlen):
                visual, outs = self.iter(t, visual, outs, returnprobs, **kwargs)

        # Sort result
        seqlogprob, sort_idxs = torch.sort(self.seqlogprob, 1, descending=True)
        outs = torch.cat(outs, -1)
        outs = torch.gather(outs, 1, sort_idxs.expand(self.BS, self.beamsize, self.maxlen))
        logprobs_ = torch.cat(self.logprobs_, -1)
        logprobs_ = torch.gather(logprobs_, 1, sort_idxs.expand(self.BS, self.beamsize, self.maxlen))
        if returnprobs:
            alllog_probs = torch.cat(self.alllog_probs, 2)
            alllog_probs = torch.gather(alllog_probs, 1, sort_idxs.unsqueeze(-1).expand(self.BS, self.beamsize,
                                                                                          self.maxlen,
                                                                                          alllog_probs.shape[-1]))
        outs = outs.contiguous()[:, :out_size]
        logprobs_ = logprobs_.contiguous()[:, :out_size]
        if out_size == 1:
            outs = outs.squeeze(1)
            logprobs_ = logprobs_.squeeze(1)

        if returnprobs:
            return outs, logprobs_, alllog_probs
        else:
            return outs, logprobs_

    def select(self, t, _candidate_logprob, **kwargs):
        selectedlogprob, selectedidx = torch.sort(_candidate_logprob.view(self.BS, -1), -1, descending=True)
        selectedlogprob, selectedidx = selectedlogprob[:, :self.beamsize], selectedidx[:, :self.beamsize]
        return selectedidx, selectedlogprob

    def iter(self, t: int, visual: utils.TensorOrSequence, outs, returnprobs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beamsize

        wordlogprob = self.model.step(t, self.selectedwords, visual, None, mode='feedback', **kwargs)
        wordlogprob = wordlogprob.view(self.BS, cur_beam_size, -1)
        _candidate_logprob = self.seqlogprob + wordlogprob

        # Mask sequence if it reaches EOS
        if t > 0:
            mask = (self.selectedwords.view(self.BS, cur_beam_size) != self.eosidx).float().unsqueeze(-1)
            self.seqmask = self.seqmask * mask
            wordlogprob = wordlogprob * self.seqmask.expand_as(wordlogprob)
            oldseq_logprob = self.seqlogprob.expand_as(_candidate_logprob).contiguous()
            oldseq_logprob[:, :, 1:] = -999
            _candidate_logprob = self.seqmask * _candidate_logprob + oldseq_logprob * (1 - self.seqmask)

        selectedidx, selectedlogprob = self.select(t, _candidate_logprob, **kwargs)
        selected_beam = selectedidx // _candidate_logprob.shape[-1]  # //
        selectedwords = selectedidx % _candidate_logprob.shape[-1]  # 取余

        self.model.applytostates(self._expand_state(selected_beam, cur_beam_size))
        visual = self._expand_visual(visual, cur_beam_size, selected_beam)

        self.seqlogprob = selectedlogprob.unsqueeze(-1)
        self.seqmask = torch.gather(self.seqmask, 1, selected_beam.unsqueeze(-1))
        outs = list(torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outs)
        outs.append(selectedwords.unsqueeze(-1))

        if returnprobs:
            if t == 0:
                self.alllog_probs.append(wordlogprob.expand((self.BS, self.beamsize, -1)).unsqueeze(2))
            else:
                self.alllog_probs.append(wordlogprob.unsqueeze(2))

        thisword_logprob = torch.gather(wordlogprob, 1,
                                         selected_beam.unsqueeze(-1).expand(self.BS, self.beamsize,
                                                                            wordlogprob.shape[-1]))
        thisword_logprob = torch.gather(thisword_logprob, 2, selectedwords.unsqueeze(-1))
        self.logprobs_ = list(
            torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(self.BS, self.beamsize, 1)) for o in self.logprobs_)
        self.logprobs_.append(thisword_logprob)
        self.selectedwords = selectedwords.view(-1, 1)

        return visual, outs
