# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the lenbeam search strategy """

import copy
from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class LengthBeamDecoder(Decoder):
    """This beam decoder variant finds hypotheses for all lengths up
    to the maximum hypo length. At each time step, all EOS extensions
    are added to the results set.
    """

    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. The following values
        are fetched from `decoder_args`:
        
            beam (int): Absolute beam size. A beam of 12 means
                        that we keep track of 12 active hypotheses
        """
        super(LengthBeamDecoder, self).__init__(decoder_args)
        self.beam_size = decoder_args.beam

    def _expand_hypo(self, hypo):
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        posterior, score_breakdown = self.apply_predictors()
        hypo.predictor_states = self.get_predictor_states()
        top = utils.argmax_n(posterior, self.beam_size)
        # EOS hypo
        eos_hypo = hypo.cheap_expand(utils.EOS_ID,
                                     posterior[utils.EOS_ID],
                                     score_breakdown[utils.EOS_ID])
        self.add_full_hypo(eos_hypo.generate_full_hypothesis())
        # All other hypos
        return [hypo.cheap_expand(
                      trgt_word,
                      posterior[trgt_word],
                      score_breakdown[trgt_word]) for trgt_word in top 
                                                  if trgt_word != utils.EOS_ID]

    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]
        for _ in range(self.max_len):
            next_hypos = []
            for hypo in hypos:
                next_hypos.extend(self._expand_hypo(hypo))
            next_hypos.sort(key=lambda h: -h.score)
            hypos = next_hypos[:self.beam_size]
        return self.get_full_hypos_sorted()

