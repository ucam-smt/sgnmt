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

"""Implementation of the greedy search strategy """

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, Hypothesis
import logging


class GreedyDecoder(Decoder):
    """The greedy decoder does not revise decisions and therefore does
    not have to maintain predictor states. Therefore, this 
    implementation is particularly simple and can be used as template
    for more complex decoders. The greedy decoder can be imitated with
    the ``BeamDecoder`` with beam size 1.
    """
    
    def __init__(self, decoder_args):
        """Initialize the greedy decoder. """
        super(GreedyDecoder, self).__init__(decoder_args)
    
    def decode(self, src_sentence):
        """Decode a single source sentence in a greedy way: Always take
        the highest scoring word as next word and proceed to the next
        position. This makes it possible to decode without using the 
        predictors ``get_state()`` and ``set_state()`` methods as we
        do not have to keep track of predictor states.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of a single best ``Hypothesis`` instance."""
        self.initialize_predictors(src_sentence)
        trgt_sentence = []
        score_breakdown = []
        trgt_word = None
        score = 0.0
        while trgt_word != utils.EOS_ID and len(trgt_sentence) <= self.max_len:
            posterior,breakdown = self.apply_predictors(1)
            trgt_word = utils.argmax(posterior)
            score += posterior[trgt_word]
            trgt_sentence.append(trgt_word)
            logging.debug("Partial hypothesis (%f): %s" % (
                    score, " ".join([str(i) for i in trgt_sentence]))) 
            score_breakdown.append(breakdown[trgt_word])
            self.consume(trgt_word)
        self.add_full_hypo(Hypothesis(trgt_sentence, score, score_breakdown))
        return self.full_hypos
