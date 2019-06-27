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

"""Implementation of beam search with explicit limits on culmulative
predictor scores at each node expansion.
"""

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
import logging


class PredLimitBeamDecoder(BeamDecoder):
    """Beam search variant with explicit limits on the culmulative
    predictor scores at each node expansion.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder with culmulative predictor score
        limits. In addition to the constructor of `BeamDecoder`, the 
        following values are fetched from `decoder_args`:
        
            pred_limits (string): Comma-separated list of predictor
                                  score limits.
        """
        super(PredLimitBeamDecoder, self).__init__(decoder_args)
        self.pred_limits = []
        for l in utils.split_comma(decoder_args.pred_limits):
          try:
              self.pred_limits.append(float(l))
          except:
              self.pred_limits.append(utils.NEG_INF)
        logging.info("Cumulative predictor score limits: %s" % self.pred_limits)
    
    def _expand_hypo(self, hypo):
        n_preds = len(self.pred_limits)
        all_hypos = super(PredLimitBeamDecoder, self)._expand_hypo(hypo)
        next_hypos = []
        all_accs = [0.0] * n_preds
        for hypo in sorted(all_hypos, key=lambda h: -h.score):
            valid = True
            for limit, acc, breakdown in zip(self.pred_limits,
                                             all_accs,
                                             hypo.score_breakdown[-1]):
                if acc + breakdown[0] < limit:
                    valid = False
                    break
            if valid:
                for i in range(n_preds):
                    all_accs[i] += hypo.score_breakdown[-1][i][0]
                next_hypos.append(hypo)
        return next_hypos

