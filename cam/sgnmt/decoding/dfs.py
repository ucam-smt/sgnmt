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

"""Implementation of the dfs search strategy """

import copy
import logging
import operator
import math
import numpy as np

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class DFSDecoder(Decoder):
    """This decoder implements depth first search without using
    heuristics. This is the most efficient search algorithm for 
    complete enumeration of the search space as it minimizes the 
    number of ``get_state()`` and ``set_state()`` calls. Note that
    this DFS implementation has no cycle detection, i.e. if the search
    space has cycles this decoder may run into an infinite loop.
    """
    
    def __init__(self, decoder_args):
        """Creates new DFS decoder instance. The following values are
        fetched from `decoder_args`:
        
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            early_stopping (bool): Enable safe (admissible) branch
                                   pruning if the accumulated score
                                   is already worse than the currently
                                   best complete score. Do not use if
                                   scores can be positive
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(DFSDecoder, self).__init__(decoder_args)
        self.early_stopping = decoder_args.early_stopping
        self.max_expansions_param = decoder_args.max_node_expansions
    
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        ATTENTION: Early stopping plus DFS produces wrong results if 
        you have positive scores!
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        if (partial_hypo.get_last_word() == utils.EOS_ID
                or len(partial_hypo.trgt_sentence) > self.max_len):
            self.add_full_hypo(partial_hypo.generate_full_hypothesis())
            self.best_score = max(self.best_score, partial_hypo.score)
            return
        if self.apply_predictors_count > self.max_expansions: # pruning
            return
        posterior,score_breakdown = self.apply_predictors() 
        reload_states = False
        if self.early_stopping:
            worst_score = self.best_score - partial_hypo.score
            children = [i for i in posterior.items() if i[1] > worst_score]
        else:
            children = [i for i in posterior.items()]
        if len(children) > 1: # deep copy only if necessary
            pred_states = copy.deepcopy(self.get_predictor_states())
        logging.debug("Expand: best_score: %f exp: %d partial_score: "
                      "%f children: %d sentence: %s" %
                      (self.best_score,
                       self.apply_predictors_count,
                       partial_hypo.score,
                       len(children),
                       partial_hypo.trgt_sentence))
        for trgt_word,score in sorted(children,
                                      key=operator.itemgetter(1),
                                      reverse=True):
            new_hypo =  partial_hypo.expand(trgt_word,
                                            None, # Do not store states
                                            score,
                                            score_breakdown[trgt_word])
            if self.early_stopping and new_hypo.score < self.best_score:
                return
            if reload_states: # TODO: save one deepcopy (in last iteration)
                self.set_predictor_states(copy.deepcopy(pred_states))
            self.consume(trgt_word)
            self._dfs(new_hypo)
            reload_states = True
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using depth first search.
        If ``max_expansions`` equals 0, this corresponds to exhaustive
        search for the globally best scoring hypothesis. Note that with
        ``early_stopping`` enabled, the returned set of hypothesis are
        not necessarily the global n-best hypotheses. To create an 
        exact n-best list, disable both ``max_expansions`` and 
        ``early_stopping`` in the constructor.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score. If ``max_expansions`` equals 0, the first element
            holds the global best scoring hypothesis
        """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        self.best_score = self.get_lower_score_bound()
        self._dfs(PartialHypothesis())
        return self.get_full_hypos_sorted()


class SimpleDFSDecoder(Decoder):
    """This is a stripped down version of the DFS decoder which is
    designed to explore the entire search space. SimpleDFS is
    intended to be used with a `score_lower_bounds_file` from a
    previous beam search run which already contains good lower
    bounds. SimpleDFS verifies whether the lower bound is actually
    the global best score.

    SimpleDFS can only be used with a single predictor.

    SimpleDFS does not support max_expansions or max_len_factor.
    early_stopping cannot be disabled.
    """
    
    def __init__(self, decoder_args):
        """Creates new SimpleDFS decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SimpleDFSDecoder, self).__init__(decoder_args)
        #self._min_length_ratio = 0.25  # TODO: Make configurable
        self._min_length_ratio = -0.1
        self._min_length = -100
    
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        if partial_hypo.get_last_word() == utils.EOS_ID:
            if len(partial_hypo.trgt_sentence) >= self._min_length:
                self.add_full_hypo(partial_hypo.generate_full_hypothesis())
                if partial_hypo.score > self.best_score:
                    self.best_score = partial_hypo.score
                    logging.info("New best: score: %f exp: %d sentence: %s" %
                          (self.best_score,
                           self.apply_predictors_count,
                           partial_hypo.trgt_sentence))
            return

        self.apply_predictors_count += 1
        posterior = self.dfs_predictor.predict_next()
        logging.debug("Expand: best_score: %f exp: %d partial_score: "
                      "%f sentence: %s" %
                      (self.best_score,
                       self.apply_predictors_count,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        first_expansion = True
        for trgt_word, score in utils.common_iterable(posterior):
            if partial_hypo.score + score > self.best_score:
                if first_expansion:
                    pred_states = copy.deepcopy(self.get_predictor_states())
                    first_expansion = False
                else:
                    self.set_predictor_states(copy.deepcopy(pred_states))
                self.consume(trgt_word)
                self._dfs(partial_hypo.expand(trgt_word,
                                              None, # Do not store states
                                              score,
                                              [(score, 1.0)]))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence exhaustively using depth 
        first search.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        if len(self.predictors) != 1:
            logging.fatal("SimpleDFS only works with a single predictor!")
        self.dfs_predictor = self.predictors[0][0]
        if self._min_length_ratio > 0.0:
            self._min_length = int(math.ceil(
              self._min_length_ratio * len(src_sentence))) + 1
        self.initialize_predictors(src_sentence)
        self.best_score = self.get_lower_score_bound()
        self._dfs(PartialHypothesis())
        return self.get_full_hypos_sorted()


class SimpleLengthDFSDecoder(Decoder):
    """This is a length dependent version of SimpleDFS. This
    decoder finds the global best scores for certain hypo lengths.
    The `simplelendfs_lower_bounds_file` contains lines of the form
    
      <length1>:<lower-bound> ... <lengthN>:<lower-boundN>

    that specify length dependent score lower bounds. The decoder
    will search for the optimal model scores for the specified
    lengths.

    SimpleDFS can only be used with a single predictor.

    SimpleDFS does not support max_expansions or max_len_factor.
    early_stopping cannot be disabled.
    """
    
    def __init__(self, decoder_args):
        """Creates new SimpleDFS decoder instance. 

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SimpleLengthDFSDecoder, self).__init__(decoder_args)
        with open(decoder_args.simplelendfs_lower_bounds_file) as f:
            self.all_lower_bounds = [{tuple(el.split(":"))
                                 for el in line.strip().split()} for line in f]
    
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        partial_hypo_length = len(partial_hypo.trgt_sentence)
        self.apply_predictors_count += 1
        posterior = self.dfs_predictor.predict_next()
        logging.debug("Expand: exp: %d partial_score: "
                      "%f sentence: %s" %
                      (self.apply_predictors_count,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        # Check EOS
        eos_score = posterior[utils.EOS_ID]
        if (self.len_enabled[partial_hypo_length] 
               and partial_hypo.score + eos_score 
                   > self.len_lower_bounds[partial_hypo_length]):
            eos_hypo = partial_hypo.expand(utils.EOS_ID,
                                           None, # Do not store states
                                           eos_score,
                                           [(eos_score, 1.0)])
            logging.info("New best: len: %d score: %f -> %f exp: %d" %
                          (partial_hypo_length,
                           self.len_lower_bounds[partial_hypo_length],
                           eos_hypo.score,
                           self.apply_predictors_count))
            self.len_lower_bounds[partial_hypo_length] = eos_hypo.score
            self.len_best_hypos[partial_hypo_length] = eos_hypo
            self._update_min_lower_bounds()
        if partial_hypo_length >= self.max_len:
            return
        first_expansion = True
        for trgt_word, score in utils.common_iterable(posterior):
            if trgt_word != utils.EOS_ID:
                lower_bound = self.len_min_lower_bounds[partial_hypo_length+1]
                if partial_hypo.score + score > lower_bound:
                    if first_expansion:
                        pred_states = copy.deepcopy(self.get_predictor_states())
                        first_expansion = False
                    else:
                        self.set_predictor_states(copy.deepcopy(pred_states))
                    self.consume(trgt_word)
                    self._dfs(partial_hypo.expand(trgt_word,
                                                  None, # Do not store states
                                                  score,
                                                  [(score, 1.0)]))

    def _update_min_lower_bounds(self):
        min_lower_bound = np.inf
        for l in range(self.max_len, -1, -1):
            if self.len_enabled[l]:
                min_lower_bound = min(min_lower_bound, self.len_lower_bounds[l])
            self.len_min_lower_bounds[l] = min_lower_bound
        logging.info("Lower bounds: %s" % (self.len_min_lower_bounds,))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence exhaustively using depth 
        first search under length constraints.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        if len(self.predictors) != 1:
            logging.fatal("SimpleDFS only works with a single predictor!")
        self.dfs_predictor = self.predictors[0][0]
        self.initialize_predictors(src_sentence)
        lower_bounds = self.all_lower_bounds[self.current_sen_id]
        self.max_len = max(int(el[0]) for el in lower_bounds)
        self.len_enabled = np.zeros((self.max_len + 1,), np.bool)
        self.len_lower_bounds = -np.ones((self.max_len + 1,)) * np.inf
        self.len_min_lower_bounds = np.zeros((self.max_len + 1,))
        self.len_best_hypos = [None] * (self.max_len+1)
        for el in lower_bounds:
            l = int(el[0])
            self.len_enabled[l] = True
            self.len_lower_bounds[l] = float(el[1])
        self._update_min_lower_bounds()
        self._dfs(PartialHypothesis())
        for hypo in self.len_best_hypos:
            if hypo is not None:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()

