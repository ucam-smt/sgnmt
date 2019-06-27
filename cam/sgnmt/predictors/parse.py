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

import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor
import numpy as np


import collections

def load_external_ids(path):
    """
    load file of ids to list
    """
    logging.info('Loading ids from file {}'.format(path))
    with open(path) as f:
        ids = [int(line.strip()) for line in f]
    return set(ids)

class InternalHypo(object):
    """Helper class for internal parse predictor beam search over nonterminals
    """

    def __init__(self, score, token_score, predictor_state, word_to_consume):
        self.score = score
        self.predictor_state = predictor_state
        self.word_to_consume = word_to_consume
        self.norm_score = score
        self.token_score = token_score
        self.beam_len = 1

    def extend(self, score, predictor_state, word_to_consume):
        self.score += score
        self.predictor_state = predictor_state
        self.word_to_consume = word_to_consume
        self.beam_len += 1


class ParsePredictor(Predictor):
    """Predictor wrapper allowing internal beam search over a representation
    which contains some pre-defined 'non-terminal' ids, which should not appear 
    in the output.
    """
    def __init__(self, slave_predictor, normalize_scores=True, beam_size=4,
                 max_internal_len=35, nonterminal_ids=None):
        """Create a new parse wrapper for a predictor

        Args:
          slave_predictor: predictor to wrap with parse wrapper
          normalize_scores (bool): whether to normalize posterior scores,
                                   e.g. after some tokens have been removed
          beam_size (int): beam size for internal beam search over non-terminals
          max_internal_len (int): number of consecutive non-terminal tokens
                            allowed in internal search before path is ignored
          nonterminal_ids: file containing non-terminal ids, one per line          
        """
        super(ParsePredictor, self).__init__()
        self.predictor = slave_predictor
        self.normalize_scores = normalize_scores
        self.beam_size = beam_size
        self.max_internal_len = max_internal_len
        self.nonterminals = load_external_ids(nonterminal_ids)
        self.nonterminals.discard(utils.EOS_ID)
        self.nonterminals.discard(utils.UNK_ID)
        self.tok_to_hypo = {}

    def get_unk_probability(self, posterior):
        """Return unk probability as determined by slave predictor
        Returns:
            float, unk prob
        """
        return self.predictor.get_unk_probability(posterior)
   

    def are_best_terminal(self, posterior):
        """Return true if most probable tokens in posterior are all terminals
        (including EOS)
        """
        best_rule_ids = utils.argmax_n(posterior, self.beam_size)
        for tok in best_rule_ids:
            if tok in self.nonterminals:
                return False
        return True


    def predict_next(self, predicting_internally=False):
        """Predict next tokens.

        Args: 
          predicting_internally: will be true if called from internal 
                                 beam search, prevents infinite loop
        """
        original_posterior = self.predictor.predict_next()
        all_keys = utils.common_viewkeys(original_posterior)
        scores = {rule_id: original_posterior[rule_id] for rule_id in all_keys}

        scores = self.finalize_posterior(
            scores, 
            use_weights=True,
            normalize_scores=self.normalize_scores)
        if not predicting_internally:
            scores = self.find_word_beam(scores)
        return scores
    
    def maybe_add_new_top_tokens(self, top_terminals, hypo, next_hypos):
        new_post = self.predict_next(predicting_internally=True)
        top_tokens = utils.argmax_n(new_post, self.beam_size)
        next_state = copy.deepcopy(self.predictor.get_state())
        for tok in top_tokens:
            score = hypo.score + new_post[tok]
            new_hypo = InternalHypo(score, new_post[tok], next_state, tok)
            if tok not in self.nonterminals:
                add_hypo = False
                found = False
                for t in top_terminals:
                    if t == tok:
                        found = True
                        if self.tok_to_hypo[tok].score < new_hypo.score:
                            add_hypo = True
                            top_terminals.remove(t)
                        break
                if not found:
                    add_hypo = True
                if add_hypo:
                    top_terminals.append(tok)
                    self.tok_to_hypo[tok] = new_hypo
            else:
                next_hypos.append(new_hypo)

    def initialize_internal_hypos(self, posterior):
        top_tokens = utils.argmax_n(posterior, self.beam_size)
        hypos = []
        top_terminals = []
        for tok in top_tokens:
            new_hypo = InternalHypo(posterior[tok],
                                    posterior[tok],
                                    copy.deepcopy(self.predictor.get_state()),
                                    tok)
            if tok not in self.nonterminals:
                self.tok_to_hypo[tok] = new_hypo
                top_terminals.append(tok)
            hypos.append(new_hypo)
        return hypos, top_terminals

    def find_word_beam(self, posterior):
        """Internal beam search over posterior until a beam of terminals
        is found
        """
        hypos, top_terminals = self.initialize_internal_hypos(posterior)
        min_score = utils.NEG_INF
        if top_terminals:
            top_terminals.sort(key=lambda h: -self.tok_to_hypo[h].score)
            min_score = self.tok_to_hypo[top_terminals[-1]].score
        hypos.sort(key=lambda h: -h.score)
        # search until we have n terminal hypos with path scores better
        # than further internal search can give us
        while hypos and hypos[0].score > min_score:
            next_hypos = []
            for hypo in hypos:
                if not hypo.word_to_consume in self.nonterminals:
                    continue
                self.predictor.set_state(copy.deepcopy(hypo.predictor_state))
                self.consume(hypo.word_to_consume, internal=True)
                self.maybe_add_new_top_tokens(top_terminals, hypo, next_hypos)
            next_hypos.sort(key=lambda h: -h.score)
            hypos = next_hypos[:self.beam_size]
            top_terminals.sort(key=lambda t: -self.tok_to_hypo[t].score)
            top_terminals = top_terminals[:self.beam_size]
            if top_terminals:
                min_score = self.tok_to_hypo[top_terminals[-1]].score
        token_scores = [self.tok_to_hypo[t].score for t in top_terminals]
        return_post = {t: s for t, s in zip(top_terminals, token_scores)}
        return return_post

    def initialize(self, src_sentence):
        """Initializes slave predictor with source sentence 
        
        Args:
            src_sentence (list)
        """
        self.predictor.initialize(src_sentence)
    
    def consume(self, word, internal=False):
        try:
            if not internal:
                self.predictor.set_state(
                    copy.deepcopy(self.tok_to_hypo[word].predictor_state))
        except KeyError:
            logging.info('Consuming {}, not in tok-to-hypo'.format(word))
        return self.predictor.consume(word) 
    
    def get_state(self):
        """Returns the current state. """
        return self.predictor.get_state(), self.tok_to_hypo
    
    def set_state(self, state):
        """Sets the current state. """
        slave_state, tok_to_hypo = state
        self.tok_to_hypo = tok_to_hypo
        self.predictor.set_state(slave_state)
            
    def initialize_heuristic(self, src_sentence):
        """Creates a matrix of shortest distances between nodes. """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true if the current node is the same """
        return state1 == state2


class TokParsePredictor(ParsePredictor):
    """
    Unlike ParsePredictor, the grammar predicts tokens according to a grammar. 
    Use BPEParsePredictor if including rules to connect BPE units inside words.
    """
    
    def __init__(self, grammar_path, slave_predictor, word_out=True,
                 normalize_scores=True, norm_alpha=1.0, 
                 beam_size=1, max_internal_len=35, allow_early_eos=False,
                 consume_out_of_class=False):
        """Creates a new parse predictor wrapper.

        Args:
            grammar_path (string): Path to the grammar file
            slave_predictor: predictor to wrap
            word_out (bool): since this wrapper can be used for grammar
                             constraint, this bool determines whether we
                             also do internal beam search over non-terminals
            normalize_scores (bool): true if normalizing scores, e.g. if some
                                     are removed from the posterior
            norm_alpha (float): may be used for path weight normalization
            beam_size (int): beam size for internal beam search
            max_internal_len (int): max number of consecutive nonterminals
                                    before path is ignored by internal search
            allow_early_eos (bool): true if permitting EOS consumed even if
                                    it is not permitted by the grammar
                                    at that point
            consume_out_of_class (bool): true if permitting any tokens to be 
                                         consumed even if not allowed by the
                                         grammar at that point
        """
        super(TokParsePredictor, self).__init__(slave_predictor,
            normalize_scores, beam_size, max_internal_len, nonterminal_ids)
        self.grammar_path = grammar_path
        self.word_out = word_out
        self.stack = []
        self.norm_alpha = norm_alpha
        self.check_n_best_terminal = False
        self.current_lhs = None
        self.current_rhs = []
        self.allow_early_eos=allow_early_eos
        self.consume_ooc = consume_out_of_class
        self.prepare_grammar()
        self.tok_to_internal_state = {}
        
    def norm_hypo_score(self, hypo):
        hypo.norm_score = self.norm_score(hypo.score, hypo.beam_len)
        
    def norm_score(self, score, beam_len):
        length_penalty = (5.0 + beam_len) / 6
        if self.norm_alpha != 1.0:
            length_penalty = pow(length_penalty, self.norm_alpha)
        return score / length_penalty

    def prepare_grammar(self):
        self.lhs_to_can_follow = {}
        with open(self.grammar_path) as f:
            for line in f:
                nt, rule = line.split(':')
                nt = int(nt.strip())
                self.lhs_to_can_follow[nt] = set(
                    [int(r) for r in rule.strip().split()])
        self.last_nt_in_rule = {nt: True for nt in self.lhs_to_can_follow}
        for nt, following in self.lhs_to_can_follow.items():
            if 0 in following:
                following.remove(0)
                self.last_nt_in_rule[nt] = False
            if self.allow_early_eos and utils.UNK_ID in following:
                self.lhs_to_can_follow[nt].add(utils.EOS_ID)
        self.lhs_to_can_follow[utils.EOS_ID].add(utils.UNK_ID)
                    

    def initialize(self, src_sentence):
        self.predictor.initialize(src_sentence)
        self.current_lhs = None
        self.current_rhs = []
        self.stack = [utils.EOS_ID]
        self.consume(utils.GO_ID)

    def replace_lhs(self):
        while self.current_rhs:
            self.stack.append(self.current_rhs.pop())
        if self.stack:
            self.current_lhs = self.stack.pop()
        else:
            self.current_lhs = utils.EOS_ID
            
    def get_current_allowed(self):
        if self.current_lhs:
            return self.lhs_to_can_follow[self.current_lhs]
        return set([utils.GO_ID])


    def predict_next(self, predicting_next_word=False):
        """predict next tokens as permitted by 
        the current stack and the grammar
        """
        original_posterior = self.predictor.predict_next()
        outgoing_rules = self.lhs_to_can_follow[self.current_lhs]       
        scores = {rule_id: original_posterior[rule_id] 
                  for rule_id in outgoing_rules}
        scores = self.finalize_posterior(
            scores, 
            use_weights=True,
            normalize_scores=self.normalize_scores)
        if self.word_out and not predicting_next_word:
            scores = self.find_word(scores)
        return scores
    
    def find_word_greedy(self, posterior):
        while not self.are_best_terminal(posterior):
            best_rule_id = utils.argmax(posterior)
            self.consume(best_rule_id)
            posterior = self.predict_next(predicting_next_word=True)
        return posterior

    def find_word_beam(self, posterior):
        """
        Do an internal beam search over non-terminal functions to find 
        the next best n terminal tokens, as ranked by normalized path score
        
        Returns: posterior containing up to n terminal tokens 
                 and their normalized path score
        """
        top_tokens = utils.argmax_n(posterior, self.beam_size)
        hypos = [InternalHypo(posterior[tok], self.get_state(), tok) 
                 for tok in top_tokens if tok in self.nonterminals]
        best_hypo = InternalHypo(utils.NEG_INF, None, None)
        best_posterior = None
        while hypos and hypos[0].norm_score > best_hypo.norm_score:
            next_hypos = []
            for hypo in hypos:
                self.set_state(copy.deepcopy(hypo.predictor_state))
                self.consume(hypo.word_to_consume)
                new_post = self.predict_next(predicting_next_word=True)
                top_tokens = utils.argmax_n(new_post, self.beam_size)
                next_state = copy.deepcopy(self.get_state())
                new_norm_score = self.norm_score(
                    new_post[top_tokens[0]] + hypo.score, hypo.beam_len + 1)
                if (self.are_best_terminal(new_post) and
                    new_norm_score > best_hypo.norm_score):
                    best_hypo = copy.deepcopy(hypo)
                    best_hypo.predictor_state = next_state
                    best_hypo.norm_score = new_norm_score
                    best_posterior = new_post
                    self.norm_score(best_hypo)
                else:
                    if hypo.beam_len == self.max_internal_len:
                        logging.info('cutting off internal hypo - too long')
                        continue
                    for tok in top_tokens:
                        if tok in self.nonterminals:
                            new_hypo = copy.deepcopy(hypo)
                            new_hypo.extend(new_post[tok], next_state, tok)
                            next_hypos.append(new_hypo)

            map(self.norm_hypo_score, next_hypos)
            next_hypos.sort(key=lambda h: -h.norm_score)
            hypos = next_hypos[:self.beam_size]
        self.set_state(best_hypo.predictor_state)
        for tok in best_posterior.keys():
            best_posterior[tok] = self.norm_score(
                best_hypo.score + best_posterior[tok], best_hypo.beam_len + 1)
            if tok in self.nonterminals:
                del best_posterior[tok]
        return best_posterior

    def find_word(self, posterior):
        """Check whether rhs of best option in posterior is a terminal
        if it is, return the posterior for decoding
        if not, take the best result and follow that path until a word is found
        this follows a greedy 1best or a beam path through non-terminals
        """
        if self.beam_size <= 1:
            return self.find_word_greedy(posterior)
        else:
          if self.are_best_terminal(posterior):
              return posterior
          else:
              return self.find_word_beam(posterior)

    def consume(self, word):
        """
        Args:
            word (int): word token being consumed
        """
        change_to_unk = (
            (word == utils.UNK_ID) or
            (not self.consume_ooc and word not in self.get_current_allowed()))
        if change_to_unk:
            word = utils.UNK_ID
        self.update_stacks(word)
        return self.predictor.consume(word) 
        
    
    def update_stacks(self, word):
        if word in self.nonterminals:
            self.current_rhs.append(word)
            if self.last_nt_in_rule[word]:
                self.replace_lhs()
        else:
            self.replace_lhs()  

    def get_state(self):
        """Returns the current state, including slave predictor state """
        return (self.stack, self.current_lhs, self.current_rhs,
                self.predictor.get_state())
    
    def set_state(self, state):
        """Sets the current state """
        self.stack, self.current_lhs, self.current_rhs, slave_state = state
        self.predictor.set_state(slave_state)


class BpeParsePredictor(TokParsePredictor):
    """                                                                         
    Predict over a BPE-based grammar with two possible grammar constraints:
    one between non-terminals and bpe start-of-word tokens, one over
    bpe tokens in a word
    """

    def __init__(self, grammar_path, bpe_rule_path, slave_predictor,
                 word_out=True, normalize_scores=True,
                 norm_alpha=1.0, beam_size=1, max_internal_len=35,
                 allow_early_eos=False, consume_out_of_class=False, 
                 eow_ids=None, terminal_restrict=True, terminal_ids=None,
                 internal_only_restrict=False):
        """Creates a new parse predictor wrapper which can be constrained to 2
           grammars: one over non-terminals / terminals, one internally to
           constrain BPE units within a single word
        Args: 
            grammar_path (string): Path to the grammar file
            bpe_rule_path (string): Path to file defining rules between BPEs
            slave_predictor: predictor to wrap
            word_out (bool): since this wrapper can be used for grammar
                             constraint, this bool determines whether we
                             also do internal beam search over non-terminals
            normalize_scores (bool): true if normalizing scores, e.g. if some
                                     are removed from the posterior
            norm_alpha (float): may be used for path weight normalization
            beam_size (int): beam size for internal beam search
            max_internal_len (int): max number of consecutive nonterminals
                                    before path is ignored by internal search
            allow_early_eos (bool): true if permitting EOS consumed even if
                                    it is not permitted by the grammar
                                    at that point
            consume_out_of_class (bool): true if permitting any tokens to be
                                         consumed even if not allowed by the
                                         grammar at that point
            eow_ids (string): path to file containing ids of BPEs that mark
                              the end of a word
            terminal_restrict (bool): true if applying grammar constraint over 
                                      nonterminals and terminals
            terminal_ids (string): path to file containing all terminal ids
            internal_only_restrict (bool): true if applying grammar constraint
                                           over BPE units inside words
        """
        super(BpeParsePredictor, self).__init__(grammar_path,
                                                slave_predictor,
                                                word_out,
                                                normalize_scores,
                                                norm_alpha,
                                                beam_size,
                                                max_internal_len,
                                                allow_early_eos,
                                                consume_out_of_class)
        self.internal_only_restrict = internal_only_restrict
        self.terminal_restrict = terminal_restrict
        self.eow_ids = self.get_eow_ids(eow_ids)
        self.all_terminals = self.get_all_terminals(terminal_ids)
        self.get_bpe_can_follow(bpe_rule_path)
        
    def get_eow_ids(self, eow_ids):
        eows = set()
        if eow_ids:
            with open(eow_ids) as f:
                for line in f:
                    eows.add(int(line.strip()))
        return eows
        
    def get_all_terminals(self, terminal_ids):
        all_terminals = set([utils.EOS_ID])
        if terminal_ids:
            with open(terminal_ids) as f:
                for line in f:
                    all_terminals.add(int(line.strip()))
            if not self.terminal_restrict:
                for terminal in all_terminals:
                    self.lhs_to_can_follow[terminal] = all_terminals
        return all_terminals

    def get_bpe_can_follow(self, rule_path):
        with open(rule_path) as f:
            for line in f:
                nt, following = line.split(' : ')
                nt_tuple = tuple(map(int, nt.split(',')))
                following = set([int(r) for r in following.strip().split()])
                self.lhs_to_can_follow[nt_tuple] = following
                
    def update_stacks(self, word):
        if word in self.nonterminals:
            self.current_rhs.append(word)
            if self.last_nt_in_rule[word]:
                self.replace_lhs()
        else:
            if self.terminal_restrict:
                try:
                    internal_lhs = self.current_lhs + (word,)
                except TypeError:
                    internal_lhs = (self.current_lhs, word)
            else:
                internal_lhs = word
            if not self.terminal_restrict and word in self.eow_ids:
                self.replace_lhs()
            elif (not self.terminal_restrict or 
                  internal_lhs in self.lhs_to_can_follow):
                self.current_lhs = internal_lhs
            else:
                self.replace_lhs()

    def is_nt(self, word):
        if word in self.all_terminals:
            return False
        return True

    def predict_next(self, predicting_next_word=False):
        """predict next tokens as permitted by 
        the current stack and the BPE grammar
        """
        original_posterior = self.predictor.predict_next()
        outgoing_rules = self.lhs_to_can_follow[self.current_lhs]
        scores = {rule_id: original_posterior[rule_id] 
                  for rule_id in outgoing_rules}
        if self.internal_only_restrict and self.are_best_terminal(scores):
            outgoing_rules = self.all_terminals
            scores = {rule_id: original_posterior[rule_id]
                      for rule_id in outgoing_rules}
        scores = self.finalize_posterior(
            scores, 
            use_weights=True,
            normalize_scores=self.normalize_scores)
        if self.word_out and not predicting_next_word:
            scores = self.find_word(scores)
        return scores

