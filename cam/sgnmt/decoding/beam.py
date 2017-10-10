"""Implementation of the beam search strategy """

import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
import numpy as np


class BeamDecoder(Decoder):
    """This decoder implements standard beam search and several
    variants of it such as diversity promoting beam search and beam
    search with heuristic future cost estimates. This implementation
    supports risk-free pruning and hypotheses recombination.
    """
    
    def __init__(self,
                 decoder_args,
                 hypo_recombination,
                 beam_size,
                 pure_heuristic_scores = False, 
                 diversity_factor = -1.0,
                 early_stopping = True):
        """Creates a new beam decoder instance
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination 
            beam_size (int): Absolute beam size. A beam of 12 means
                             that we keep track of 12 active hypotheses
            pure_heuristic_scores (bool): Hypotheses to keep in the beam
                                          are normally selected 
                                          according the sum of partial
                                          hypo score and future cost
                                          estimates. If set to true, 
                                          partial hypo scores are 
                                          ignored.
            diversity_factor (float): If this is set to a positive 
                                      value we add diversity promoting
                                      penalization terms to the partial
                                      hypothesis scores following Li
                                      and Jurafsky, 2016
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable
        """
        super(BeamDecoder, self).__init__(decoder_args)
        self.diversity_factor = diversity_factor
        self.diverse_decoding = (diversity_factor > 0.0)
        if diversity_factor > 0.0:
            logging.fatal("Diversity promoting beam search is not implemented "
                          "yet")
        self.beam_size = beam_size
        self.hypo_recombination = hypo_recombination
        self.stop_criterion = self._best_eos if early_stopping else self._all_eos
        self.maintain_best_scores = early_stopping and not hypo_recombination 
        if self.maintain_best_scores:
            logging.debug("Risk-free beam-search pruning enabled")
        self.pure_heuristic_scores = pure_heuristic_scores
    
    def _get_combined_score(self, hypo):
        """Combines hypo score with future cost estimates."""
        est_score = -self.estimate_future_cost(hypo)
        if not self.pure_heuristic_scores:
            return est_score + hypo.score
        return est_score

    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        return hypos[0].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        for hypo in hypos:
            if hypo.get_last_word() != utils.EOS_ID:
                return True
        return False
    
    def _expand_hypo(self, hypo):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expans
        
        Returns:
            list. List of child hypotheses
        """
        if hypo.score <= self.min_score:
            return []
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        posterior,score_breakdown = self.apply_predictors()
        hypo.predictor_states = self.get_predictor_states()
        top = utils.argmax_n(posterior, self.beam_size)
        return [hypo.cheap_expand(
                            trgt_word,
                            posterior[trgt_word],
                            score_breakdown[trgt_word]) for trgt_word in top]
    
    def _filter_equal_hypos(self, hypos, scores):
        """Apply hypo recombination to the hypotheses in ``hypos``.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses in ``hypos`` after applying
            hypotheses recombination.
        """
        new_hypos = []
        for idx in reversed(np.argsort(scores)):
            candidate = hypos[idx]
            self.set_predictor_states(copy.deepcopy(candidate.predictor_states))
            if not candidate.word_to_consume is None:
                self.consume(candidate.word_to_consume)
                candidate.word_to_consume = None
                candidate.predictor_states = self.get_predictor_states()
            valid = True
            for hypo in new_hypos:
                if self.are_equal_predictor_states(
                                                hypo.predictor_states,
                                                candidate.predictor_states):
                    logging.debug("Hypo recombination: %s > %s" % (
                                                 hypo.trgt_sentence,
                                                 candidate.trgt_sentence))
                    valid = False
                    break
            if valid:
                new_hypos.append(candidate)
                if len(new_hypos) >= self.beam_size:
                    break
        return new_hypos

    def _get_next_hypos(self, all_hypos, all_scores):
        """Get hypos for the next iteration. """
        hypos = [all_hypos[idx]
                        for idx in np.argsort(all_scores)[-self.beam_size:]]
        hypos.reverse()
        return hypos
    
    def _register_score(self, score):
        """Updates best_scores and min_score. """
        if not self.maintain_best_scores:
            return
        self.best_scores.append(score)
        self.best_scores.sort(reverse=True)
        if len(self.best_scores) >= self.beam_size:
            self.best_scores = self.best_scores[:self.beam_size]
            self.min_score = self.best_scores[-1] 
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [PartialHypothesis(self.get_predictor_states())]
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        while self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            self.min_score = utils.NEG_INF
            self.best_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self._get_combined_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo):
                    next_score = self._get_combined_score(next_hypo)
                    if next_score > self.min_score:
                        next_hypos.append(next_hypo)
                        next_scores.append(next_score)
                        self._register_score(next_score)
            if self.hypo_recombination:
                hypos = self._filter_equal_hypos(next_hypos, next_scores)
            else:
                hypos = self._get_next_hypos(next_hypos, next_scores)
        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()
