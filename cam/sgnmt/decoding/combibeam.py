"""Implementation of beam search which applies combination_sheme at
each time step.
"""

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding import combination
from cam.sgnmt.decoding.core import PartialHypothesis
import copy
import logging


class CombiStatePartialHypo(PartialHypothesis):

    def __init__(self, initial_states=None):
        super(CombiStatePartialHypo, self).__init__(initial_states)
        self.score_minus_last = 0 # score not counting last step
        self.pred_weights = [] # overall state-dependent predictor weights
        
    def _new_partial_hypo(self, states, word, score, score_breakdown):
        new_hypo = CombiStatePartialHypo(states)
        new_hypo.score_minus_last = self.score
        new_hypo.score = self.score + score
        new_hypo.score_breakdown = copy.copy(self.score_breakdown)
        new_hypo.trgt_sentence = self.trgt_sentence + [word]
        new_hypo.pred_weights = copy.copy(self.pred_weights)
        new_hypo.score_breakdown.append(score_breakdown)
        return new_hypo


class CombiBeamDecoder(BeamDecoder):
    """This beam search implementation is a modification to the hypo
    expansion strategy. Rather than selecting hypotheses based on
    the sum of the previous hypo scores and the current one, we
    apply combination_scheme in each time step. This makes it possible
    to use schemes like Bayesian combination on the word rather than
    the full sentence level.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. In addition to the 
        constructor of `BeamDecoder`, the following values are fetched 
        from `decoder_args`:
        
            combination_scheme (string): breakdown2score strategy
        """
        super(CombiBeamDecoder, self).__init__(decoder_args)
        # Whether to pass combination cached predictor weights
        self.store_combination_states = True 
        if decoder_args.combination_scheme == 'length_norm':
            self.breakdown2score = combination.breakdown2score_length_norm
        if decoder_args.combination_scheme == 'bayesian_loglin':
            self.breakdown2score = combination.breakdown2score_bayesian_loglin
        if decoder_args.combination_scheme == 'bayesian_state_dependent':
            self.breakdown2score = combination.breakdown2score_bayesian_state_dependent
        if decoder_args.combination_scheme == 'bayesian':
            self.breakdown2score = combination.breakdown2score_bayesian
        if decoder_args.combination_scheme == 'sum':
            self.breakdown2score = combination.breakdown2score_sum
        if decoder_args.combination_scheme in ['sum', 'length_norm']:
            self.store_combination_states = False
            logging.warn("Using the %s combination strategy has no effect "
                         "under the combibeam decoder."
                         % decoder_args.combination_scheme)
        self.maintain_best_scores = False

    def _get_initial_hypos(self):
        """Get list containing an initial CombiStatePartialHypothesis"""
        return [CombiStatePartialHypo(self.get_predictor_states())]


    def _expand_hypo(self, hypo):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand        
        Returns:
            list. List of child hypotheses
        """
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        posterior, score_breakdown = self.apply_predictors()
        hypo.predictor_states = self.get_predictor_states()
        expanded_hypos = [hypo.cheap_expand(w, s, score_breakdown[w]) 
                          for w, s in utils.common_iterable(posterior)]
        for expanded_hypo in expanded_hypos:
            if self.store_combination_states:
                expanded_hypo.score = self.breakdown2score(
                    expanded_hypo.score,
                    expanded_hypo.score_breakdown,
                    prev_score=expanded_hypo.score_minus_last,
                    pred_weights=expanded_hypo.pred_weights)
            else:
                expanded_hypo.score = self.breakdown2score(
                    expanded_hypo.score,
                    expanded_hypo.score_breakdown)
        expanded_hypos.sort(key=lambda x: -x.score)
        return expanded_hypos[:self.beam_size]

