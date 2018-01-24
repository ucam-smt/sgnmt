"""Implementation of beam search which applies combination_sheme at
each time step.
"""

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding import core
import copy
import logging


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
        if decoder_args.combination_scheme == 'length_norm':
            self.breakdown2score = core.breakdown2score_length_norm
        if decoder_args.combination_scheme == 'bayesian_loglin':
            self.breakdown2score = core.breakdown2score_bayesian_loglin
        if decoder_args.combination_scheme == 'bayesian':
            self.breakdown2score = core.breakdown2score_bayesian
        if decoder_args.combination_scheme == 'sum':
            self.breakdown2score = core.breakdown2score_sum
            logging.warn("Using the sum combination strategy has no effect "
                          "under the combibeam decoder.")
        self.maintain_best_scores = False

    def _expand_hypo(self, hypo):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expans
        
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
            expanded_hypo.score = self.breakdown2score(
                    expanded_hypo.score, expanded_hypo.score_breakdown)
        expanded_hypos.sort(key=lambda x: -x.score)
        return expanded_hypos[:self.beam_size]

