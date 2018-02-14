"""The syntax beam secoding strategy ensures diversity in the terminals."""


import copy
import logging
import numpy as np

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder


class SyntaxBeamDecoder(BeamDecoder):
    """The syntax beam search strategy is an extension of beam search 
    which ensures diversity amongst the terminals in the active
    hypotheses. The decoder clusters hypotheses by their terminal 
    history. Each cluster cannot have more than beam_size hypos, and
    the number of clusters is topped by beam_size. This means that
    the effective beam size varies between beam_size and beam_size^2,
    and there are always beam_size different terminal histories in the
    active beam.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance for system level 
        combination. In addition to the constructor of
        `BeamDecoder`, the following values are fetched from 
        `decoder_args`:
        
            syntax_max_terminal_id (int): Synchronization symbol. If negative, fetch
                             '</w>' from ``utils.trg_cmap`` 
        """
        super(SyntaxBeamDecoder, self).__init__(decoder_args)
        self.max_terminal_id = decoder_args.syntax_max_terminal_id

    def _get_next_hypos_diverse(self, hypos, scores):
        """Get hypotheses of the next time step.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses.
        """
        new_hypos = []
        terminal_history_counts = {}
        for idx in reversed(np.argsort(scores)):
            candidate = hypos[idx]
            key = " ".join([str(i) for i in candidate.trgt_sentence 
                                   if i <= self.max_terminal_id])
            cnt = terminal_history_counts.get(key, 0)
            if cnt >= self.beam_size:
                continue
            valid = True
            if self.hypo_recombination:
                self.set_predictor_states(copy.deepcopy(candidate.predictor_states))
                if not candidate.word_to_consume is None:
                    self.consume(candidate.word_to_consume)
                    candidate.word_to_consume = None
                    candidate.predictor_states = self.get_predictor_states()
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
                terminal_history_counts[key] = cnt + 1
                new_hypos.append(candidate)
                if len(terminal_history_counts) >= self.beam_size:
                    break
        return new_hypos
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        self.min_score = utils.NEG_INF
        while self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self._get_combined_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo):
                    next_score = self._get_combined_score(next_hypo)
                    next_hypos.append(next_hypo)
                    next_scores.append(next_score)
            hypos = self._get_next_hypos_diverse(next_hypos, next_scores)
        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()
