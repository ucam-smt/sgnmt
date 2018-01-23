"""Implementation of beam search which does not combine all predictor
scores but keeps only one predictor alive for each hypo in the 
beam. Good for approximate and efficient ensembling.
"""


import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.core import PartialHypothesis


class SepBeamDecoder(BeamDecoder):
    """This beam search implementation breaks with the predictor
    abstraction via the ``Decoder.apply_predictors()`` and 
    ``Decoder.consume()`` interfaces. We do not use combined scores
    of all predictors, but link single predictors to hypotheses in 
    the beam. On hypo expansion, we call ``predict_next()`` only on
    this predictor. This is suitable for approximated ensembling as
    it reduces the runtime nearly to a single system run.
    
    Note that ``PartialHypothesis.predictor_states`` holds a list
    with ``None`` objects except for one position.
    
    Also note that predictor weights are ignored for this decoding
    strategy.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance for system level 
        combination. See the docstring of the BeamDecoder constructor
        for a description of which arguments are fetched from
        `decoder_args`.
        """
        super(SepBeamDecoder, self).__init__(decoder_args)
        if self.hypo_recombination:
            logging.warn("Hypothesis recombination is not applicable "
                         "to the sepbeam decoder.")
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. This is not
        a single empty hypo but one empty hypo for each predictor.
        """
        states = self.get_predictor_states()
        none_states = [None] * len(states)
        ret = []
        for idx, state in enumerate(states):
            pred_states = list(none_states)
            pred_states[idx] = state
            ret.append(PartialHypothesis(pred_states))
        return ret
    
    def _expand_hypo(self, hypo):
        """Expands hypothesis by calling predict_next() only on one
        single predictor.
        """
        if hypo.score <= self.min_score:
            return []
        pred_idx = 0
        for idx, s in enumerate(hypo.predictor_states):
            if not s is None:
                pred_idx = idx
                break
        self.apply_predictors_count += 1
        predictor = self.predictors[pred_idx][0]
        predictor.set_state(copy.deepcopy(hypo.predictor_states[pred_idx]))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            predictor.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        posterior = predictor.predict_next()
        hypo.predictor_states = list(hypo.predictor_states)
        hypo.predictor_states[pred_idx] = predictor.get_state()
        breakdown_dummy = [(0.0, 1.0)] * len(self.predictors)
        ret = []
        for trgt_word in utils.argmax_n(posterior, self.beam_size):
            score_breakdown = list(breakdown_dummy)
            score_breakdown[pred_idx] = (posterior[trgt_word], 1.0)
            ret.append(hypo.cheap_expand(trgt_word,
                                         posterior[trgt_word],
                                         score_breakdown))
        return ret
