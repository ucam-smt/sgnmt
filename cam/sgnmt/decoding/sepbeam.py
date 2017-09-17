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
    
    def __init__(self,
                 decoder_args,
                 hypo_recombination,
                 beam_size,
                 pure_heuristic_scores = False, 
                 diversity_factor = -1.0,
                 early_stopping = True):
        """Creates a new beam decoder instance with explicit
        synchronization symbol.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination 
            beam_size (int): Absolute beam size. A beam of 12 means
                             that we keep track of 12 active hypothesis
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
        super(SepBeamDecoder, self).__init__(decoder_args,
                                              hypo_recombination,
                                              beam_size,
                                              pure_heuristic_scores, 
                                              diversity_factor,
                                              early_stopping)
        if hypo_recombination:
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
