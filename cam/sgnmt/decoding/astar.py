"""Implementation of the A* search strategy """


import copy
from heapq import heappush, heappop
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class AstarDecoder(Decoder):
    """This decoder implements A*. For heuristics, see the the 
    ``decoding.core`` module for interfaces and the general handling of
    heuristics, and the ``decoding.heuristics`` package for heuristic
    implementations. This A* implementation does not have a 'closed
    set', i.e. we do not keep track of already visited states. Make 
    sure that your search space is acyclic (normally it is unless you
    decode on cyclic lattices with the fst predictor.
    """
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): Maximum number of active hypotheses.
            pure_heuristic_scores (bool): For standard A* set this to
                                          false. If set to true, partial
                                          hypo scores are ignored when
                                          scoring hypotheses.
            early_stopping (bool): If this is true, partial hypotheses
                                   with score worse than the current
                                   best complete scores are not
                                   expanded. This applies when nbest is
                                   larger than one and inadmissible
                                   heuristics are used
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.

        """
        super(AstarDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.capacity = decoder_args.beam
        self.early_stopping = decoder_args.early_stopping
        self.pure_heuristic_scores = decoder_args.pure_heuristic_scores
    
    def _get_combined_score(self, hypo):
        est_score = -self.estimate_future_cost(hypo)
        if not self.pure_heuristic_scores:
            return est_score + hypo.score
        return est_score

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.initialize_predictors(src_sentence)
        open_set = []
        best_score = self.get_lower_score_bound()
        heappush(open_set, (0.0,
                            PartialHypothesis(self.get_predictor_states())))
        while open_set:
            c,hypo = heappop(open_set)
            if self.early_stopping and hypo.score < best_score:
                continue
            logging.debug("Expand (est=%f score=%f exp=%d best=%f): sentence: %s"
                          % (-c, 
                             hypo.score, 
                             self.apply_predictors_count, 
                             best_score, 
                             hypo.trgt_sentence))
            if hypo.get_last_word() == utils.EOS_ID: # Found best hypothesis
                if hypo.score > best_score:
                    logging.debug("New best hypo (score=%f exp=%d): %s" % (
                            hypo.score,
                            self.apply_predictors_count,
                            ' '.join([str(w) for w in hypo.trgt_sentence])))
                    best_score = hypo.score
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if len(self.full_hypos) >= self.nbest: # if we have enough hypos
                    return self.get_full_hypos_sorted()
                continue
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
            if not hypo.word_to_consume is None: # Consume if cheap expand
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()
            for trgt_word in posterior: # Estimate future cost, add to heap
                next_hypo = hypo.cheap_expand(trgt_word, posterior[trgt_word],
                                                  score_breakdown[trgt_word])
                combined_score = -self.estimate_future_cost(next_hypo)
                if not self.pure_heuristic_scores:
                    combined_score += next_hypo.score
                heappush(open_set, (-self._get_combined_score(next_hypo),
                                    next_hypo))
            # Limit heap capacity
            if self.capacity > 0 and len(open_set) > self.capacity:
                new_open_set = []
                for _ in xrange(self.capacity):
                    heappush(new_open_set, heappop(open_set))
                open_set = new_open_set
        return self.get_full_hypos_sorted()
