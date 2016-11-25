"""Heuristics are used during A* decoding and are called to compose the
estimated look ahead costs. The ``Heuristic`` super class is defined
in the ``core`` module. 
"""

import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Heuristic, Decoder
from cam.sgnmt.decoding.greedy import GreedyDecoder
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.misc.unigram import FileUnigramTable, BestStatsUnigramTable, \
    FullStatsUnigramTable, AllStatsUnigramTable
from cam.sgnmt.utils import MESSAGE_TYPE_DEFAULT


class PredictorHeuristic(Heuristic):
    """The predictor heuristic relies on the 
    ``estimate_future_costs()`` implementation of the predictors. Use
    this heuristic to access predictor specific future cost functions,
    e.g. shortest path for the fst predictor.
    """
    
    def estimate_future_cost(self, hypo):
        """Returns the weighted sum of predictor estimates. """
        return Decoder.combi_arithmetic_unnormalized([
                                    (pred.estimate_future_cost(hypo), w)
                                            for (pred, w) in self.predictors])
    
    def initialize(self, src_sentence):
        """Calls ``initialize_heuristic()`` on all predictors. """
        for (pred, _) in self.predictors:
            pred.initialize_heuristic(src_sentence)
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """This heuristic passes through notifications to the 
        predictors.
        """
        for (pred, _) in self.predictors:
            pred.notify(message, message_type)


class ScorePerWordHeuristic(Heuristic):
    """Using this heuristic results in length normalized scores instead
    of the pure sum of predictor scores for a partial hypothesis.
    Therefore, it is not a heuristic like in the classical A* sense.
    Instead, using the A* decoder with this heuristic simulates beam
    search which always keeps the hypotheses with the best per word
    scores.
    """
    
    def estimate_future_cost(self, hypo):
        """A* will put ``cost-score`` on the heap. In order to simulate
        length normalized beam search, we want to use ``-score/length``
        as partial hypothesis score. Therefore, this method returns
        ``-score/length + score``
        """
        if len(hypo.trgt_sentence) > 0:
            return hypo.score - hypo.score/len(hypo.trgt_sentence)
        return 0.0
    
    def initialize(self, src_sentence):
        """Empty method."""
        pass


class GreedyHeuristic(Heuristic):
    """This heuristic performs greedy decoding to get future cost 
    estimates. This is expensive but can lead to very close estimates.
    """
    
    def __init__(self, decoder_args, cache_estimates = True):
        """Creates a new ``GreedyHeuristic`` instance. The greedy 
        heuristic performs full greedy decoding from the current
        state to get accurate cost estimates. However, this can be very
        expensive.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            cache_estimates (bool): Set to true to enable a cache for
                                    predictor states which have been
                                    visited during the greedy decoding.
        """
        super(GreedyHeuristic, self).__init__()
        self.cache_estimates = cache_estimates
        self.decoder = GreedyDecoder(decoder_args)
        self.cache = SimpleTrie()
        
    def set_predictors(self, predictors):
        """Override ``Decoder.set_predictors`` to redirect the 
        predictors to ``self.decoder``
        """
        self.predictors = predictors
        self.decoder.predictors = predictors
    
    def initialize(self, src_sentence):
        """Initialize the cache. """
        self.cache = SimpleTrie()
    
    def estimate_future_cost(self, hypo):
        """Estimate the future cost by full greedy decoding. If
        ``self.cache_estimates`` is enabled, check cache first
        """
        if self.cache_estimates:
            return self.estimate_future_cost_with_cache(hypo)
        else:
            return self.estimate_future_cost_without_cache(hypo)
    
    def estimate_future_cost_with_cache(self, hypo):
        """Enabled cache... """
        cached_cost = self.cache.get(hypo.trgt_sentence)
        if not cached_cost is None:
            return cached_cost
        old_states = self.decoder.get_predictor_states()
        self.decoder.set_predictor_states(copy.deepcopy(old_states))
        # Greedy decoding
        trgt_word = hypo.trgt_sentence[-1]
        scores = []
        words = []
        while trgt_word != utils.EOS_ID:
            self.decoder.consume(trgt_word)
            posterior,_ = self.decoder.apply_predictors()
            trgt_word = utils.argmax(posterior)
            scores.append(posterior[trgt_word])
            words.append(trgt_word)
        # Update cache using scores and words
        for i in xrange(1,len(scores)):
            self.cache.add(hypo.trgt_sentence + words[:i], -sum(scores[i:]))
        # Reset predictor states
        self.decoder.set_predictor_states(old_states)
        return -sum(scores)
    
    def estimate_future_cost_without_cache(self, hypo):
        """Disabled cache... """
        old_states = self.decoder.get_predictor_states()
        self.decoder.set_predictor_states(copy.deepcopy(old_states))
        # Greedy decoding
        trgt_word = hypo.trgt_sentence[-1]
        score = 0.0
        while trgt_word != utils.EOS_ID:
            self.decoder.consume(trgt_word)
            posterior,_ = self.decoder.apply_predictors()
            trgt_word = utils.argmax(posterior)
            score += posterior[trgt_word]
        # Reset predictor states
        self.decoder.set_predictor_states(old_states)
        return -score


class StatsHeuristic(Heuristic):
    """This heuristic is based on the sum of unigram costs of consumed 
    words. Unigram statistics are collected via a ``UnigramTable``.
    """
    
    def __init__(self, heuristic_scores_file="", collect_stats_strategy='best'):
        """Creates a new ``StatsHeuristic`` instance. The constructor
        initializes the unigram table.
        
        Args:
            heuristic_scores_file (string): Path to the unigram scores 
                                            which are used if this 
                                            predictor estimates future
                                            costs
            collect_stats_strategy (string): best, full, or all. Defines 
                                             how unigram estimates are 
                                             collected for heuristic
        """
        super(StatsHeuristic, self).__init__()
        if heuristic_scores_file:
            self.estimates = FileUnigramTable(heuristic_scores_file)
        elif collect_stats_strategy == 'best':
            self.estimates = BestStatsUnigramTable()
        elif collect_stats_strategy == 'full':
            self.estimates = FullStatsUnigramTable()
        elif collect_stats_strategy == 'all':
            self.estimates = AllStatsUnigramTable()
        else:
            logging.error("Unknown statistics collection strategy")
    
    def initialize(self, src_sentence):
        """Calls ``reset`` to reset collected statistics from previous
        sentences
        
        Args:
            src_sentence (list): Not used
        """
        self.estimates.reset()
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Passing through to the unigram table ``self.estimates``.
        """
        self.estimates.notify(message, message_type)
    
    def estimate_future_cost(self, hypo):
        """Returns the sum of heuristic unigram estimates of the words
        in the translation prefix of ``hypo``. Combined with the hypo
        score, this leads to using the ratio between actual hypo score 
        and an idealistic score (product of unigrams) to discriminate
        partial hypotheses.
        """
        return sum([self.estimates.estimate(w, -1000.0)
                            for w in hypo.trgt_sentence])
