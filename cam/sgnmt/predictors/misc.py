"""This module provides helper predictors and predictor wrappers which
are not directly used for scoring. An example is the altsrc predictor 
wrapper which loads source sentences from a different file.
"""

import operator
import numpy as np

from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor
from cam.sgnmt import utils


class AltsrcPredictor(Predictor):
    """This wrapper loads the source sentences from an alternative 
    source file. The ``src_sentence`` arguments of ``initialize`` and
    ``initialize_heuristic`` are overridden with sentences loaded from
    the file specified via the argument ``--altsrc_test``. All other
    methods are pass through calls to the slave predictor.
    """
    
    def __init__(self, src_test, slave_predictor):
        """Creates a new altsrc wrapper predictor.
        
        Args:
            src_test (string): Path to the text file with source
                               sentences
            slave_predictor (Predictor): Instance of the predictor which
                                         uses the source sentences in
                                         ``src_test``
        """
        super(AltsrcPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.altsens = []
        with open(src_test) as f:
            for line in f:
                self.altsens.append([int(x) for x in line.strip().split()])
    
    def _get_current_sentence(self):
        return self.altsens[self.current_sen_id]
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor but replace 
        ``src_sentence`` with a sentence from ``self.altsens``
        """
        self.slave_predictor.initialize(self._get_current_sentence())
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor but replace 
        ``src_sentence`` with a sentence from ``self.altsens``
        """
        self.slave_predictor.initialize_heuristic(self._get_current_sentence())
    
    def predict_next(self):
        """Pass through to slave predictor """
        return self.slave_predictor.predict_next()
        
    def get_unk_probability(self, posterior):
        """Pass through to slave predictor """
        return self.slave_predictor.get_unk_probability(posterior)
    
    def consume(self, word):
        """Pass through to slave predictor """
        self.slave_predictor.consume(word)
    
    def get_state(self):
        """Pass through to slave predictor """
        return self.slave_predictor.get_state()
    
    def set_state(self, state):
        """Pass through to slave predictor """
        self.slave_predictor.set_state(state)

    def estimate_future_cost(self, hypo):
        """Pass through to slave predictor """
        return self.slave_predictor.estimate_future_cost(hypo)

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(AltsrcPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)
        

class UnboundedAltsrcPredictor(AltsrcPredictor, UnboundedVocabularyPredictor):
    """This class is a version of ``AltsrcPredictor`` for unbounded 
    vocabulary predictors. This needs an adjusted ``predict_next`` 
    method to pass through the set of target words to score correctly.
    """
    
    def __init__(self, src_test, slave_predictor):
        """Pass through to ``AltsrcPredictor.__init__`` """
        super(UnboundedAltsrcPredictor, self).__init__(src_test,
                                                       slave_predictor)

    def predict_next(self, trgt_words):
        """Pass through to slave predictor """
        return self.slave_predictor.predict_next(trgt_words)



class RankPredictor(Predictor):
    """This wrapper converts predictor scores to (negative) ranks, i.e.
    the best word gets a score of -1, the second best of -2 and so on.

    Note: Using this predictor with UNK matching or predictor heuristics
    is not recommended.
    """
    
    def __init__(self, slave_predictor):
        """Creates a new rank wrapper predictor.
        
        Args:
            slave_predictor (Predictor): Use score of this predictor to
                                         compute ranks.
        """
        super(RankPredictor, self).__init__()
        self.slave_predictor = slave_predictor
    
    def score2rank(self, scores):
        if isinstance(scores, dict):
            l = list(utils.common_iterable(scores))
            l.sort(key=operator.itemgetter(1), reverse=True)
            return {el[0]: -(r+1) for r, el in enumerate(l)}
        # scores is a list or numpy array.
        indices = np.argsort(-scores)
        ranks = np.empty_like(scores)
        ranks[indices] =  np.arange(-1, -len(scores)-1, -1, dtype=scores.dtype)
        return ranks
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor. """
        self.slave_predictor.initialize(src_sentence)
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor. """
        self.slave_predictor.initialize_heuristic(src_sentence)
    
    def predict_next(self):
        """Pass through to slave predictor """
        return self.score2rank(self.slave_predictor.predict_next())
        
    def get_unk_probability(self, posterior):
        """Pass through to slave predictor """
        return self.slave_predictor.get_unk_probability(posterior)
    
    def consume(self, word):
        """Pass through to slave predictor """
        self.slave_predictor.consume(word)
    
    def get_state(self):
        """Pass through to slave predictor """
        return self.slave_predictor.get_state()
    
    def set_state(self, state):
        """Pass through to slave predictor """
        self.slave_predictor.set_state(state)

    def estimate_future_cost(self, hypo):
        """Pass through to slave predictor """
        return self.slave_predictor.estimate_future_cost(hypo)

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(RankPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)
        

class UnboundedRankPredictor(RankPredictor, UnboundedVocabularyPredictor):
    """This class is a version of ``RankPredictor`` for unbounded 
    vocabulary predictors. This needs an adjusted ``predict_next`` 
    method to pass through the set of target words to score correctly.
    """
    
    def predict_next(self, trgt_words):
        """Return ranks instead of slave scores """
        return self.score2rank(self.slave_predictor.predict_next(trgt_words))


