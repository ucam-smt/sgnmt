"""This module provides helper predictors and predictor wrappers which
are not directly used for scoring. An example is the altsrc predictor 
wrapper which loads source sentences from a different file.
"""

import operator
import numpy as np
import logging

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


class GluePredictor(Predictor):
    """This wrapper masks sentence-level predictors when SGNMT runs on
    the document level. The SGNMT hypotheses consist of multiple
    sentences, glued together with <s>, but the wrapped predictor is
    trained on the sentence level. This predictor splits input
    sequences at <s> and feed them to the predictor one by one. The
    wrapped predictor is initialized with a new source sentence when
    the sentence boundary symbol <s> is emitted. Note that using the
    predictor heuristic of the wrapped predictor estimates the future
    cost for the current sentence, not for the entire document.
    """
    
    def __init__(self, max_len_factor, slave_predictor):
        """Creates a new glue wrapper predictor.
        
        Args:
            max_len_factor (int): Target sentences cannot be longer
                                  than this times source sentence length
            slave_predictor (Predictor): Instance of the sentence-level
                                         predictor.
        """
        super(GluePredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.max_len_factor = max_len_factor
    
    def initialize(self, src_sentence):
        """Splits ``src_sentence`` at ``utils.GO_ID``, stores all
        segments for later use, and calls ``initialize()`` of the
        slave predictor with the first segment.
        """
        self._src_sentences = []
        last_pos = 0
        for pos, word in enumerate(src_sentence):
            if word == utils.GO_ID:
                self._src_sentences.append(src_sentence[last_pos:pos])
                last_pos = pos + 1
        self._src_sentences.append(src_sentence[last_pos:])
        self._next_src_sentence_idx = 0
        self._next_src_sentence()

    def _next_src_sentence(self):
        src = self._src_sentences[self._next_src_sentence_idx]
        self.slave_predictor.initialize(src)
        self._sen_budget = self.max_len_factor * len(src)
        self._next_src_sentence_idx += 1

    def _sen_len_exceeded(self):
        return self._sen_budget < 0

    def is_last_sentence(self):
        """Returns True if the current sentence is the last sentence
        in this document - i.e. we have already consumed n-1 <s>
        symbols since the last call of ``initialize()``.
        """
        return self._next_src_sentence_idx >= len(self._src_sentences)
    
    def predict_next(self):
        """Calls predict_next() of the wrapped predictor. Replaces BOS
        scores with EOS score if we still have source sentences left.
        """
        if self._sen_len_exceeded():
            end_token = utils.EOS_ID if self.is_last_sentence() else utils.GO_ID
            logging.info("Sentence length exceeded!")
            return {end_token: 0.0}
        slave_scores = self.slave_predictor.predict_next()
        if not self.is_last_sentence():
            slave_scores[utils.GO_ID] = slave_scores[utils.EOS_ID]
            slave_scores[utils.EOS_ID] = utils.NEG_INF
        else:
            slave_scores[utils.GO_ID] = utils.NEG_INF
        return slave_scores
        
    def consume(self, word):
        """If ``word`` is <s>, initialize the slave predictor with the
        next source sentence. Otherwise, pass through ``word`` to the
        ``consume()`` method of the slave.
        """
        if word == utils.GO_ID:
            self._next_src_sentence()
        else:
            self.slave_predictor.consume(word)
            self._sen_budget -= 1
    
    def get_state(self):
        """State is the slave state plus the source sentence index."""
        return (self._next_src_sentence_idx,
                self._sen_budget,
                self.slave_predictor.get_state())
    
    def set_state(self, state):
        """State is the slave state plus the source sentence index."""
        self._next_src_sentence_idx, self._sen_budget, slave_state = state
        self.slave_predictor.set_state(slave_state)

    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor."""
        self.slave_predictor.initialize_heuristic(self._get_current_sentence())

    def get_unk_probability(self, posterior):
        """Pass through to slave predictor """
        if self._sen_len_exceeded():
            return utils.NEG_INF
        return self.slave_predictor.get_unk_probability(posterior)

    def estimate_future_cost(self, hypo):
        """Pass through to slave predictor """
        return self.slave_predictor.estimate_future_cost(hypo)

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(GluePredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)
        

class UnboundedGluePredictor(GluePredictor, UnboundedVocabularyPredictor):
    """This class is a version of ``GluePredictor`` for unbounded 
    vocabulary predictors.
    """

    def predict_next(self, trgt_words):
        """Pass through to slave predictor """
        return self.slave_predictor.predict_next(trgt_words)
