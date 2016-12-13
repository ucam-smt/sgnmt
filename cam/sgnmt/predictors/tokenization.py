"""This module contains wrapper predictors which support decoding with
diverse tokenization. The ``Word2charPredictor`` can be used if the 
decoder operates on fine-grained tokens such as characters, but the
tokenization of a predictor is coarse-grained (e.g. words or subwords).

The ``word2char`` predictor maintains an explicit list of word boundary
characters and applies consume and predict_next whenever a word boundary
character is consumed.

The ``char2word`` predictor masks character-based predictors and 
presents word level posterior distributions to the decoder. In order to
form these posteriors, ``char2word`` internally searches until a list
of full words is found. Note that since char2word can produce words
without word ID, it maintains an internal word map which is updated
accordingly. Therefore, the ``output_chars`` option should be used to
create output files on the character level, otherwise the new word IDs
cannot be mapped back to words.
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor, Predictor
from cam.sgnmt.utils import NEG_INF, common_get


class Char2wordPredictor(Predictor):
    """This wrapper can be used if the SGNMT decoder operates on the
    word level, but a predictor uses a finer grained tokenization. The
    wrapper internally implements a search for full words with the
    character-based model (predictor) and presents the found full word
    continuations to the decoder. If the wrapped predictor produces a
    word which has not have an assigned word ID, it updates
    ``Char2wordPredictor.wmap`` with a new word. This word map is taken
    into account when the SGNMT option ``output_chars`` is activated.
    
    TODO Implement
    """
    
    def __init__(self, src_test, slave_predictor):
        """TODO
        
        Args:
            TODO
        """
        super(Char2wordPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        if isinstance(slave_predictor, UnboundedVocabularyPredictor):
            logging.fatal("char2word cannot wrap an unbounded "
                          "vocabulary predictor.")
        raise NotImplementedError
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize(src_sentence)
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize_heuristic(src_sentence)
    
    def predict_next(self):
        pass
        
    def get_unk_probability(self, posterior):
        pass
    
    def consume(self, word):
        pass
    
    def get_state(self):
        pass
    
    def set_state(self, state):
        pass

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

    def estimate_future_cost(self, hypo):
        pass

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(Char2wordPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        pass
    

class Word2charPredictor(UnboundedVocabularyPredictor):
    """This predictor wraps word level predictors when SGNMT is running
    on the character level. The mapping between word ID and character 
    ID sequence is loaded from the file system. All characters which
    do not appear in that mapping are treated as word boundary
    makers. The wrapper blocks consume and predict_next calls until a
    word boundary marker is consumed, and updates the slave predictor
    according the word between the last two word boundaries. The 
    mapping is done only on the target side, and the source sentences
    are passed through as they are. To use alternative tokenization on
    the source side, see the altsrc predictor wrapper. The word2char
    wrapper is always an ``UnboundedVocabularyPredictor``.
    """
    
    def __init__(self, map_path, slave_predictor):
        """Creates a new word2char wrapper predictor. The map_path 
        file has to be plain text files, each line containing the 
        mapping from a word index to the character index sequence
        (format: word char1 char2... charn).
        
        Args:
            map_path (string): Path to the mapping file
            slave_predictor (Predictor): Instance of the predictor with
                                         a different wmap than SGNMT
        """
        super(Word2charPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.words = SimpleTrie()
        self.word_chars = {}
        with open(map_path) as f:
            for line in f:
                l = [int(x) for x in line.strip().split()]
                word = l[0]
                chars = l[1:]
                self.words.add(chars, word)
                for c in chars:
                    self.word_chars[c] = True   
        if isinstance(slave_predictor, UnboundedVocabularyPredictor): 
            self._get_stub_prob = self._get_stub_prob_unbounded
            self._start_new_word = self._start_new_word_unbounded
        else:
            self._get_stub_prob = self._get_stub_prob_bounded
            self._start_new_word = self._start_new_word_bounded             
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize(src_sentence)
        self._start_new_word()
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize_heuristic(src_sentence)
    
    def _update_slave_vars(self, posterior):
        self.slave_unk = self.slave_predictor.get_unk_probability(posterior)
        self.slave_go = common_get(posterior, utils.GO_ID, self.slave_unk)
        self.slave_eos = common_get(posterior, utils.EOS_ID, self.slave_unk)
        
    def _start_new_word_unbounded(self):
        """start_new_word implementation for unbounded vocabulary slave
        predictors. Needs to set slave_go, slave_eos, and slave_unk
        """
        self.word_stub = []
        posterior = self.slave_predictor.predict_next([utils.UNK_ID,
                                                       utils.GO_ID,
                                                       utils.EOS_ID])
        self._update_slave_vars(posterior)
    
    def _start_new_word_bounded(self):
        """start_new_word implementation for bounded vocabulary slave
        predictors. Needs to set slave_go, slave_eos, slave_unk, and
        slave_posterior
        """
        self.word_stub = []
        self.slave_posterior = self.slave_predictor.predict_next()
        self._update_slave_vars(self.slave_posterior)
    
    def _get_stub_prob_unbounded(self):
        """get_stub_prob implementation for unbounded vocabulary slave
        predictors.
        """
        word = self.words.get(self.word_stub)
        if word:
            posterior = self.slave_predictor.predict_next([word])
            return common_get(posterior, word, self.slave_unk)
        return self.slave_unk
    
    def _get_stub_prob_bounded(self):
        """get_stub_prob implementation for bounded vocabulary slave
        predictors.
        """
        word = self.words.get(self.word_stub)
        return common_get(self.slave_posterior,
                          word if word else utils.UNK_ID,
                          self.slave_unk)
    
    def predict_next(self, trgt_words):
        posterior = {}
        stub_prob = False
        for ch in trgt_words:
            if ch in self.word_chars:
                posterior[ch] = 0.0
            else: # Word boundary marker
                if stub_prob is False:
                    stub_prob = self._get_stub_prob() if self.word_stub else 0.0
                posterior[ch] = stub_prob
        if utils.GO_ID in posterior:
            posterior[utils.GO_ID] += self.slave_go
        if utils.EOS_ID in posterior:
            posterior[utils.EOS_ID] += self.slave_eos
        return posterior
        
    def get_unk_probability(self, posterior):
        """This is about the unkown character, not word. Since the word
        level slave predictor has no notion of the unknown character, 
        we return NEG_INF unconditionally.
        """
        return NEG_INF
    
    def consume(self, word):
        """If ``word`` is a word boundary marker, truncate ``word_stub``
        and let the slave predictor consume word_stub. Otherwise, 
        extend ``word_stub`` by the character.
        """
        if word in self.word_chars:
            self.word_stub.append(word)
        elif self.word_stub:
            word = self.words.get(self.word_stub)
            self.slave_predictor.consume(word if word else utils.UNK_ID)
            self._start_new_word()
    
    def get_state(self):
        """Pass through to slave predictor """
        return self.word_stub, self.slave_predictor.get_state()
    
    def set_state(self, state):
        """Pass through to slave predictor """
        self.word_stub, slave_state = state
        self.slave_predictor.set_state(slave_state)

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

    def estimate_future_cost(self, hypo):
        """Not supported """
        logging.warn("Cannot use future cost estimates of predictors "
                     "wrapped by word2char")
        return 0.0

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(Word2charPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        stub1, slave_state1 = state1
        stub2, slave_state2 = state2
        return (stub1 == stub2 
                and self.slave_predictor.is_equal(slave_state1, slave_state2))

