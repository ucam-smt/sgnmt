"""This module provides helper predictors and predictor wrappers which
are not directly used for scoring. An example is the indexmap wrapper
predictor, which can be used if a predictor uses a different word map.
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor
from cam.sgnmt.utils import NEG_INF, common_get
import numpy as np


class IdxmapPredictor(Predictor):
    """This wrapper predictor can be applied to slave predictors which
    use different wmaps than SGNMT. It translates between SGNMT word 
    indices and predictors indices each time the predictor is called.
    This mapping is transparent to both the decoder and the wrapped
    slave predictor.
    """
    
    def __init__(self,
                 src_idxmap_path,
                 trgt_idxmap_path,
                 slave_predictor,
                 slave_weight):
        """Creates a new idxmap wrapper predictor. The index maps have
        to be plain text files, each line containing the mapping from
        a SGNMT word index to the slave predictor word index.
        
        Args:
            src_idxmap_path (string): Path to the source index map
            trgt_idxmap_path (string): Path to the target index map
            slave_predictor (Predictor): Instance of the predictor with
                                         a different wmap than SGNMT
            slave_weight (float): Slave predictor weight
        """
        super(IdxmapPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.slave_weight = slave_weight
        # src_map goes from sgnmt index -> slave index for the source 
        # trgt map goes from sgnmt index -> slave index for the target 
        # trgt map_inverse goes from slave index -> sgnmt index for the target 
        self.src_map = self.load_map(src_idxmap_path)
        self.trgt_map = self.load_map(trgt_idxmap_path)
        self.trgt_map_inverse = {slave_idx: gnmt_idx 
                        for gnmt_idx, slave_idx in enumerate(self.trgt_map)}
    
    def load_map(self, path):
        """Load a index map file. Mappings should be bijections, but
        there is no sanity check in place to verify this.
        
        Args:
            path (string): Path to the mapping file
        
        Returns:
            dict. Mapping from SGNMT index to slave predictor index
        """
        with open(path) as f:
            d = dict(map(int, line.strip().split(None, 1)) for line in f)
            if (d[utils.UNK_ID] != utils.UNK_ID
                    or d[utils.EOS_ID] != utils.EOS_ID
                    or d[utils.GO_ID] != utils.GO_ID):
                logging.fatal(
                   "idxmap %s contains non-identical maps for reserved indices"
                        % path)
            logging.debug("Loaded wmap from %s" % path)
            return [d[idx] if idx in d else 0 for idx in range(max(d)+1)]
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize([self.src_map[idx]
                                            for idx in src_sentence])
    
    def predict_next(self):
        """Pass through to slave predictor """
        posterior = self.slave_predictor.predict_next()
        return {self.trgt_map_inverse.get(idx, utils.UNK_ID): self.slave_weight * prob 
            for idx, prob in utils.common_iterable(posterior)}
        
    def get_unk_probability(self, posterior):
        """ATTENTION: We should translate the posterior array 
        back to slave predictor indices. However, the unk_id is 
        translated to the identical index, and others normally do not
        matter when computing the UNK probability. Therefore, we 
        refrain from a complete conversion and pass through
        ``posterior`` without changing its word indices.
        """
        return self.slave_predictor.get_unk_probability(posterior)
    
    def consume(self, word):
        """Pass through to slave predictor """
        self.slave_predictor.consume(self.trgt_map[word])
    
    def get_state(self):
        """Pass through to slave predictor """
        return self.slave_predictor.get_state()
    
    def set_state(self, state):
        """Pass through to slave predictor """
        self.slave_predictor.set_state(state)

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

    def estimate_future_cost(self, hypo):
        """Pass through to slave predictor """
        old_sen = hypo.trgt_sentence
        hypo.trgt_sentence = [self.trgt_map[idx] for idx in old_sen]
        ret = self.slave_predictor.estimate_future_cost(hypo)
        hypo.trgt_sentence = old_sen
        return ret

    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize_heuristic([self.src_map[idx] 
                                                    for idx in src_sentence])

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(IdxmapPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)
        

class UnboundedIdxmapPredictor(IdxmapPredictor,UnboundedVocabularyPredictor):
    """This class is a version of ``IdxmapPredictor`` for unbounded 
    vocabulary predictors. This needs an adjusted ``predict_next`` 
    method to pass through the set of target words to score correctly.
    """
    
    def __init__(self,
                 src_idxmap_path,
                 trgt_idxmap_path,
                 slave_predictor,
                 slave_weight):
        """Pass through to ``IdxmapPredictor.__init__`` """
        super(UnboundedIdxmapPredictor, self).__init__(src_idxmap_path,
                                                       trgt_idxmap_path,
                                                       slave_predictor,
                                                       slave_weight)

    def predict_next(self, trgt_words):
        """Pass through to slave predictor """
        posterior = self.slave_predictor.predict_next([self.trgt_map[w] 
                                                       for w in trgt_words])
        return {self.trgt_map_inverse.get(idx,
                                          utils.UNK_ID): self.slave_weight*prob 
                            for idx, prob in utils.common_iterable(posterior)}


class AltsrcPredictor(Predictor):
    """This wrapper loads the source sentences from an alternative 
    source file. The ``src_sentence`` arguments of ``initialize`` and
    ``initialize_heuristic`` are overridden with sentences loaded from
    the file specified via the argument ``--altsrc_test``. All other
    methods are pass through calls to the slave predictor.
    """
    
    def __init__(self, src_test, slave_predictor):
        """Creates a new idxmap wrapper predictor. The index maps have
        to be plain text files, each line containing the mapping from
        a SGNMT word index to the slave predictor word index.
        
        Args:
            src_idxmap_path (string): Path to the source index map
            trgt_idxmap_path (string): Path to the target index map
            slave_predictor (Predictor): Instance of the predictor with
                                         a different wmap than SGNMT
            slave_weight (float): Slave predictor weight
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

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

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
        

class UnboundedAltsrcPredictor(AltsrcPredictor,UnboundedVocabularyPredictor):
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


class UnkCountPredictor(Predictor):
    """This predictor regulates the number of UNKs in the output. We 
    assume that the number of UNKs in the target sentence is Poisson 
    distributed. This predictor is configured with n lambdas for
    0,1,...,>=n-1 UNKs in the source sentence. """
    
    def __init__(self, src_vocab_size, lambdas):
        """Initializes the UNK count predictor.

        Args:
            src_vocab_size (int): Size of source language vocabulary.
                                  Indices greater than this are 
                                  considered as UNK.
            lambdas (list): List of floats. The first entry is the 
                            lambda parameter given that the number of
                            unks in the source sentence is 0 etc. The
                            last float is lambda given that the source
                            sentence has more than n-1 unks.
        """
        self.lambdas = lambdas
        self.l = lambdas[0]
        self.src_vocab_size = src_vocab_size
        super(UnkCountPredictor, self).__init__()
        
    def get_unk_probability(self, posterior):
        """Always returns 0 (= log 1) except for the first time """
        if self.n_consumed == 0:
            return self.max_prob
        return 0.0
    
    def predict_next(self):
        """Set score for EOS to the number of consumed words """
        if self.n_consumed == 0:
            return {utils.EOS_ID : self.unk_prob}
        if self.n_unk < self.max_prob_idx:
            return {utils.EOS_ID : self.unk_prob - self.max_prob}
        return {utils.UNK_ID : self.unk_prob - self.consumed_prob}
    
    def initialize(self, src_sentence):
        """Count UNKs in ``src_sentence`` and reset counters.
        
        Args:
            src_sentence (list): Count UNKs in this list
        """
        src_n_unk = len([w for w in src_sentence if w == utils.UNK_ID 
                                                    or w > self.src_vocab_size])
        self.l = self.lambdas[min(len(self.lambdas)-1, src_n_unk)]
        self.n_consumed = 0
        self.n_unk = 0
        self.unk_prob = self._get_poisson_prob(1)
        # Mode at lambda is the maximum of the poisson function
        self.max_prob_idx = int(self.l)
        self.max_prob = self._get_poisson_prob(self.max_prob_idx)
        ceil_prob = self._get_poisson_prob(self.max_prob_idx + 1)
        if ceil_prob > self.max_prob:
            self.max_prob = ceil_prob
            self.max_prob_idx = self.max_prob_idx + 1
        self.consumed_prob = self.max_prob

    def _get_poisson_prob(self, n):
        """Get the log of the poisson probability for n events. """
        return n * np.log(self.l) - self.l - sum([np.log(i+1) for i in xrange(n)])
    
    def consume(self, word):
        """Increases unk counter by one if ``word`` is unk.
        
        Args:
            word (int): Increase counter if ``word`` is UNK
        """
        self.n_consumed += 1
        if word == utils.UNK_ID:
            if self.n_unk >= self.max_prob_idx:
                self.consumed_prob = self.unk_prob
            self.n_unk += 1
            self.unk_prob = self._get_poisson_prob(self.n_unk+1)
    
    def get_state(self):
        """Returns the number of consumed words """
        return self.n_unk,self.n_consumed,self.unk_prob,self.consumed_prob
    
    def set_state(self, state):
        """Set the number of consumed words """
        self.n_unk,self.n_consumed,self.unk_prob,self.consumed_prob = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the state is the same"""
        return state1 == state2
    