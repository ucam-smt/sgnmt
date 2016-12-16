"""This module provides helper predictors and predictor wrappers which
are not directly used for scoring. An example is the indexmap wrapper
predictor, which can be used if a predictor uses a different word map.
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor
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


class UnkvocabPredictor(Predictor):
    """If the predictor wrapped by the unkvocab wrapper produces an UNK
    with predict next, this wrapper adds explicit NEG_INF scores to all
    in-vocabulary words not in its posterior. This can control which 
    words are matched by the UNK scores of other predictors.
    """
    
    def __init__(self, trg_vocab_size, slave_predictor):
        """Creates a new unkvocab wrapper predictor.
        
        Args:
            trg_vocab_size (int): Size of the target vocabulary
        """
        super(UnkvocabPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.trg_vocab_size = trg_vocab_size
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize(src_sentence)
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize_heuristic(src_sentence)
 
    def predict_next(self):
        """Pass through to slave predictor. If the posterior from the
        slave predictor contains util.UNK_ID, add NEG_INF for all 
        word ids lower than trg_vocab_size that are not already
        defined """
        posterior = self.slave_predictor.predict_next()
        if utils.UNK_ID in posterior:
            for w in xrange(self.trg_vocab_size):
                if not w in posterior:
                    posterior[w] = utils.NEG_INF
        return posterior
        
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
        super(UnkvocabPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)


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
    
