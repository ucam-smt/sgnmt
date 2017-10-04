"""Predictor wrappers in this module work with the vocabulary of the
wrapped predictor. An example is the idxmap wrapper which makes it
possible to use an alternative word map.
"""

import logging
import copy

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor


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


class SkipvocabInternalHypothesis(object):
    """Helper class for internal beam search in skipvocab."""

    def __init__(self, score, predictor_state, word_to_consume):
        self.score = score
        self.predictor_state = predictor_state
        self.word_to_consume = word_to_consume

    
class SkipvocabPredictor(Predictor):
    """This predictor wrapper masks predictors with a larger vocabulary
    than the SGNMT vocabulary. The SGNMT OOV words are not scored with 
    UNK scores from the other predictors like usual, but are hidden by 
    this wrapper. Therefore, this wrapper does not produce any word
    from the larger vocabulary, but searches internally until enough
    in-vocabulary word scores are collected from the wrapped predictor.
    """
    
    def __init__(self, max_id, stop_size, beam, slave_predictor):
        """Creates a new skipvocab wrapper predictor.
        
        Args:
            max_id (int): All words greater than this are skipped
            stop_size (int): Stop internal beam search when the best
                             stop_size words are in-vocabulary
            beam (int): Beam size of internal beam search
            slave_predictor (Predictor): Wrapped predictor.
        """
        super(SkipvocabPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.max_id = max_id
        self.stop_size = stop_size
        self.beam = beam
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize(src_sentence)
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor """
        self.slave_predictor.initialize_heuristic(src_sentence)

    def get_unk_probability(self, posterior):
        """Pass through to slave predictor """
        return self.slave_predictor.get_unk_probability(posterior)

    def _is_stopping_posterior(self, posterior):
        for word, _ in sorted(utils.common_iterable(posterior),
                              key=lambda h: -h[1])[:self.stop_size]:
            if word > self.max_id:
                return False
        return True
 
    def predict_next(self):
        """This method first performs beam search internally to update
        the slave predictor state to a point where the best stop_size 
        entries in the predict_next() return value are in-vocabulary
        (bounded by max_id). Then, it returns the slave posterior in 
        that state.
        """
        hypos = [SkipvocabInternalHypothesis(0.0, 
                                             self.slave_predictor.get_state(),
                                             None)]
        best_score = utils.NEG_INF
        best_predictor_state = None
        best_posterior = None
        while hypos and hypos[0].score > best_score:
            next_hypos = []
            for hypo in hypos:
                self.slave_predictor.set_state(copy.deepcopy(
                    hypo.predictor_state))
                if hypo.word_to_consume is not None:
                    self.slave_predictor.consume(hypo.word_to_consume)
                posterior = self.slave_predictor.predict_next()
                pred_state = copy.deepcopy(self.slave_predictor.get_state())
                if (self._is_stopping_posterior(posterior) 
                        and hypo.score > best_score):
                    # This is the new best result of the internal beam search
                    best_score = hypo.score
                    best_predictor_state = pred_state
                    best_posterior = posterior
                else:
                    # Look for ways to expand this hypo with OOV words.
                    for word, score in utils.common_iterable(posterior):
                        if word > self.max_id:
                            next_hypos.append(SkipvocabInternalHypothesis(
                                hypo.score + score, pred_state, word))
            next_hypos.sort(key=lambda h: -h.score)
            hypos = next_hypos[:self.beam]
        self.slave_predictor.set_state(copy.deepcopy(best_predictor_state))
        return best_posterior
        
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
        super(SkipvocabPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        return self.slave_predictor.is_equal(state1, state2)

