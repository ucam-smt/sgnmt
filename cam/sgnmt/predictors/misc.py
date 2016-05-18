"""This module provides helper predictors and predictor wrappers which
are not directly used for scoring. An example is the indexmap wrapper
predictor, which can be used if a predictor uses a different word map.
"""

from cam.sgnmt.decoding.core import Predictor,UnboundedVocabularyPredictor
from cam.sgnmt import utils
import logging

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
        """ATTENTION: Here we should translate the posterior array 
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
