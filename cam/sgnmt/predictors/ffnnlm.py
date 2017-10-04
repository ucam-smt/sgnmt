"""This module integrates neural language models, for example feed-
forward language models like NPLM. It depends on the Python interface
to NPLM.

http://nlg.isi.edu/software/nplm/
"""

from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor
from cam.sgnmt import utils
import logging

try:
    import nplm
except ImportError:
    pass # Deal with it in decode.py


class NPLMPredictor(UnboundedVocabularyPredictor):
    """NPLM language model predictor. Even though NPLM normally has a
    limited vocabulary size, we implement it as a unbounded vocabulary
    predictor because it is more efficient to score only a subset of
    the vocabulary. This predictor uses the python interface to NPLM
    from
     
    http://nlg.isi.edu/software/nplm/
    """
    
    def __init__(self, path, normalize_scores):
        """Creates a new NPLM predictor instance.
        
        Args:
            path (string): Path to the NPLM model file
            normalize_scores (bool): Whether to renormalize scores s.t.
                                     scores returned by ``predict_next``
                                     sum up to 1
        
        Raises:
            NameError. If NPLM is not installed
        """
        super(NPLMPredictor, self).__init__()
        self.model = nplm.NeuralLM.from_file(path)
        self.normalize_scores = normalize_scores
        ngram_order = self.model.ngram_size
        self.history_len = ngram_order-1
        self.unk_id = self.model.word_to_index['<unk>']
        self.bos_id = self.model.word_to_index['<s>']
        self.eos_id = self.model.word_to_index['</s>']
        if (self.unk_id != utils.UNK_ID 
                or self.bos_id != utils.GO_ID 
                or self.eos_id != utils.EOS_ID):
            logging.error("NPLM reserved word IDs inconsistent with SGNMT")
    
    def initialize(self, src_sentence):
        """Set the n-gram history to initial value.
        
        Args:
            src_sentence (list): Not used
        """
        self.history = [self.bos_id] * self.history_len

    def _get_nplm_idx(self, w):
        """Get word index for internal NPLM word map """
        if w == utils.UNK_ID or w == utils.GO_ID or w == utils.EOS_ID:
            return w
        return self.model.word_to_index.get(str(w), self.unk_id)   
    
    def predict_next(self, words):
        """Scores the words in ``words`` using NPLM. """
        ngram_words = []
        ngrams = []
        for w in words:
            ngram_words.append(w) # cannot use words directly because its a set
            ngrams.append(self.history + [self._get_nplm_idx(w)])
        ngrams = self.model.make_data(ngrams)
        posteriors = self.model.forward_prop(ngrams[:-1],
                                             output=ngrams[-1])[:,0]
        scores = {w: posteriors[idx] for idx,w in enumerate(ngram_words)}
        return self.finalize_posterior(scores, True, self.normalize_scores)
        
    def get_unk_probability(self, posterior):
        """Use NPLM UNK score if exists """
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
    
    def consume(self, word):
        """Extend current history by ``word`` """
        self.history = self.history[1:] + [self._get_nplm_idx(word)]
    
    def get_state(self):
        """Returns the current history """
        return self.history
    
    def set_state(self, state):
        """Sets the current history """
        self.history = state

    def reset(self):
        """Set the n-gram history to initial value."""
        self.history = [self.bos_id] * self.history_len
    
    def is_equal(self, state1, state2):
        """Returns true if the ngram history is the same"""
        return state1 == state2
    
        