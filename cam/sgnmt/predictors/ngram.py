"""This module contains predictors for n-gram (Kneser-Ney) language
modeling. This is a ``UnboundedVocabularyPredictor`` as the vocabulary
size ngram models normally do not permit complete enumeration of the
posterior.

This module is based on the swig-srilm package.

https://github.com/desilinguist/swig-srilm
"""

from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor
from cam.sgnmt import utils
import math

try:
    # Requires swig-srilm
    from srilm import readLM, initLM, getNgramProb, getIndexForWord, howManyNgrams
except ImportError:
    pass # Deal with it in decode.py


class SRILMPredictor(UnboundedVocabularyPredictor):
    """SRILM predictor based on swig 
    https://github.com/desilinguist/swig-srilm
    
    The predictor state is described by the n-gram history. The language
    model has to use word indices rather than the string word 
    representations.
    """
    
    def __init__(self, path, ngram_order, convert_to_ln=False):
        """Creates a new n-gram language model predictor.
        
        Args:
            path (string): Path to the ARPA language model file
            ngram_order (int): Order of the language model
            
        Raises:
            NameError. If srilm-swig is not installed
        """
        super(SRILMPredictor, self).__init__()
        self.history_len = ngram_order-1
        self.lm = initLM(ngram_order)
        readLM(self.lm, path)
        self.vocab_size = howManyNgrams(self.lm, 1)
        self.convert_to_ln = convert_to_ln
        if convert_to_ln:
            import logging
            logging.info("SRILM: Convert log scores to ln scores")
    
    def initialize(self, src_sentence):
        """Initializes the history with the start-of-sentence symbol.
        
        Args:
            src_sentence (list): Not used
        """
        self.history = ['<s>'] if self.history_len > 0 else []
    
    def predict_next(self, words):
        """Score the set of target words with the n-gram language 
        model given the current history
        
        Args:
            words (list): Set of words to score
        Returns:
            dict. Language model scores for the words in ``words``
        """
        prefix = "%s " % ' '.join(self.history)
        order = len(self.history) + 1
        scaling_factor = math.log(10) if self.convert_to_ln else 1.0
        ret = {w: getNgramProb(
                        self.lm,
                        prefix + ("</s>" if w == utils.EOS_ID else str(w)),
                        order) * scaling_factor for w in words}
        return ret
    
        
    def get_unk_probability(self, posterior):
        """Use the probability for '<unk>' in the language model """
        order = len(self.history) + 1
        return getNgramProb(self.lm,
                            "%s <unk>" % ' '.join(self.history),
                            order)
    
    def consume(self, word):
        """Extends the current history by ``word`` """
        if len(self.history) >= self.history_len:
            self.history = self.history[1:]
        self.history.append(str(word))
    
    def get_state(self):
        """Returns the current n-gram history """
        return self.history
    
    def set_state(self, state):
        """Sets the current n-gram history """
        self.history = state

    def reset(self):
        """Resets the current history to <S> """
        self.history = ['<s>'] if self.history_len > 0 else []

    def _replace_unks(self, hist):
        return ['<unk>' if getIndexForWord(w) > self.vocab_size else w for w in hist]
    
    def is_equal(self, state1, state2):
        """Returns true if the ngram history is the same"""
        return self._replace_unks(state1) == self._replace_unks(state2)
    
