"""This module contains predictors for n-gram (Kneser-Ney) language
modeling. This is a ``UnboundedVocabularyPredictor`` as the vocabulary
size ngram models normally do not permit complete enumeration of the
posterior.
"""

from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor
from cam.sgnmt import utils
import math

try:
    # Requires kenlm
    import kenlm
except ImportError:
    pass # Deal with it in decode.py


class KenLMPredictor(UnboundedVocabularyPredictor):
    """KenLM predictor based on
    https://github.com/kpu/kenlm 
    
    The predictor state is described by the n-gram history.
    """
    
    def __init__(self, path):
        """Creates a new n-gram language model predictor.
        
        Args:
            path (string): Path to the ARPA language model file
            
        Raises:
            NameError. If KenLM is not installed
        """
        super(KenLMPredictor, self).__init__()
        self.lm = kenlm.Model(path)
        self.lm_state2 = kenlm.State()
    
    def initialize(self, src_sentence):
        """Initializes the KenLM state.
        
        Args:
            src_sentence (list): Not used
        """
        self.history = []
        self._update_lm_state()

    def _update_lm_state(self):
        self.lm_state = kenlm.State()
        tmp_state = kenlm.State()
        self.lm.BeginSentenceWrite(self.lm_state)
        for w in self.history[-6:]:
            self.lm.BaseScore(self.lm_state, w, tmp_state)
            self.lm_state, tmp_state = tmp_state, self.lm_state
    
    def predict_next(self, words):
        return {w: self.lm.BaseScore(self.lm_state, 
                                     "</s>" if w == utils.EOS_ID else str(w),
                                     self.lm_state2)
                for w in words}
    
        
    def get_unk_probability(self, posterior):
        """Use the probability for '<unk>' in the language model """
        return self.lm.BaseScore(self.lm_state, "<unk>", self.lm_state2)
    
    def consume(self, word):
        self.lm.BaseScore(self.lm_state, str(word), self.lm_state2)
        self.lm_state, self.lm_state2 = self.lm_state2, self.lm_state
        self.history.append(str(word))
    
    def get_state(self):
        return self.lm_state.clone()
    
    def get_state(self):
        """Returns the current n-gram history """
        return self.history
    
    def set_state(self, state):
        """Sets the current n-gram history and LM state """
        self.history = state
        self._update_lm_state()

    def is_equal(self, state1, state2):
        return state1 == state2

