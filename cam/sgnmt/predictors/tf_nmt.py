"""TensorFlow NMT implementation
"""

from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils
import logging

NEG_INF = float("-inf")

class TensorFlowNMTPredictor(Predictor):
    
    def __init__(self, nmt_config):
        super(TensorFlowNMTPredictor, self).__init__()
        raise NotImplementedError
        
    def get_unk_probability(self, posterior):
        # posterior is the returned value of the last predict_next call
        raise NotImplementedError
    
    def predict_next(self):
        # should return list, numpy array, or dictionary
        raise NotImplementedError
    
    def initialize(self, src_sentence):
        # src_sentence is list of integers, without <s> and </s>
        raise NotImplementedError
    
    def consume(self, word):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def set_state(self, state):
        raise NotImplementedError

    def reset(self):
        pass
