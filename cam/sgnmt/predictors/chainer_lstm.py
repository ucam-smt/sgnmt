
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils
import logging

try:
    # TODO Marcin: add all chainer imports here
    pass
except ImportError:
    pass # Deal with it in decode.py

NEG_INF = float("-inf")

class ChainerLstmPredictor(Predictor):
    
    def __init__(self, path):
        super(ChainerLstmPredictor, self).__init__()
        # TODO Marcin: Load model from path
        
    def get_unk_probability(self, posterior):
        return NEG_INF
    
    def predict_next(self):
        # TODO Marcin: Return a dictionary of log-likelihoods given 
        # current predictor state. The probably of ending the sequence
        # is taken from the token utils.EOS_ID
        if self.n_consumed < 7:
            return {3 : 0.0} # log 1 = 0.0, i.e. predict 3 with probability 1
        else:
            return {utils.EOS_ID : 0.0} # predict EOS_ID to end the sequence  
    
    def initialize(self, src_sentence):
        # TODO Marcin: src_sentence is a list of integers in a line of --src_test
        # This is called for each line in the --src_test file
        self.n_consumed = 0
    
    def consume(self, word):
        # TODO Marcin: E.g., if you feed the previous output as new input to the
        # LSTM, you can update the LSTM hidden state here (the parameter word is
        # the previous output
        self.n_consumed += 1
    
    def get_state(self):
        # TODO Marcin: Return LSTM hidden state
        return self.n_consumed
    
    def set_state(self, state):
        # TODO Marcin: Set the LSTM hidden state
        self.n_consumed = state

    def reset(self):
        pass
        