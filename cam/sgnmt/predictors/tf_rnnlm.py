"""TensorFlow RNNLM implementation
"""
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils

from tensorflow.models.rnn.ptb.utils import model_utils as rnnlm_model_utils
from tensorflow.models.rnn.ptb.utils import train_utils as rnnlm_train_utils
import tensorflow as tf

NEG_INF = float("-inf")

class TensorFlowRNNLMPredictor(Predictor):
    
  def __init__(self, path, model_config, variable_prefix="model"):
    super(TensorFlowRNNLMPredictor, self).__init__()
    self.session = tf.Session()

    self.model, config = rnnlm_model_utils.load_model(self.session, model_config, path,
                                                      use_log_probs=True,
                                                      variable_prefix=variable_prefix)
    self.vocab_size = config.vocab_size

    self.input = [utils.EOS_ID]
    self.model_state = {}
    self.word_count = 0

  def initialize(self, src_sentence):
    # src_sentence is list of integers, without <s> and </s>
    self.reset()

    # Initialize rnnlm state
    initial_state = self.model.initial_state.eval(session=self.session)
    self.model_state = initial_state

  def predict_next(self):
    # should return list, numpy array, or dictionary
    if self.input[0] == utils.EOS_ID and self.word_count > 0: # Predict EOS
      return {utils.EOS_ID: 0}

    posterior, self.model_state = rnnlm_train_utils.run_step_eval(self.session, self.model, self.input[0], self.model_state)
    return posterior

  def get_unk_probability(self, posterior):
    # posterior is the returned value of the last predict_next call
    return posterior[utils.UNK_ID] if len(posterior) > 1 else float("-inf")

  def consume(self, word):
    if word >= self.vocab_size:
      word = utils.UNK_ID
    self.consumed.append(word)
    self.input = [word]
    self.word_count = self.word_count + 1
    
  def get_state(self):
    return (self.input, self.model_state, self.word_count, self.consumed)

  def set_state(self, state):
    self.input, self.model_state, self.word_count, self.consumed = state

  def reset(self):
    self.input = [utils.EOS_ID]
    self.model_state = {}
    self.word_count = 0
    self.consumed = []

  def is_equal(self, state1, state2):
    """Returns true if the history is the same """
    _, _, _, consumed1 = state1
    _, _, _, consumed2 = state2
    return consumed1 == consumed2
