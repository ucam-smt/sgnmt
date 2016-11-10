"""TensorFlow NMT implementation
"""
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils
import copy
import logging
import numpy as np

from cam.sgnmt.misc.trie import SimpleTrie
from tensorflow.models.rnn.translate.utils import data_utils as tf_data_utils
from tensorflow.models.rnn.translate.utils import model_utils as tf_model_utils

NEG_INF = float("-inf")

class TensorFlowNMTPredictor(Predictor):
  '''Neural MT predictor'''
  def __init__(self, enable_cache, config, session):
      super(TensorFlowNMTPredictor, self).__init__()
      self.config = config
      self.session = session

      # Add missing entries in config
      if self.config['encoder'] == "bow":
        self.config['init_backward'] = False
        self.config['use_seqlen'] = False
      else:
        self.config['bow_init_const'] = False
        self.config['use_bow_mask'] = False

      # Load tensorflow model
      self.model, self.training_graph, self.encoding_graph, \
        self.single_step_decoding_graph, self.buckets = tf_model_utils.load_model(self.session, config)
      self.model.batch_size = 1  # We decode one sentence at a time.

      self.enc_out = {}
      self.decoder_input = [tf_data_utils.GO_ID]
      self.dec_state = {}
      self.bucket_id = -1
      self.num_heads = 1
      self.word_count = 0

      if config['no_pad_symbol']:
        # This needs to be set in tensorflow data_utils for correct source masks
        tf_data_utils.no_pad_symbol()
        logging.info("UNK_ID=%d" % tf_data_utils.UNK_ID)
        logging.info("PAD_ID=%d" % tf_data_utils.PAD_ID)

      self.enable_cache = enable_cache
      if self.enable_cache:
        logging.info("Cache enabled..")

  def initialize(self, src_sentence):
    # src_sentence is list of integers, without <s> and </s>
    self.reset()
    self.posterior_cache = SimpleTrie()
    self.states_cache = SimpleTrie()

    src_sentence = [w if w < self.config['src_vocab_size'] else tf_data_utils.UNK_ID
                    for w in src_sentence]
    if self.config['add_src_eos']:
      src_sentence.append(tf_data_utils.EOS_ID)

    feasible_buckets = [b for b in xrange(len(self.buckets))
                        if self.buckets[b][0] >= len(src_sentence)]
    if not feasible_buckets:
      # Get a new bucket
      bucket = tf_model_utils.make_bucket(len(src_sentence))
      logging.info("Add new bucket={} and update model".format(bucket))
      self.buckets.append(bucket)
      self.model.update_buckets(self.buckets)
      self.bucket_id = len(self.buckets) - 1
    else:
      self.bucket_id = min(feasible_buckets)

    encoder_inputs, _, _, sequence_length, src_mask, bow_mask = self.training_graph.get_batch(
            {self.bucket_id: [(src_sentence, [])]}, self.bucket_id, self.config['encoder'])
    logging.info("bucket={}".format(self.buckets[self.bucket_id]))

    last_enc_state, self.enc_out = self.encoding_graph.encode(
            self.session, encoder_inputs, self.bucket_id, sequence_length)

    # Initialize decoder state with last encoder state
    self.dec_state["dec_state"] = last_enc_state
    for a in xrange(self.num_heads):
      self.dec_state["dec_attns_%d" % a] = np.zeros((1, self.enc_out['enc_v_0'].size), dtype=np.float32)

    if self.config['use_src_mask']:
      self.dec_state["src_mask"] = src_mask
      self.src_mask_orig = src_mask.copy()

    if self.config['use_bow_mask']:
      self.dec_state["bow_mask"] = bow_mask
      self.bow_mask_orig = bow_mask.copy()

  def is_history_cachable(self):
    """Returns true if cache is enabled and history contains UNK """
    if not self.enable_cache:
      return False
    for w in self.consumed:
      if w == tf_data_utils.UNK_ID:
        return True
    return False

  def predict_next(self):
    # should return list, numpy array, or dictionary
    if self.decoder_input[0] == tf_data_utils.EOS_ID: # Predict EOS
        return {tf_data_utils.EOS_ID: 0}

    use_cache = self.is_history_cachable()
    if use_cache:
      posterior = self.posterior_cache.get(self.consumed)
      if not posterior is None:
        logging.debug("Loaded NMT posterior from cache for %s" %
                      self.consumed)
        return posterior

    output, self.dec_state = self.single_step_decoding_graph.decode(self.session, self.enc_out,
                                               self.dec_state, self.decoder_input, self.bucket_id,
                                               self.config['use_src_mask'], self.word_count,
                                               self.config['use_bow_mask'])
    if use_cache:
      self.posterior_cache.add(self.consumed, output[0])

    return output[0]

  def get_unk_probability(self, posterior):
    # posterior is the returned value of the last predict_next call
    return posterior[utils.UNK_ID] if len(posterior) > 1 else float("-inf")

  def consume(self, word):
    if word >= self.config['trg_vocab_size']:
      word = tf_data_utils.UNK_ID  # history is kept according to nmt vocab
    logging.debug("Consume word={}".format(word))
    self.consumed.append(word)

    use_cache = self.is_history_cachable()
    if use_cache:
      s = self.states_cache.get(self.consumed)
      if not s is None:
        logging.debug("Loaded NMT decoder states from cache for %s" %
                      self.consumed)
        states = copy.deepcopy(s)
        self.decoder_input = states[0]
        self.dec_state = states[1]
        self.word_count = states[2]
        return

    self.decoder_input = [word]
    self.word_count = self.word_count + 1

    if use_cache:
      states = (self.decoder_input, self.dec_state, self.word_count)
      self.states_cache.add(self.consumed, copy.deepcopy(states))

  def get_state(self):
    return (self.decoder_input, self.dec_state, self.word_count, self.consumed)

  def set_state(self, state):
    self.decoder_input, self.dec_state, self.word_count, self.consumed = state

  def reset(self):
    self.enc_out = {}
    self.decoder_input = [tf_data_utils.GO_ID]
    self.dec_state = {}
    self.word_count = 0
    self.consumed = []

  def is_equal(self, state1, state2):
    """Returns true if the history is the same """
    _, _, _, consumed1 = state1
    _, _, _, consumed2 = state2
    return consumed1 == consumed2
