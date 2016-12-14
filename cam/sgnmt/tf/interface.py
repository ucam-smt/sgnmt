"""This module is the interface to the TensorFlow NMT and RNNLM implementations.
"""
import logging
import os

TENSORFLOW_AVAILABLE = True
try:
    from cam.sgnmt.tf.vanilla_decoder import TensorFlowNMTVanillaDecoder
    from cam.sgnmt.predictors.tf_nmt import TensorFlowNMTPredictor
    from cam.sgnmt.predictors.tf_rnnlm import TensorFlowRNNLMPredictor
    from tensorflow.models.rnn.translate.utils import model_utils as tf_model_utils
    import tensorflow as tf
    session = None
except:
    TENSORFLOW_AVAILABLE = False

def tf_get_nmt_predictor(args, nmt_config, path=None):
  """Get the TensorFlow NMT predictor.
    
  Args:
    args (object): SGNMT arguments from ``ArgumentParser``
    nmt_config (string): NMT configuration
    path (string): Path to NMT model or directory if passed separately from config file
    
  Returns:
    Predictor. An instance of ``TensorFlowNMTPredictor``
  """
  if not TENSORFLOW_AVAILABLE:
    logging.fatal("Could not find TensorFlow!")
    return None

  logging.info("Loading predictor {}".format(nmt_config))
  tf_config = dict()
  tf_model_utils.read_config(nmt_config, tf_config)
  if path:
    if os.path.isdir(path):
      tf_config['train_dir'] = path
    elif os.path.isfile(path):
      tf_config['model_path'] = path
  global session
  if not session:
    session = tf.Session()
  return TensorFlowNMTPredictor(args.cache_nmt_posteriors, tf_config, session)

def tf_get_nmt_vanilla_decoder(args, nmt_config):
  """Get the TensorFlow NMT vanilla decoder.

  Args:
    args (object): SGNMT arguments from ``ArgumentParser``
    nmt_config (string): NMT configuration

  Returns:
    Predictor. An instance of ``TensorFlowNMTVanillaDecoder``
  """
  if not TENSORFLOW_AVAILABLE:
    logging.fatal("Could not find TensorFlow!")
    return None
  return TensorFlowNMTVanillaDecoder(nmt_config)

def tf_get_rnnlm_predictor(rnnlm_config, path, variable_prefix="model"):
  """Get the TensorFlow RNNLM predictor.
    
  Args:    
    rnnlm_config (string): RNNLM configuration
    path (string): Path to RNNLM model or directory
    variable_prefix(string): prefix of model variables
    
  Returns:
    Predictor. An instance of ``TensorFlowRNNLMPredictor``
  """
  if not TENSORFLOW_AVAILABLE:
    logging.fatal("Could not find TensorFlow!")
    return None

  logging.info("Loading Rnnlm predictor %s" % rnnlm_config) 
  return TensorFlowRNNLMPredictor(rnnlm_config, path, variable_prefix)
