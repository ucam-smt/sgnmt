"""This module is the interface to the TensorFlow NMT and RNNLM implementations.
"""
import logging
import os

TENSORFLOW_AVAILABLE = True
try:
    from cam.sgnmt.predictors.tf_nmt import TensorFlowNMTPredictor
    from cam.sgnmt.predictors.tf_rnnlm import TensorFlowRNNLMPredictor
    import tensorflow as tf
    session = None
except:
    TENSORFLOW_AVAILABLE = False

def tf_get_nmt_predictor(args, nmt_path, nmt_config):
  """Get the TensorFlow NMT predictor.
    
  Args:
    args (object): SGNMT arguments from ``ArgumentParser``
    nmt_config (string): NMT configuration
    path (string): Path to NMT model or directory
    
  Returns:
    Predictor. An instance of ``TensorFlowNMTPredictor``
  """
  if not TENSORFLOW_AVAILABLE:
    logging.fatal("Could not find TensorFlow!")
    return None

  logging.info("Loading tensorflow nmt predictor")
  if os.path.isdir(nmt_path):
    nmt_config['train_dir'] = nmt_path
  elif os.path.isfile(nmt_path):
    nmt_config['model_path'] = nmt_path
  global session
  if not session:
    session = tf.Session()
  return TensorFlowNMTPredictor(args.cache_nmt_posteriors, nmt_config, session)

def tf_get_default_nmt_config():
    """Get default NMT configuration. """
    config = {}

    def _parse_flags(flags):
        """Replicated here from tensorflow.python.platform.flags to avoid processing
        sgnmt command line flags. """
        from tensorflow.python.platform.flags import _global_parser
        result, unknown_args = _global_parser.parse_known_args([])
        if unknown_args:
            logging.error("Unknown arguments: {}".format(unknown_args))
            exit(1)
        for flag_name, val in vars(result).items():
            flags.__dict__['__flags'][flag_name] = val
        flags.__dict__['__parsed'] = True
        flags = vars(flags)

    from tensorflow.models.rnn.translate.train import FLAGS as train_flags
    _parse_flags(train_flags)
    for key,value in train_flags.__dict__['__flags'].iteritems():
      config[key] = value
    return config

def tf_get_nmt_vanilla_decoder(args, nmt_specs):
  """Get the TensorFlow NMT vanilla decoder. Not implemented yet.

  Returns:
    None.
  """
  if not TENSORFLOW_AVAILABLE:
    logging.fatal("Could not find TensorFlow!")
  logging.fatal("Vanilla decoder not implemented in TensorFlow!")
  return None

def tf_get_rnnlm_predictor(rnnlm_path, rnnlm_config, variable_prefix="model"):
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

  logging.info("Loading tensorflow rnnlm predictor")
  return TensorFlowRNNLMPredictor(rnnlm_path, rnnlm_config, variable_prefix)

_rnnlm_count = 0
def tf_get_rnnlm_prefix(var_prefix='model'):
    """This is a helper function to increment the variable prefix when
    decoding with multiple rnnlm models. This assumes that models have
    been prefixed 'model', 'model2', 'model3' etc.

    Args:
        variable prefix for first model

    Returns:
        variable prefix according to model index
    """
    global _rnnlm_count
    _rnnlm_count += 1
    return "%s%d" % (var_prefix, _rnnlm_count) if _rnnlm_count > 1 else var_prefix
