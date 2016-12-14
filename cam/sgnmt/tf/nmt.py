"""This module is the interface to the TensorFlow NMT implementation.
"""

import logging

TENSORFLOW_AVAILABLE = True
try:    
    # TODO: Insert some TensorFlow specific imports here
    
    from cam.sgnmt.tf.vanilla_decoder import TensorFlowNMTVanillaDecoder
    from cam.sgnmt.predictors.tf_nmt import TensorFlowNMTPredictor
except:
    TENSORFLOW_AVAILABLE = False

def tf_get_nmt_predictor(args, nmt_path, nmt_config):
    """Get the TensorFlow NMT predictor.
    
    Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_config (dict): NMT configuration
    
    Returns:
        Predictor. An instance of ``TensorFlowNMTPredictor``
    """
    if not TENSORFLOW_AVAILABLE:
        logging.fatal("Could not find TensorFlow!")
        return None
    return TensorFlowNMTPredictor(nmt_path, nmt_config)


def tf_get_nmt_vanilla_decoder(args, nmt_path, nmt_config):
    """Get the TensorFlow NMT vanilla decoder.
    
    Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_config (dict): NMT configuration
    
    Returns:
        Predictor. An instance of ``TensorFlowNMTVanillaDecoder``
    """
    if not TENSORFLOW_AVAILABLE:
        logging.fatal("Could not find TensorFlow!")
        return None
    return TensorFlowNMTVanillaDecoder(nmt_path, nmt_config)

def tf_get_default_nmt_config():
    """Get default NMT configuration. """
    config = {}
    # like in blocks.blocks_get_default_nmt_config
    return config

def tf_get_default_rnnlm_config():
    """Get default RNNLM configuration. """
    config = {}
    # like in blocks.blocks_get_default_nmt_config
    return config