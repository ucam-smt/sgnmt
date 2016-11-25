"""This module is the interface to the blocks NMT implementation.
"""

import logging
import os
import re

from cam.sgnmt.misc.sparse import FileBasedFeatMap


BLOCKS_AVAILABLE = True
try:
    from blocks.search import BeamSearch # To check if blocks is available
    
    from cam.sgnmt.blocks.vanilla_decoder import BlocksNMTVanillaDecoder
    from cam.sgnmt.predictors.blocks_nmt import BlocksNMTPredictor
    from cam.sgnmt.predictors.blocks_nmt import BlocksUnboundedNMTPredictor
except:
    BLOCKS_AVAILABLE = False 


def _add_sparse_feat_maps_to_config(nmt_config):
    """Adds the sparse feature map instances to the nmt config """
    new_config = dict(nmt_config)
    if nmt_config['src_sparse_feat_map']:
        new_config['src_sparse_feat_map'] = FileBasedFeatMap(
                                        nmt_config['enc_embed'],
                                        nmt_config['src_sparse_feat_map'])
    if nmt_config['trg_sparse_feat_map']:
        new_config['trg_sparse_feat_map'] = FileBasedFeatMap(
                                        nmt_config['dec_embed'],
                                        nmt_config['trg_sparse_feat_map'])
    return new_config


def blocks_get_nmt_predictor(args, nmt_config):
    """Get the Blocks NMT predictor. If a target sparse feature map is
    used, we create an unbounded vocabulary NMT predictor. Otherwise,
    the normal bounded NMT predictor is returned
    
    Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_config (dict): NMT configuration
    
    Returns:
        Predictor. The NMT predictor
    """
    if not BLOCKS_AVAILABLE:
        logging.fatal("Could not find Blocks!")
        return None
    nmt_config = _add_sparse_feat_maps_to_config(nmt_config)
    if nmt_config['trg_sparse_feat_map']:
        return BlocksUnboundedNMTPredictor(
                                    get_nmt_model_path(args.nmt_model_selector,
                                                       nmt_config),
                                    args.gnmt_beta,
                                    nmt_config)
    return BlocksNMTPredictor(get_nmt_model_path(args.nmt_model_selector,
                                                 nmt_config),
                              args.gnmt_beta,
                              args.cache_nmt_posteriors,
                              nmt_config)


def blocks_get_nmt_vanilla_decoder(args, nmt_config):
    """Get the Blocks NMT vanilla decoder.
    
    Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_config (dict): NMT configuration
    
    Returns:
        Predictor. An instance of ``BlocksNMTVanillaDecoder``
    """
    if not BLOCKS_AVAILABLE:
        logging.fatal("Could not find Blocks!")
        return None
    nmt_config = _add_sparse_feat_maps_to_config(nmt_config)
    return BlocksNMTVanillaDecoder(get_nmt_model_path(args.nmt_model_selector,
                                                      nmt_config),
				                    nmt_config, args)


PARAMS_FILE_NAME = 'params.npz'
"""Name of the default model file (not checkpoints) """


BEST_BLEU_PATTERN = re.compile('^best_bleu_params_([0-9]+)_BLEU([.0-9]+).npz$')
"""Pattern for checkpoints created in training for model selection """


def get_nmt_model_path_params(nmt_config):
    """Returns the path to the params.npz. This file usually contains 
    the latest model parameters.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the params.npz
    """
    return '%s/%s' % (nmt_config['saveto'], PARAMS_FILE_NAME)


def get_nmt_model_path_best_bleu(nmt_config):
    """Returns the path to the checkpoint with the best BLEU score. If
    no checkpoint can be found, back up to params.npz.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the checkpoint file with best BLEU score
    """
    best = 0.0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and float(m.group(2)) > best:
            best = float(m.group(2))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


def get_nmt_model_path_most_recent(nmt_config):
    """Returns the path to the most recent checkpoint. If
    no checkpoint can be found, back up to params.npz.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the most recent checkpoint file
    """
    best = 0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and int(m.group(1)) > best:
            best = int(m.group(1))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


def get_nmt_model_path(nmt_model_selector, nmt_config):
    """Get the path to the NMT model according the given NMT config.
    This switches between the most recent checkpoint, the best BLEU 
    checkpoint, or the latest parameters (params.npz). This method
    delegates to ``get_nmt_model_path_*``. This
    method relies on the global ``args`` variable.
    
    Args:
        nmt_model_selector (string): the ``--nmt_model_selector`` arg
                                     which defines the policy to decide
                                     which NMT model to load (params,
                                     bleu, or time)
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        string. Path to the NMT model file
    """
    if nmt_model_selector == 'params':
        return get_nmt_model_path_params(nmt_config)
    elif nmt_model_selector == 'bleu':
        return get_nmt_model_path_best_bleu(nmt_config)
    elif nmt_model_selector == 'time':
        return get_nmt_model_path_most_recent(nmt_config)
    logging.fatal("NMT model selector %s not available. Please double-check "
                  "the --nmt_model_selector parameter." % nmt_model_selector)


