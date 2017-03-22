"""This module is the interface to the blocks NMT implementation.
"""

import logging
import os
import re

from cam.sgnmt.misc.sparse import FileBasedFeatMap


BLOCKS_AVAILABLE = True
try:
    from blocks.search import BeamSearch # To check if blocks is available
    
    from cam.sgnmt.blocks.vanilla_decoder import BlocksNMTVanillaDecoder,\
                                                 BlocksNMTEnsembleVanillaDecoder
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


def blocks_get_nmt_predictor(args, nmt_path, nmt_config):
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
    if nmt_path:
        nmt_config['saveto'] = nmt_path
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


def blocks_get_nmt_vanilla_decoder(args, nmt_specs):
    """Get the Blocks NMT vanilla decoder.
    
    Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_specs (list): List of (nmt_path,nmt_config) tuples, one
                          entry for each model in the ensemble
    
    Returns:
        Predictor. An instance of ``BlocksNMTVanillaDecoder``
    """
    if not BLOCKS_AVAILABLE:
        logging.fatal("Could not find Blocks!")
        return None
    nmt_specs_blocks = []
    for nmt_path, nmt_config in nmt_specs:
        nmt_config = _add_sparse_feat_maps_to_config(nmt_config)
        if nmt_path:
            nmt_config['saveto'] = nmt_path
        nmt_specs_blocks.append((get_nmt_model_path(args.nmt_model_selector,
                                                    nmt_config),
                                 nmt_config))
    if len(nmt_specs_blocks) == 1:        
        return BlocksNMTVanillaDecoder(nmt_specs_blocks[0][0],
                                       nmt_specs_blocks[0][1],
                                       args)
    return BlocksNMTEnsembleVanillaDecoder(nmt_specs_blocks, args)


def blocks_add_nmt_config(parser):
    """Adds the nmt options to the command line configuration.
    
    Args:
        parser (object): Parser or ArgumentGroup object
    """
    default_config = blocks_get_default_nmt_config()
    nmt_help_texts = blocks_get_nmt_config_help()
    for k in default_config:
        arg_type = type(default_config[k])
        if arg_type == bool:
            arg_type = 'bool'
        parser.add_argument(
                    "--%s" % k,
                    default=default_config[k],
                    type=arg_type,
                    help=nmt_help_texts[k])


def blocks_get_default_nmt_config():
    """Get default NMT configuration. """
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['att_nhids'] = -1
    config['maxout_nhids'] = -1

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620
    
    # Number of layers in encoder and decoder
    config['enc_layers'] = 1
    config['dec_layers'] = 1
    
    # Network layout
    config['dec_readout_sources'] = "sfa"
    config['dec_attention_sources'] = "s"
    
    config['enc_share_weights'] = True
    config['dec_share_weights'] = True
    
    # Skip connections
    config['enc_skip_connections'] = False
    
    # How to derive annotations from the encoder. Comma
    # separated list of strategies.
    # - 'direct': directly use encoder hidden state
    # - 'hierarchical': Create higher level annotations with an 
    #                   attentional RNN 
    config['annotations'] = "direct"
    
    # Decoder initialisation
    config['dec_init'] = "last"

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = './train'
    
    # Attention
    config['attention'] = 'content'
    
    # External memory structure
    config['memory'] = 'none'
    config['memory_size'] = 500

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = 0.0

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = './data/'
    scriptsdir = '../scripts/'

    # Source and target datasets
    config['src_data'] = datadir + 'train.ids.shuf.en'
    config['trg_data'] = datadir + 'train.ids.shuf.fr'
    
    # Monolingual data (for use see --mono_data_integration
    config['src_mono_data'] = datadir + 'mono.ids.en'
    config['trg_mono_data'] = datadir + 'mono.ids.fr'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30003
    config['trg_vocab_size'] = 30003
    
    # Mapping files for using sparse feature word representations
    config['src_sparse_feat_map'] = ""
    config['trg_sparse_feat_map'] = ""

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = 'perl ' + scriptsdir + 'multi-bleu.perl %s <'

    # Validation set source file
    config['val_set'] = datadir + 'dev.ids.en'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'dev.ids.fr'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 750

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 6000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 80000

    # fs439: Blocks originally creates dumps of the entire main loop
    # when the BLEU on the dev set improves. This, however, cannot be
    # read to load parameters from, so we create BEST_BLEU_PARAMS*
    # files instead. Set the following parameter to true if you still
    # want to create the old style archives
    config['store_full_main_loop'] = False
    
    # fs439: Fix embeddings when training
    config['fix_embeddings'] = False

    return config


def blocks_get_nmt_config_help():
    """Creates a dictionary with help text for the NMT configuration """

    config = {}
    config['seq_len'] = "Sequences longer than this will be discarded"
    config['enc_nhids'] = "Number of hidden units in encoder GRU"
    config['dec_nhids'] = "Number of hidden units in decoder GRU"
    config['att_nhids'] = "Dimensionality of attention match vector (-1 to " \
                          "use dec_nhids)"
    config['maxout_nhids'] = "Dimensionality of maxout output layer (-1 to " \
                             "use dec_nhids)"
    config['enc_embed'] = "Dimension of the word embedding matrix in encoder"
    config['dec_embed'] = "Dimension of the word embedding matrix in decoder"
    config['enc_layers'] = "Number of encoder layers"
    config['dec_layers'] = "Number of decoder layers (NOT IMPLEMENTED for != 1)"
    config['dec_readout_sources'] = "Sources used by readout network: f for " \
                                    "feedback, s for decoder states, a for " \
                                    "attention (context vector)"
    config['dec_attention_sources'] = "Sources used by attention: f for " \
                                      "feedback, s for decoder states"
    config['enc_share_weights'] = "Whether to share weights in deep encoders"
    config['dec_share_weights'] = "Whether to share weights in deep decoders"
    config['enc_skip_connections'] = "Add skip connection in deep encoders"
    config['annotations'] = "Annotation strategy (comma-separated): " \
                            "direct, hierarchical"
    config['dec_init'] = "Decoder state initialisation: last, average, constant"
    config['attention'] = "Attention mechanism: none, content, nbest-<n>, " \
                          "coverage-<n>, tree, content-<n>"
    config['memory'] = 'External memory: none, stack'
    config['memory_size'] = 'Size of external memory structure'
    config['saveto'] = "Where to save model, same as 'prefix' in groundhog"
    config['batch_size'] = "Batch size"
    config['sort_k_batches'] = "This many batches will be read ahead and sorted"
    config['step_rule'] = "Optimization step rule"
    config['step_clipping'] = "Gradient clipping threshold"
    config['weight_scale'] = "Std of weight initialization"
    config['weight_noise_ff'] = "Weight noise flag for feed forward layers"
    config['weight_noise_rec'] = "Weight noise flag for recurrent layers"
    config['dropout'] = "Dropout ratio, applied only after readout maxout"
    config['src_data'] = "Source dataset"
    config['trg_data'] = "Target dataset"
    config['src_mono_data'] = "Source language monolingual data (for use " \
                              "see --mono_data_integration)"
    config['trg_mono_data'] = "Target language monolingual data (for use " \
                              "see --mono_data_integration)"
    config['src_vocab_size'] = "Source vocab size, including special tokens"
    config['trg_vocab_size'] = "Target vocab size, including special tokens"
    config['src_sparse_feat_map'] = "Mapping files for using sparse feature " \
                                    "word representations on the source side"
    config['trg_sparse_feat_map'] = "Mapping files for using sparse feature " \
                                    "word representations on the target side"
    config['normalized_bleu'] = "Length normalization IN TRAINING"
    config['bleu_script'] = "BLEU script used during training for model selection"
    config['val_set'] = "Validation set source file"
    config['val_set_grndtruth'] = "Validation set gold file"
    config['output_val_set'] = "Print validation output to file"
    config['val_set_out'] = "Validation output file"
    config['beam_size'] = "Beam-size for decoding DURING TRAINING"
    config['finish_after'] = "Maximum number of updates"
    config['reload'] = "Reload model from files if exist"
    config['save_freq'] = "Save model after this many updates"
    config['bleu_val_freq'] = "Validate bleu after this many updates"
    config['val_burn_in'] = "Start bleu validation after this many updates"
    config['store_full_main_loop'] = "Old style archives (not recommended)"
    config['fix_embeddings'] = "Fix embeddings during training"
    return config


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


