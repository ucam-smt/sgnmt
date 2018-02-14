"""This module is the interface to the blocks NMT implementation.
"""

import argparse
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


def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


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


def get_blocks_train_parser():
    """Get the parser object for NMT training configuration. """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument("--bokeh",  default=False, action="store_true",
                        help="Use bokeh server for plotting")
    parser.add_argument("--reshuffle",  default=False, action="store_true",
                        help="Reshuffle before each epoch")
    parser.add_argument("--slim_iteration_state",  default=False, action="store_true",
                        help="Per default the iteration state stores the data "
                        "stream and the main loop epoch iterator. Enabling "
                        "this option only stores the epoch iterator. This "
                        "results in a much smaller iteration state, but the "
                        "data stream is reset after reloading. Normally, you "
                        "can use slim iteration states if your data stream "
                        "does reshuffling")
    parser.add_argument("--reset_epoch",  default=False, action="store_true",
                        help="Set epoch_started in main loop status to false. "
                        "Sometimes required if you change training parameters "
                        "such as --mono_data_integration")
    parser.add_argument("--mono_data_integration", default="none",
                        choices=['none'],
                        help="This parameter specifies how to use "
                        "monolingual data. Currently, we only support "
                        "using the target data.\n\n"
                        "* 'none': Do not use monolingual data\n")
    parser.add_argument("--loss", default="default",
                        choices=['default', 'gleu'],
                        help="Training loss function.\n\n"
                        "* 'default': Standard loss function: squared error "
                        "with target feature maps, else cross entropy\n"
                        "* 'gleu': Reinforcement learning objective function "
                        "as proposed by Wu et al., 2016 (Googles NMT)")
    parser.add_argument("--add_mono_dummy_data", default=True, type='bool',
                        help="If the method specified with mono_data_"
                        "integration uses monolingual data, it usually "
                        "combines synthetic and dummy source sentences. Set "
                        "this to false to disable dummy source sentences.")
    parser.add_argument("--backtrans_nmt_config",  default="",
                        help="A string describing the configuration of the "
                        "back-translating NMT system. Syntax is equal to nmt_"
                        "config2 in decode.py: Comma separated list of name-"
                        "value pairs, where name is one of the NMT "
                        "configuration parameters. E.g. saveto=train.back,"
                        "src_vocab_size=50000,trg_vocab_size=50000")
    parser.add_argument("--backtrans_reload_frequency", default=0, type=int,
                        help="The back-translating NMT model is reloaded every"
                        " n updates. This is useful if the back-translating "
                        "NMT system is currently trained by itself with the "
                        "same policy. This enables us to train two NMT "
                        "systems in opposite translation directions and "
                        "benefit from gains in the other system immediately. "
                        "Set to 0 to disable reloading")
    parser.add_argument("--backtrans_store", default=True, type='bool',
                        help="Write the back-translated sentences to the "
                        "file system.")
    parser.add_argument("--backtrans_max_same_word", default=0.3, type=float,
                        help="Used for sanity check of the backtranslation. "
                        "If the most frequent word in the backtranslated "
                        "sentence has relative frequency higher than this, "
                         "discard this sentence pair")
    parser.add_argument("--learning_rate", default=0.002, type=float,
                        help="Learning rate for AdaGrad and Adam")
    parser.add_argument("--prune_every", default=-1, type=int,
                        help="Prune model every n iterations. Pruning is " 
                        "disabled if this is < 1")
    parser.add_argument("--prune_reset_every", default=-1, type=int,
                        help="Reset pruning statistics every n iterations. If " 
                        "set to -1, use --prune_every")
    parser.add_argument("--prune_n_steps", default=10, type=int,
                        help="Number of pruning steps until the target layer "
                        "sizes should be reached")
    parser.add_argument("--prune_layers",  
                        default="encfwdgru:1000,encbwdgru:1000,decgru:1000",
                        help="A comma separated list of <layer>:<size> pairs. "
                        "<layer> is one of 'encfwdgru', 'encbwdgru', 'decgru',"
                        " 'decmaxout' which should be shrunk to <size> during "
                        "training. Pruned neurons are marked by setting all "
                        "in- and output connection to zero.")
    parser.add_argument("--prune_layout_path",  
                        default="prune.layout",
                        help="Points to a file which defines which weight "
                        "matrices are connected to which prunable layers. The "
                        "rows/columns of these matrices are set to zero for "
                        "all removed neurons. The format of this file is \n"
                        "<layer> <in|out> <mat_name> <dim> <start-idx>=0.0\n"
                        "<layer> is one of the layer names specified via "
                        "--prune_layers. Set <start-idx> to 0.5 to add an "
                        "offset of half the matrix dimension to the indices.")
    parser.add_argument("--sampling_freq", default=13, type=int,
                        help="NOT USED, just to prevent old code from breaking")
    parser.add_argument("--hook_samples", default=0, type=int,
                        help="NOT USED, just to prevent old code from breaking")
    blocks_add_nmt_config(parser)
    return parser


def get_blocks_align_parser():
    """Get the parser object for NMT alignment configuration. """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    parser.add_argument("--iterations", default=50, type=int,
                        help="Number of optimization iterations for each token")
    parser.add_argument("--nmt_model_selector", default="bleu",
                        choices=['params', 'bleu', 'time'],
                        help="NMT training normally creates several files in "
                        "the ./train/ directory from which we can load the NMT"
                        " model. Possible options:\n\n"
                        "* 'params': Load parameters from params.npz. This is "
                        "usually the most recent model.\n"
                        "* 'bleu': Load from the best_bleu_params_* file with "
                        "the best BLEU score.\n"
                        "* 'time': Load from the most recent "
                        "best_bleu_params_* file.")
    parser.add_argument("--alignment_model", default="nam",
                        choices=['nam', 'nmt'],
                        help="Defines the alignment model.\n\n"
                        "* 'nam': Neural alignment model. Similar to NMT but "
                        "trains the alignment weights explicitly for each "
                        "sentence pair instead of using the NMT attention "
                        "model.\n"
                        "* 'nmt': Standard NMT attention model following "
                        "Bahdanau et. al., 2015.")
    parser.add_argument("--output_path", default="sgnmt-out.%s",
                        help="Path to the output files generated by SGNMT. You "
                        "can use the placeholder %%s for the format specifier.")
    parser.add_argument("--outputs", default="",
                        help="Comma separated list of output formats: \n\n"
                        "* 'csv': Plain text file with alignment matrix\n"
                        "* 'npy': Alignment matrices in numpy's npy format\n"
                        "* 'align': Usual (Pharaoh) alignment format.\n")
    
    blocks_add_nmt_config(parser)
    return parser


def get_blocks_batch_decode_parser():
    """Get the parser object for NMT batch decoding. """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    parser.add_argument("--src_test", default="test_en",
                        help="Path to source test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be indexed, i.e. use word IDs "
                        "instead of their string representations.")
    parser.add_argument("--range", default="",
                         help="Defines the range of sentences to be processed. "
                         "Syntax is equal to HiFSTs printstrings and lmerts "
                         "idxrange parameter: <start-idx>:<end-idx> (both "
                         "inclusive, start with 1). E.g. 2:5 means: skip the "
                         "first sentence, process next 4 sentences")
    parser.add_argument("--enc_max_words", default=5000, type=int,
                        help="Maximum number of words in an encoder batch. "
                        "These batches compute source side annotations. "
                        "Encoder batches are clustered by source sentence "
                        "length, so smaller batches are possible.")
    parser.add_argument("--min_jobs", default=2, type=int,
                        help="The CPU scheduler starts to construct small "
                        "jobs when the total number of jobs in the pipelines "
                        "is below this threshold. This prevents the computation "
                        "thread from being idle, at the cost of smaller " 
                        "batches")
    parser.add_argument("--max_tasks_per_job", default=450, type=int,
                        help="The maximum number of tasks in a single decoder "
                        "batch. Larger batches can exploit GPU parallelism "
                        "more efficiently, but limit the flexibility of the "
                        "CPU scheduler")
    parser.add_argument("--max_tasks_per_state_update_job", default=100, type=int,
                        help="Maximum number of tasks in a state update batch. "
                        "Larger batches are more efficient to compute on the "
                        "GPU, but delaying state updates for too long may "
                        "lead to smaller forward pass jobs.")
    parser.add_argument("--max_rows_per_job", default=20, type=int,
                        help="Maximum number of entries in a forward pass "
                        "batch. Note that each task in the batch gets at least "
                        "one entry, so this parameters applies only if there "
                        "are less than this threshold tasks left.")
    parser.add_argument("--min_tasks_per_bucket", default=100, type=int,
                        help="Minimum number of tasks in a bucket. Large "
                        "buckets give the CPU scheduler more flexibility, "
                        "but more padding may be required on the source "
                        "side, leading to more wasted computation.")
    parser.add_argument("--min_bucket_tolerance", default=8, type=int,
                        help="Minimum padding width in a bucket. Increasing "
                        "this leads to larger buckets and more flexible "
                        "scheduling and larger batches, but potentially "
                        "more wasteful state update computation due to "
                        "padding.")
    parser.add_argument("--beam", default=5, type=int,
                        help="Size of the beam.")
    
    blocks_add_nmt_config(parser)
    return parser
