"""This is the main runner script for SGNMT decoding. 
SGNMT can run in three different modes. The standard mode 'file' reads
sentences to translate from a plain text file. The mode 'stdin' can be
used to parse stdin. The last mode 'shell' enables interactive inter-
action with SGNMT via keyboard. Note that SGNMT exclusively operates on 
integer representations of words, the mapping between word ids and
their string representations needs to be done outside SGNMT. For 
detailed usage descriptions please visit the tutorial home page:

http://ucam-smt.github.io/tutorial/sgnmt

Note that the configuration is handled through this module and the ``ui``
module. 
"""

import logging
import os
import pprint
import sys
import time
import traceback

from cam.sgnmt import ui
from cam.sgnmt import utils
from cam.sgnmt.blocks.nmt import blocks_get_nmt_predictor, \
                                 blocks_get_nmt_vanilla_decoder
from cam.sgnmt.decoding import core
from cam.sgnmt.decoding.astar import AstarDecoder
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.bigramgreedy import BigramGreedyDecoder
from cam.sgnmt.decoding.bow import BOWDecoder
from cam.sgnmt.decoding.bucket import BucketDecoder
from cam.sgnmt.decoding.core import CLOSED_VOCAB_SCORE_NORM_NONE, \
                                   CLOSED_VOCAB_SCORE_NORM_EXACT, \
                                   CLOSED_VOCAB_SCORE_NORM_REDUCED, \
                                   CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK
from cam.sgnmt.decoding.core import UnboundedVocabularyPredictor
from cam.sgnmt.decoding.dfs import DFSDecoder
from cam.sgnmt.decoding.flip import FlipDecoder
from cam.sgnmt.decoding.greedy import GreedyDecoder
from cam.sgnmt.decoding.heuristics import GreedyHeuristic, \
                                         PredictorHeuristic, \
                                         ScorePerWordHeuristic, StatsHeuristic
from cam.sgnmt.decoding.restarting import RestartingDecoder
from cam.sgnmt.output import TextOutputHandler, \
                             NBestOutputHandler, \
                             FSTOutputHandler, \
                             StandardFSTOutputHandler
from cam.sgnmt.predictors.automata import FstPredictor, \
                                         RtnPredictor, \
                                         NondeterministicFstPredictor
from cam.sgnmt.predictors.bow import BagOfWordsPredictor, \
    BagOfWordsSearchPredictor
from cam.sgnmt.predictors.chainer_lstm import ChainerLstmPredictor
from cam.sgnmt.predictors.ffnnlm import NPLMPredictor
from cam.sgnmt.predictors.forced import ForcedPredictor, ForcedLstPredictor
from cam.sgnmt.predictors.grammar import RuleXtractPredictor
from cam.sgnmt.predictors.length import WordCountPredictor, NBLengthPredictor, \
    ExternalLengthPredictor, NgramCountPredictor
from cam.sgnmt.predictors.misc import IdxmapPredictor, UnboundedIdxmapPredictor, \
    UnboundedAltsrcPredictor, AltsrcPredictor, Word2charPredictor
from cam.sgnmt.predictors.misc import UnkCountPredictor
from cam.sgnmt.predictors.ngram import SRILMPredictor
from cam.sgnmt.predictors.tf_rnnlm import TensorFlowRNNLMPredictor
from cam.sgnmt.tf.nmt import tf_get_nmt_predictor, \
                             tf_get_nmt_vanilla_decoder
from cam.sgnmt.ui import get_args, get_parser, validate_args


# Load configuration from command line arguments or configuration file
args = get_args()

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
if args.verbosity == 'debug':
    logging.getLogger().setLevel(logging.DEBUG)
elif args.verbosity == 'info':
    logging.getLogger().setLevel(logging.INFO)
elif args.verbosity == 'warn':
    logging.getLogger().setLevel(logging.WARN)
elif args.verbosity == 'error':
    logging.getLogger().setLevel(logging.ERROR)

validate_args(args)

# Load NMT engine(s)
engines = args.nmt_engine.split(',')
get_nmt_predictors = []
if len(engines) > 1:
    logging.info("Found multiple nmt engines")
    if args.decoder == "vanilla":
        logging.fatal("Vanilla decoder currently not supported for multiple engines!")
for engine in engines:
    if engine == "blocks":
        get_nmt_predictors.append(blocks_get_nmt_predictor)
        get_nmt_vanilla_decoder = blocks_get_nmt_vanilla_decoder
    elif engine == "tensorflow":
        get_nmt_predictors.append(tf_get_nmt_predictor)
        get_nmt_vanilla_decoder = tf_get_nmt_vanilla_decoder
    elif args.nmt_engine != 'none':
        logging.fatal("NMT engine %s is not supported (yet)!" % args.nmt_engine)

# Prepare tensorflow config(s)
tf_configs = [ config for config in args.tensorflow_config.split(',') ]
tf_paths = None if not args.tensorflow_path else \
  [ path for path in args.tensorflow_path.split(',') ]
num_tf_models = len(tf_configs)

# Support old scheme for reserved word indices
if args.legacy_indexing:
    utils.switch_to_old_indexing()
    
# Log summation (how to compute log(exp(l1)+exp(l2)) for log values l1,l2)
if args.log_sum == 'tropical':
    utils.log_sum = utils.log_sum_tropical_semiring

# Predictor combination schemes
if args.combination_scheme == 'length_norm':
    if args.apply_combination_scheme_to_partial_hypos:
        core.breakdown2score_partial = core.breakdown2score_length_norm
    else:
        core.breakdown2score_full = core.breakdown2score_length_norm
if args.combination_scheme == 'bayesian':
    if args.apply_combination_scheme_to_partial_hypos:
        core.breakdown2score_partial = core.breakdown2score_bayesian
    else:
        core.breakdown2score_full = core.breakdown2score_bayesian  


def get_nmt_config(args):
    """Get the NMT model configuration array. This should correspond to
    the settings during NMT training. See the module 
    ``machine_translation.configurations.get_gnmt_config()`` for the 
    default settings. Values can be overriden with values from the
    ``args`` argument coming from command line arguments or a 
    configuration file.
    
    Args:
        args (object):  SGNMT configuration from ``ui.get_args()``
    
    Returns:
        dict. NMT model configuration updated through ``args``
    """
    nmt_config = ui.get_nmt_config()
    for k in dir(args):
        if k in nmt_config:
            nmt_config[k] = getattr(args, k)
    logger.debug("Model options:\n{}".format(pprint.pformat(nmt_config)))
    return nmt_config


_override_args_cnts = {}
def _get_override_args(field):
    """This is a helper function for arguments which can be overridden 
    when using multiple instances of the same predictor. E.g., it is 
    possible to use two fst predictors, and specify the path to the 
    first directory with --fst_path, and the path to the second 
    directory with --fst_path2.
    
    Args:
        field (string):  Field in ``args`` for which also field2, 
                         field3... is also possible
    
    Returns:
        object. If count smaller than 2, returns ``args.field``. 
        Otherwise, returns ``args.field+count`` if specified, or backup
        to ``args.field``
    """
    default = getattr(args, field)
    if field in _override_args_cnts:
        _override_args_cnts[field] += 1
        overriden = getattr(args, "%s%d" % (field, _override_args_cnts[field]))
        return overriden if overriden else default
    _override_args_cnts[field] = 1
    return default


def add_predictors(decoder, nmt_config):
    """Adds all enabled predictors to the ``decoder``. This function 
    makes heavy use of the global ``args`` which contains the
    SGNMT configuration. Particularly, it reads out ``args.predictors``
    and adds appropriate instances to ``decoder``.
    TODO: Refactor this method as it is waaaay tooooo looong
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add predictors to this instance with
            ``add_predictor()``
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    """
    preds = args.predictors.split(",")
    if not preds:
        logging.fatal("Require at least one predictor! See the --predictors "
                      "argument for more information.")
    weights = None
    if args.predictor_weights:
        weights = args.predictor_weights.strip().split(",")
        if len(preds) != len(weights):
            logging.fatal("Specified %d predictors, but %d weights. Please "
                      "revise the --predictors and --predictor_weights "
                      "arguments" % (len(preds), len(weights)))
            return
    
    pred_weight = 1.0
    nmt_pred_count = 0
    try:
        for idx,pred in enumerate(preds): # Add predictors one by one
            wrappers = []
            if '_' in pred: 
                # Handle weights when we have wrapper predictors
                wrappers = pred.split('_')
                pred = wrappers[-1]
                wrappers = wrappers[-2::-1]
                if weights:
                    wrapper_weights = [float(w) for w in weights[idx].split('_')]
                    pred_weight = wrapper_weights[-1]
                    wrapper_weights = wrapper_weights[-2::-1]
                else:
                    wrapper_weights = [1.0] * len(wrappers)
            elif weights:
                pred_weight = float(weights[idx])
                
            if pred == 'nmt' or pred == 'fnmt' or pred == 'anmt':
                # Update NMT config in case --nmt_configX has been used
                nmt_pred_count += 1
                if nmt_pred_count > 1:
                    add_config = getattr(args, "nmt_config%d" % nmt_pred_count)
                    if add_config:
                        for pair in add_config.split(","):
                            (k,v) = pair.split("=", 1)
                            nmt_config[k] = type(nmt_config[k])(v)            
            # Create predictor instances for the string argument ``pred``
            if pred == "nmt":
                get_nmt_predictor = get_nmt_predictors.pop(0)
                engine = engines.pop(0)
                if engine == "tensorflow":
                  tf_nmt_config = tf_configs.pop(0)
                  tf_path = None if not tf_paths else tf_paths.pop(0)
                  p = get_nmt_predictor(args, tf_nmt_config, tf_path)
                else:
                  p = get_nmt_predictor(args, nmt_config)
            elif pred == "fst":
                p = FstPredictor(_get_override_args("fst_path"),
                                 args.use_fst_weights,
                                 args.normalize_fst_weights,
                                 skip_bos_weight=args.fst_skip_bos_weight,
                                 to_log=args.fst_to_log)
            elif pred == "nfst":
                p = NondeterministicFstPredictor(_get_override_args("fst_path"),
                                                 args.use_fst_weights,
                                                 args.normalize_fst_weights,
                                                 args.fst_skip_bos_weight,
                                                 to_log=args.fst_to_log)
            elif pred == "forced":
                p = ForcedPredictor(args.trg_test)
            elif pred == "bow":
                p = BagOfWordsPredictor(
                                args.trg_test,
                                args.bow_accept_subsets,
                                args.bow_accept_duplicates,
                                args.heuristic_scores_file,
                                args.collect_statistics,
                                "consumed" in args.bow_heuristic_strategies,
                                "remaining" in args.bow_heuristic_strategies,
                                args.bow_diversity_heuristic_factor,
                                args.bow_equivalence_vocab_size)
            elif pred == "bowsearch":
                p = BagOfWordsSearchPredictor(
                                decoder,
                                args.hypo_recombination,
                                args.trg_test,
                                args.bow_accept_subsets,
                                args.bow_accept_duplicates,
                                args.heuristic_scores_file,
                                args.collect_statistics,
                                "consumed" in args.bow_heuristic_strategies,
                                "remaining" in args.bow_heuristic_strategies,
                                args.bow_diversity_heuristic_factor,
                                args.bow_equivalence_vocab_size)
            elif pred == "forcedlst":
                feat_name = _get_override_args("forcedlst_sparse_feat")
                p = ForcedLstPredictor(args.trg_test,
                                       args.use_nbest_weights,
                                       feat_name if feat_name else None)
            elif pred == "rtn":
                p = RtnPredictor(args.rtn_path,
                                 args.use_rtn_weights,
                                 args.normalize_rtn_weights,
                                 to_log=args.fst_to_log,
                                 minimize_rtns=args.minimize_rtns,
                                 rmeps=args.remove_epsilon_in_rtns)
            elif pred == "srilm":
                p = SRILMPredictor(args.srilm_path, args.srilm_order)
            elif pred == "nplm":
                p = NPLMPredictor(args.nplm_path, args.normalize_nplm_probs)
            elif pred == "rnnlm":
                p = TensorFlowRNNLMPredictor(args.rnnlm_path, args.rnnlm_config)
            elif pred == "lstm":
                p = ChainerLstmPredictor(args.lstm_path)
            elif pred == "wc":
                p = WordCountPredictor(args.wc_word)
            elif pred == "ngramc":
                p = NgramCountPredictor(_get_override_args("ngramc_path"),
                                        _get_override_args("ngramc_order"))
            elif pred == "unkc":
                p = UnkCountPredictor(
                         args.src_vocab_size, 
                         [float(l) for l in args.unk_count_lambdas.split(',')])
            elif pred == "length":
                length_model_weights = [float(w) for w in 
                                            args.length_model_weights.split(',')]
                p = NBLengthPredictor(args.src_test_raw, 
                                      length_model_weights, 
                                      args.use_length_point_probs,
                                      args.length_model_offset)
            elif pred == "extlength":
                p = ExternalLengthPredictor(args.extlength_path)
            elif pred == "lrhiero":
                fw = None
                if args.grammar_feature_weights:
                    fw = [float(w) for w in 
                            args.grammar_feature_weights.split(',')]
                p = RuleXtractPredictor(args.rules_path,
                                        args.use_grammar_weights,
                                        fw)
            elif pred == "vanilla":
                continue
            else:
                logging.fatal("Predictor '%s' not available. Please check "
                              "--predictors for spelling errors." % pred)
                decoder.remove_predictors()
                return
            for wrapper_idx,wrapper in enumerate(wrappers):
                # Embed predictor ``p`` into wrapper predictors if necessary
                # TODO: Use wrapper_weights
                if wrapper == "idxmap":
                    src_path = _get_override_args("src_idxmap")
                    trg_path = _get_override_args("trg_idxmap")
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedIdxmapPredictor(src_path, trg_path, p, 1.0) 
                    else: # idxmap predictor for bounded predictors
                        p = IdxmapPredictor(src_path, trg_path, p, 1.0)
                elif wrapper == "altsrc":
                    src_test = _get_override_args("altsrc_test")
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedAltsrcPredictor(src_test, p)
                    else: # altsrc predictor for bounded predictors
                        p = AltsrcPredictor(src_test, p)
                elif wrapper == "word2char":
                    map_path = _get_override_args("word2char_map")
                    # word2char is always unbounded predictors
                    p = Word2charPredictor(map_path, p)
                else:
                    logging.fatal("Predictor wrapper '%s' not available. "
                                  "Please double-check --predictors for "
                                  "spelling errors." % wrapper)
                    decoder.remove_predictors()
                    return
            decoder.add_predictor(pred, p, pred_weight)
            logging.info("Added predictor {} with weight {}".format(pred, pred_weight))
    except IOError as e:
        logging.fatal("One of the files required for setting up the "
                      "predictors could not be read: %s" % e)
        decoder.remove_predictors()
    except NameError as e:
        logging.fatal("Could not find external library: %s. Please make sure "
                      "that your PYTHONPATH and LD_LIBRARY_PATH contains all "
                      "paths required for the predictors." % e)
        decoder.remove_predictors()
    except ValueError as e:
        logging.fatal("A number format error occurred while configuring the "
                      "predictors: %s. Please double-check all integer- or "
                      "float-valued parameters such as --predictor_weights and"
                      " try again." % e)
        decoder.remove_predictors()
    except Exception as e:
        logging.fatal("An unexpected %s has occurred while setting up the pre"
                      "dictors: %s Stack trace: %s" % (sys.exc_info()[0],
                                                       e,
                                                       traceback.format_exc()))
        decoder.remove_predictors()


def create_decoder(nmt_config):
    """Creates the ``Decoder`` instance. This specifies the search 
    strategy used to traverse the space spanned by the predictors. This
    method relies on the global ``args`` variable.
    
    TODO: Refactor to avoid long argument lists
    
    Args:
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        Decoder. Instance of the search strategy
    """
    # Configure closed vocabulary normalization
    closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE
    if args.closed_vocabulary_normalization == 'exact':
        closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_EXACT
    elif args.closed_vocabulary_normalization == 'reduced':
        closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_REDUCED
    elif args.closed_vocabulary_normalization == 'rescale_unk':
        closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK
        
    # Create decoder instance and add predictors
    if args.decoder == "greedy":
        decoder = GreedyDecoder(closed_vocab_norm, args.max_len_factor)
    elif args.decoder == "beam":
        decoder = BeamDecoder(closed_vocab_norm,
                              args.max_len_factor,
                              args.hypo_recombination,
                              args.beam,
                              args.pure_heuristic_scores,
                              args.decoder_diversity_factor,
                              args.early_stopping)
    elif args.decoder == "dfs":
        decoder = DFSDecoder(closed_vocab_norm,
                             args.max_len_factor, 
                             args.early_stopping,
                             args.score_lower_bounds_file,
                             args.max_node_expansions)
    elif args.decoder == "restarting":
        decoder = RestartingDecoder(closed_vocab_norm,
                                    args.max_len_factor,
                                    args.hypo_recombination,
                                    args.max_node_expansions,
                                    args.score_lower_bounds_file,
                                    args.low_decoder_memory,
                                    args.restarting_node_score,
                                    args.stochastic_decoder,
                                    args.decode_always_single_step)
    elif args.decoder == "bow":
        decoder = BOWDecoder(closed_vocab_norm,
                             args.hypo_recombination,
                             args.max_node_expansions,
                             args.stochastic_decoder,
                             args.early_stopping,
                             args.score_lower_bounds_file,
                             args.decode_always_single_step)
    elif args.decoder == "flip":
        decoder = FlipDecoder(closed_vocab_norm,
                              args.trg_test,
                              args.max_node_expansions,
                              args.early_stopping,
                              args.score_lower_bounds_file,
                              args.flip_strategy)
    elif args.decoder == "bigramgreedy":
        decoder = BigramGreedyDecoder(closed_vocab_norm,
                                      args.trg_test,
                                      args.max_node_expansions,
                                      args.early_stopping,
                                      args.score_lower_bounds_file)
    elif args.decoder == "bucket":
        decoder = BucketDecoder(closed_vocab_norm,
                                args.max_len_factor, 
                                args.hypo_recombination,
                                args.max_node_expansions,
                                args.low_decoder_memory,
                                args.beam,
                                args.pure_heuristic_scores,
                                args.decoder_diversity_factor,
                                args.early_stopping,
                                args.score_lower_bounds_file,
                                args.stochastic_decoder,
                                args.bucket_selector,
                                args.bucket_score_strategy,
                                args.collect_statistics)
    elif args.decoder == "astar":
        decoder = AstarDecoder(closed_vocab_norm, 
                               args.max_len_factor, 
                               args.beam,
                               args.pure_heuristic_scores,
                               args.early_stopping,
                               args.score_lower_bounds_file,
                               max(1, args.nbest))
    elif args.decoder == "vanilla":
        decoder = get_nmt_vanilla_decoder(args, nmt_config)
        args.predictors = "vanilla"
    else:
        logging.fatal("Decoder %s not available. Please double-check the "
                      "--decoder parameter." % args.decoder)
    add_predictors(decoder, nmt_config)
    
    # Add heuristics for search strategies like A*
    if args.heuristics:
        add_heuristics(decoder, closed_vocab_norm, args.max_len_factor)
    
    # Update start sentence id if necessary
    if args.range:
        idx,_ = args.range.split(":") if (":" in args.range) else (args.range,0)  
        decoder.set_start_sen_id(int(idx)-1) # -1 because indices start with 1
    return decoder


def add_heuristics(decoder, closed_vocab_norm, max_len_factor):
    """Adds all enabled heuristics to the ``decoder``. This is relevant
    for heuristic based search strategies like A*. This method relies 
    on the global ``args`` variable and reads out ``args.heuristics``.
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add heuristics to this instance with
            ``add_heuristic()``
        closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
        max_len_factor (int): Hypotheses are not longer than
                                  source sentence length times this.
                                  Needs to be supported by the search
                                  strategy implementation
    """
    if args.heuristic_predictors == 'all':
        h_predictors = decoder.predictors
    else:
        h_predictors = [decoder.predictors[int(idx)]
                            for idx in args.heuristic_predictors.split(",")]
    decoder.set_heuristic_predictors(h_predictors)
    for name in args.heuristics.split(","):
        if name == 'greedy':
            decoder.add_heuristic(GreedyHeuristic(
                                        closed_vocab_norm, 
                                        max_len_factor,
                                        args.cache_heuristic_estimates))
        elif name == 'predictor':
            decoder.add_heuristic(PredictorHeuristic())
        elif name == 'stats':
            decoder.add_heuristic(StatsHeuristic(args.heuristic_scores_file,
                                                 args.collect_statistics))
        elif name == 'scoreperword':
            decoder.add_heuristic(ScorePerWordHeuristic())
        else:
            logging.fatal("Heuristic %s not available. Please double-check "
                          "the --heuristics parameter." % name)


def create_output_handlers(nmt_config):
    """Creates the output handlers defined in the ``io`` module. 
    These handlers create output files in different formats from the
    decoding results. This method reads out the global variable
    ``args.outputs``.
    
    Args:
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        list. List of output handlers according --outputs
    """
    if not args.outputs:
        return []
    outputs = []
    start_sen_id = 0
    if args.range:
        idx,_ = args.range.split(":")
        start_sen_id = int(idx)-1 # -1 because --range indices start with 1
    for name in args.outputs.split(","):
        if '%s' in args.output_path:
            path = args.output_path % name
        else:
            path = args.output_path
        if name == "text":
            outputs.append(TextOutputHandler(path))
        elif name == "nbest":
            outputs.append(NBestOutputHandler(path, args.predictors.split(","),
                                              start_sen_id))
        elif name == "fst":
            outputs.append(FSTOutputHandler(path,
                                            start_sen_id,
                                            args.output_fst_unk_id))
        elif name == "sfst":
            outputs.append(StandardFSTOutputHandler(path,
                                                    start_sen_id,
                                                    args.output_fst_unk_id))
        else:
            logging.fatal("Output format %s not available. Please double-check"
                          " the --outputs parameter." % name)
    return outputs


def update_decoder(decoder, key, val, nmt_config):
    """This method is called on a configuration update in an interactive 
    (stdin or shell) mode. It tries to update the decoder such that it 
    realizes the new configuration specified by key and val without
    rebuilding the decoder. This can save time because rebuilding the
    decoder involves reloading the predictors which can be expensive
    (e.g. the NMT predictor would reload the model). If an update on
    ``key`` cannot be realized efficiently, rebuild the whole decoder.
    
    Args:
        decoder (Decoder):  Current decoder instance
        key (string):  Parameter name to update
        val (string):  New parameter value
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        Decoder. Returns an updated decoder instance
    """
    if key == 'beam':
        decoder.beam_size = int(val)
        args.beam = int(val)
    elif key == 'nbest':
        args.nbest = int(val)
    elif key == 'range':
        idx,_ = args.range.split(":")
        decoder.set_start_sen_id(int(idx)-1) # -1 because indices start with 1
    elif key == 'predictor_weights' and val:
        logging.debug("Set predictor weights to %s on the fly" % val)
        for idx,weight in enumerate(val.split(',')):
            if '_' in weight: # wrapper predictor
                wrapper_weights = [float(w) for w in weight.split('_')]
                slave_pred = decoder.predictors[idx][0]
                for i in xrange(len(wrapper_weights)-1):
                    slave_pred.my_weight = wrapper_weights[i]
                    slave_pred = slave_pred.slave_predictor
                slave_pred.slave_weight = wrapper_weights[-1]
            else: # normal predictor (not wrapped)
                decoder.predictors[idx] = (decoder.predictors[idx][0],
                                           float(weight))
    else:
        logging.info("Need to rebuild the decoder from scratch...")
        decoder = create_decoder(nmt_config)
    return decoder


def _get_sentence_indices(range_param, src_sentences):
    """Helper method for ``do_decode`` which returns the indices of the
    sentence to decode
    
    Args:
        range_param (string): ``--range`` parameter from config
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    if args.range:
        try:
            if ":" in args.range:
                from_idx,to_idx = args.range.split(":")
            else:
                from_idx = int(args.range)
                to_idx = from_idx
            return xrange(int(from_idx)-1, int(to_idx))
        except Exception as e:
            logging.fatal("Invalid value for --range: %s" % e)
            return []
    if src_sentences is False:
        logging.fatal("Input method dummy requires --range")
        return []
    return xrange(len(src_sentences))

def get_text_output_handler(output_handlers):
    for output_handler in output_handlers:
        if isinstance(output_handler, TextOutputHandler):
            return output_handler
    return None

def do_decode(decoder, output_handlers, src_sentences):
    """This method contains the main decoding loop. It iterates through
    ``src_sentences`` and applies ``decoder.decode()`` to each of them.
    At the end, it calls the output handlers to create output files.
    
    Args:
        decoder (Decoder):  Current decoder instance
        output_handlers (list):  List of output handlers, see
                                 ``create_output_handlers()``
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    if not decoder.has_predictors():
        logging.fatal("Decoding cancelled because of an error in the "
                      "predictor configuration.")
        return
    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    all_hypos = []
    text_output_handler = get_text_output_handler(output_handlers)
    if text_output_handler:
        text_output_handler.open_file()
    for sen_idx in _get_sentence_indices(args.range, src_sentences):
        try:
            if src_sentences is False:
                src = "0"
                logging.info("Next sentence (ID: %d)" % (sen_idx+1))
            else:
                src = src_sentences[sen_idx]
                if isinstance(src[0], list):
                    src_lst = []
                    for idx in xrange(len(src)):
                        logging.info("Next sentence, input %d (ID: %d): %s" % (idx, sen_idx+1, ' '.join(src[idx])))
                        src_lst.append([int(x) for x in src[idx]])
                    src = src_lst
                else:
                    logging.info("Next sentence (ID: %d): %s" % (sen_idx+1, ' '.join(src)))
                    src = [int(x) for x in src]
            start_hypo_time = time.time()
            decoder.apply_predictors_count = 0
            hypos = [hypo for hypo in decoder.decode(src)
                                if hypo.total_score > args.min_score]
            if not hypos:
                logging.error("No translation found for ID %d!" % (sen_idx+1))
                logging.info("Stats (ID: %d): score=<not-found> "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time))
                if text_output_handler:
                    text_output_handler.write_empty_line()
                continue
            if args.remove_eos:
                for hypo in hypos:
                    if (hypo.trgt_sentence 
                            and hypo.trgt_sentence[-1] == utils.EOS_ID):
                        hypo.trgt_sentence = hypo.trgt_sentence[:-1]
            if args.nbest > 0:
                hypos = hypos[:args.nbest]
            if (args.combination_scheme != 'sum' 
                    and not args.apply_combination_scheme_to_partial_hypos):
                for hypo in hypos:
                    hypo.total_score = core.breakdown2score_full(
                                                        hypo.total_score,
                                                        hypo.score_breakdown)
                hypos.sort(key=lambda hypo: hypo.total_score, reverse=True)
            logging.info("Decoded (ID: %d): %s" % (
                            sen_idx+1,
                            ' '.join(str(w) for w in hypos[0].trgt_sentence)))
            logging.info("Stats (ID: %d): score=%f "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        hypos[0].total_score,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time))
            all_hypos.append(hypos)
            try:
              # Write text output as we go
              if text_output_handler:
                  if args.trg_wmap:
                      logging.info("Target word map={}".format(args.trg_wmap))
                      text_output_handler.write_hypos([hypos], args.trg_wmap)
                  else:
                      text_output_handler.write_hypos([hypos])
            except IOError as e:
              logging.error("I/O error %d occurred when creating output files: %s"
                            % (sys.exc_info()[0], e))
        except ValueError as e:
            logging.error("Number format error at sentence id %d: %s"
                      % (sen_idx+1, e))
        except Exception as e:
            logging.error("An unexpected %s error has occurred at sentence id "
                          "%d: %s, Stack trace: %s" % (sys.exc_info()[0],
                                                       sen_idx+1,
                                                       e,
                                                       traceback.format_exc()))
    try:
        for output_handler in output_handlers:
            if output_handler == text_output_handler:
                output_handler.close_file()
            else:
                output_handler.write_hypos(all_hypos)
    except IOError as e:
        logging.error("I/O error %s occurred when creating output files: %s"
                      % (sys.exc_info()[0], e))
    logging.info("Decoding finished. Time: %.2f" % (time.time() - start_time))


def _print_shell_help():
    """Print help text for shell usage in interactive mode."""
    print("Available SGNMT directives:")
    print("!sgnmt config <name> <value>  Update the configuration. Some changes")
    print("                             may require loading the decoder from ")
    print("                             scratch, some (like changing predictor")
    print("                             weights) can be done on the fly. For ")
    print("                             printing help text for all available")
    print("                             parameters use")
    print("                               !sgnmt config (without arguments)")
    print("!sgnmt decode <file_name>     Decode sentences in the given file")
    print("!sgnmt reset                  Reset predictors, e.g. set sentence")
    print("                             counter to 1 for fst predictor.")
    print("!sgnmt quit                   Quit SGNMT")
    print("!sgnmt help                   Print this help")


# THIS IS THE MAIN ENTRY POINT
nmt_config = get_nmt_config(args)
decoder = create_decoder(nmt_config)
outputs = create_output_handlers(nmt_config)

if args.input_method == 'file':
    inputfiles = args.src_test.split(',')
    if len(inputfiles) > 1:
        logging.info("Found multiple input files")
        inputs_tmp = [ [] for i in xrange(len(inputfiles)) ]
        for i in xrange(len(inputfiles)):
            with open(inputfiles[i]) as f:
                for line in f:
                    inputs_tmp[i].append(line.strip().split())
        inputs = []
        for i in xrange(len(inputs_tmp[0])):
            # Gather multiple input sentences for each line
            input_lst = []
            for j in xrange(len(inputfiles)):
                input_lst.append(inputs_tmp[j][i])
            inputs.append(input_lst)
        do_decode(decoder, outputs, inputs)
    else:
      with open(args.src_test) as f:
        do_decode(decoder, outputs, [line.strip().split() for line in f])
elif args.input_method == 'dummy':
    do_decode(decoder, outputs, False)
else: # Interactive mode: shell or stdin
    print("Start interactive mode.")
    print("PID: %d" % os.getpid())
    print("Test sentences are read directly from stdin.")
    print("!sgnmt help lists all available directives")
    print("Quit with ctrl-c or !sgnmt quit")
    quit_gnmt = False
    sys.stdout.flush()
    while not quit_gnmt:
        # Read input from stdin or keyboard
        if args.input_method == 'shell':
            input_ = raw_input("gnmt> ")
        else: # stdin input method
            input_ = sys.stdin.readline()
            if not input_:
                break
            logging.debug("Process input line: %s" % input_.strip())
        input_ = input_.strip().split()
        
        try:
            if input_[0] == "!sgnmt": # SGNMT directives
                cmd = input_[1]
                if cmd == "help":
                    _print_shell_help()
                elif cmd == "reset":
                    decoder.reset_predictors()
                elif cmd == "decode":
                    decoder.reset_predictors()
                    with open(input_[2]) as f:
                        do_decode(decoder,
                                  outputs,
                                  [line.strip().split() for line in f])
                elif cmd == "quit":
                    quit_gnmt = True
                elif cmd == "config":
                    if len(input_) == 2:
                        get_parser().print_help()
                    elif len(input_) >= 4:
                        key,val = (input_[2], ' '.join(input_[3:]))
                        setattr(args, key, val) # TODO: non-string args!
                        nmt_config = get_nmt_config(args)
                        outputs = create_output_handlers(nmt_config)
                        if not key in ['outputs', 'output_path']:
                            decoder = update_decoder(decoder,
                                                     key, val,
                                                     nmt_config)
                    else:
                        logging.error("Could not parse SGNMT directive")
                else:
                    logging.error("Unknown directive '%s'. Use '!sgnmt help' "
                                  "for help or exit with '!sgnmt quit'" % cmd)
            elif input_[0] == 'quit' or input_[0] == 'exit':
                quit_gnmt = True
            else: # Sentence to translate
                do_decode(decoder, outputs, [input_])
        except:
            logging.error("Error in last statement: %s" % sys.exc_info()[0])
        sys.stdout.flush()
