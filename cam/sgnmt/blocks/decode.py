"""This is the main runner script for SGNMT decoding based on blocks. 
GNMT can run in three different modes. The standard mode 'file' reads
sentences to translate from a plain text file. The mode 'stdin' can be
used to parse stdin. The last mode 'shell' enables interactive inter-
action with SGNMT via keyboard. Note that SGNMT exclusively operates on 
integer representations of words, the mapping between word ids and
their string representations needs to happen outside GNMT. For detailed
usage descriptions please visit the tutorial home page:

http://ucam-smt.github.io/tutorial/sgnmt

Note that the configuration is handled through this module, the ``ui``
module, and ``machine_translation.configurations``. All other modules
do not handle (textual) parameters directly. 
"""

import logging
import os
import pprint
import sys
import time
import readline

from cam.sgnmt import utils
from cam.sgnmt.blocks.machine_translation import configurations
from cam.sgnmt.blocks.vanilla_decoder import BlocksVanillaDecoder
from cam.sgnmt.decoding.core import CLOSED_VOCAB_SCORE_NORM_NONE, \
                                   CLOSED_VOCAB_SCORE_NORM_EXACT, \
                                   CLOSED_VOCAB_SCORE_NORM_REDUCED
from cam.sgnmt.decoding.core import UnboundedVocabularyPredictor
from cam.sgnmt.decoding import core
from cam.sgnmt.decoding.decoder import GreedyDecoder, \
                                      BeamDecoder, \
                                      DFSDecoder, \
                                      AstarDecoder, \
                                      RestartingDecoder
from cam.sgnmt.decoding.heuristics import GreedyHeuristic, \
                                         PredictorHeuristic, \
                                         ScorePerWordHeuristic
from cam.sgnmt.io import TextOutputHandler, \
                        NBestOutputHandler, \
                        FSTOutputHandler, \
                        StandardFSTOutputHandler
from cam.sgnmt.predictors import blocks_neural
from cam.sgnmt.predictors.automata import FstPredictor, \
                                         RtnPredictor, \
                                         NondeterministicFstPredictor
from cam.sgnmt.predictors.blocks_neural import NMTPredictor
from cam.sgnmt.predictors.forced import ForcedPredictor, ForcedLstPredictor
from cam.sgnmt.predictors.grammar import RuleXtractPredictor
from cam.sgnmt.predictors.length import WordCountPredictor,NBLengthPredictor
from cam.sgnmt.predictors.misc import IdxmapPredictor, UnboundedIdxmapPredictor
from cam.sgnmt.predictors.ngram import SRILMPredictor
from cam.sgnmt.predictors.nnlm import NPLMPredictor
from cam.sgnmt.blocks.ui import get_args, validate_args

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
    nmt_config = getattr(configurations, args.proto)()
    for k in dir(args):
        if k in nmt_config:
            nmt_config[k] = getattr(args, k)
    logger.debug("Model options:\n{}".format(pprint.pformat(nmt_config)))
    return nmt_config


def _get_override_args(field, count):
    """This is a helper function for arguments which can be overridden 
    when using multiple instances of the same predictor. E.g., it is 
    possible to use two fst predictors, and specify the path to the 
    first directory with --fst_path, and the path to the second 
    directory with --fst_path2.
    
    Args:
        field (string):  Field in ``args`` for which also field2, 
                         field3... is also possible
        count (int): The number of the current predictor instance
    
    Returns:
        object. If count smaller than 2, returns ``args.field``. 
        Otherwise, returns ``args.field+count`` if specified, or backup
        to ``args.field``
    """
    default = getattr(args, field)
    if count < 2:
        return default
    override_value = getattr(args, field + "%d" % count)
    return override_value if override_value else default


def add_predictors(decoder, nmt_config):
    """Adds all enabled predictors to the ``decoder``. This function 
    makes heavy use of the global ``args`` function which contains the
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
    
    pred_weight = 1.0
    nmt_pred_count = 0
    idxmap_pred_count = 0
    fst_pred_count = 0
    forcedlst_pred_count = 0
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
                    for pair in add_config.split(","):
                        (k,v) = pair.split("=", 1)
                        nmt_config[k] = type(nmt_config[k])(v)
                nmt_model_path = get_nmt_model_path(nmt_config)
            if pred == 'fst' or pred == 'nfst':
                # Update FST config in case --fst_pathX has been used
                fst_pred_count += 1
                fst_path = _get_override_args("fst_path", fst_pred_count)
            
            # Create predictor instances for the string argument ``pred``
            if pred == "nmt":
                p = NMTPredictor(nmt_model_path,
                                 args.cache_nmt_posteriors,
                                 nmt_config)
            elif pred == "fst":
                p = FstPredictor(fst_path,
                                 args.use_fst_weights,
                                 args.normalize_fst_weights,
                                 add_bos_to_eos_score=args.add_fst_bos_to_eos_weight,
                                 to_log=args.fst_to_log)
            elif pred == "nfst":
                p = NondeterministicFstPredictor(fst_path,
                                                 args.use_fst_weights,
                                                 args.normalize_fst_weights,
                                                 to_log=args.fst_to_log)
            elif pred == "forced":
                p = ForcedPredictor(args.trg_test)
            elif pred == "forcedlst":
                forcedlst_pred_count += 1
                feat_name = _get_override_args("forcedlst_sparse_feat",
                                               forcedlst_pred_count)
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
            elif pred == "wc":
                p = WordCountPredictor()
            elif pred == "length":
                length_model_weights = [float(w) for w in 
                                            args.length_model_weights.split(',')]
                p = NBLengthPredictor(args.src_test_raw, 
                                      length_model_weights, 
                                      args.use_length_point_probs)
            elif pred == "lrhiero":
                fw = None
                if args.grammar_feature_weights:
                    fw = [float(w) for w in 
                            args.grammar_feature_weights.split(',')]
                p = RuleXtractPredictor(args.rules_path, args.use_grammar_weights, fw)
            elif pred == "vanilla":
                continue
            else:
                logging.fatal("Predictor '%s' not available. Please check "
                              "--predictors for spelling errors." % pred)
            for wrapper_idx,wrapper in enumerate(wrappers):
                # Embed predictor ``p`` into wrapper predictors if necessary
                # TODO: Use wrapper_weights (okay for now because idxmap only)
                if wrapper == "idxmap":
                    idxmap_pred_count += 1
                    src_path = _get_override_args("src_idxmap",
                                                  idxmap_pred_count)
                    trg_path = _get_override_args("trg_idxmap",
                                                  idxmap_pred_count)
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedIdxmapPredictor(src_path, trg_path, p, 1.0) 
                    else: # idxmap predictor for bounded predictors
                        p = IdxmapPredictor(src_path, trg_path, p, 1.0)
                else:
                    logging.fatal("Predictor wrapper '%s' not available. "
                                  "Please double-check --predictors for "
                                  "spelling errors." % wrapper)
            decoder.add_predictor(pred, p, pred_weight)
    except ValueError as e:
        logging.fatal("A number format error (%d) while configuring the "
                      "predictors: %s. Please double-check all integer- or "
                      "float-valued parameters such as --predictor_weights and "
                      "try again." % (e.errno, e.strerror))
    except:
        logging.fatal("An unexpected error occurred while setting up the "
                      "predictors: %s" % sys.exc_info()[0])


def get_nmt_model_path(nmt_config):
    """Get the path to the NMT model according the given NMT config.
    This switches between the most recent checkpoint, the best BLEU 
    checkpoint, or the latest parameters (params.npz). This method
    delegates to ``blocks_neural.get_nmt_model_path*``. This
    method relies on the global ``args`` variable.
    
    Args:
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        string. Path to the NMT model file
    """
    if args.nmt_model_selector == 'params':
        return blocks_neural.get_nmt_model_path_params(nmt_config)
    elif args.nmt_model_selector == 'bleu':
        return blocks_neural.get_nmt_model_path_best_bleu(nmt_config)
    elif args.nmt_model_selector == 'time':
        return blocks_neural.get_nmt_model_path_most_recent(nmt_config)
    logging.fatal("NMT model selector %s not available. Please double-check "
                  "the --nmt_model_selector parameter." % args.nmt_model_selector)


def create_decoder(nmt_config):
    """Creates the ``Decoder`` instance. This specifies the search 
    strategy used to traverse the space spanned by the predictors. This
    method relies on the global ``args`` variable.
    
    Args:
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    
    Returns:
        Decoder. Path to the NMT model file
    """
    # Configure closed vocabulary normalization
    closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE
    if args.closed_vocabulary_normalization == 'exact':
        closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_EXACT
    elif args.closed_vocabulary_normalization == 'reduced':
        closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_REDUCED
        
    # Create decoder instance and add predictors
    if args.decoder == "greedy":
        decoder = GreedyDecoder(closed_vocab_norm)
    elif args.decoder == "beam":
        decoder = BeamDecoder(closed_vocab_norm, args.beam,
                              args.early_stopping)
    elif args.decoder == "dfs":
        decoder = DFSDecoder(closed_vocab_norm, args.early_stopping,
                             args.max_node_expansions)
    elif args.decoder == "restarting":
        decoder = RestartingDecoder(closed_vocab_norm,
                                    args.max_node_expansions)
    elif args.decoder == "astar":
        decoder = AstarDecoder(closed_vocab_norm, args.beam,
                               max(1, args.nbest))
    elif args.decoder == "vanilla":
        decoder = BlocksVanillaDecoder(get_nmt_model_path(nmt_config),
                                       nmt_config)
        args.predictors = "vanilla"
    else:
        logging.fatal("Decoder %s not available. Please double-check the "
                      "--decoder parameter." % args.decoder)
    add_predictors(decoder, nmt_config)
    
    # Add heuristics for search strategies like A*
    if args.heuristics:
        decoder = add_heuristics(decoder, closed_vocab_norm)
    
    # Update start sentence id if necessary
    if args.range:
        idx,_ = args.range.split(":")
        decoder.set_start_sen_id(int(idx)-1) # -1 because indices start with 1
    return decoder


def add_heuristics(decoder, closed_vocab_norm):
    """Adds all enabled heuristics to the ``decoder``. This is relevant
    for heuristic based search strategies like A*. This method relies 
    on the global ``args`` variable and reads out ``args.heuristics``.
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add heuristics to this instance with
            ``add_heuristic()``
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
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
                                        args.cache_heuristic_estimates))
        elif name == 'predictor':
            decoder.add_heuristic(PredictorHeuristic())
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
            outputs.append(FSTOutputHandler(path, start_sen_id))
        elif name == "sfst":
            outputs.append(StandardFSTOutputHandler(path, start_sen_id))
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
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
    """
    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    all_hypos = []
    sen_indices = xrange(len(src_sentences))
    if args.range:
        from_idx,to_idx = args.range.split(":")
        sen_indices = xrange(int(from_idx)-1, int(to_idx))
    for sen_idx in sen_indices:
        try:
            src = src_sentences[sen_idx]
            logging.info("Next sentence (ID: %d): %s" % (sen_idx+1, 
                                                         ' '.join(src)))
            start_hypo_time = time.time()
            decoder.apply_predictors_count = 0
            hypos = [hypo for hypo in decoder.decode([int(x) for x in src])
                                if hypo.total_score > args.min_score]
            if not hypos:
                logging.error("No translation found for ID %d!" % (sen_idx+1))
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
        except ValueError as e:
            logging.error("Number format error (%d) at sentence id %d: %s"
                      % (e.errno, sen_idx+1, e.strerror))
        except:
            logging.error("An unexpected error occurred at sentence id %d: %s"
                          % (sen_idx+1, sys.exc_info()[0]))
    try:
        for output_handler in  output_handlers:
            output_handler.write_hypos(all_hypos)
    except IOError as e:
        logging.error("I/O error %d occurred when creating output files: %s"
                      % (e.errno, e.strerror))
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
    print("!sgnmt quit                   Quit GNMT")
    print("!sgnmt help                   Print this help")


# THIS IS THE MAIN ENTRY POINT
nmt_config = get_nmt_config(args)
decoder = create_decoder(nmt_config)
outputs = create_output_handlers(nmt_config)

if args.input_method == 'file':
    with open(args.src_test) as f:
        do_decode(decoder, outputs, [line.strip().split() for line in f])
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
                        parser.print_help()
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
