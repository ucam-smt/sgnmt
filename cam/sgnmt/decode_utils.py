# -*- coding: utf-8 -*-
"""This module is the bridge between the command line configuration of
the decode.py script and the SGNMT software architecture consisting of
decoders, predictors, and output handlers. A common use case is to call
`create_decoder()` first, which reads the SGNMT configuration and loads
the right predictors and decoding strategy with the right arguments.
The actual decoding is implemented in `do_decode()`. See `decode.py`
to learn how to use this module.
"""

import logging
import codecs
import sys
import time
import traceback
import os
import uuid

from cam.sgnmt import ui
from cam.sgnmt import utils
from cam.sgnmt.blocks.nmt import blocks_get_nmt_predictor, \
                                 blocks_get_nmt_vanilla_decoder, \
                                 blocks_get_default_nmt_config
from cam.sgnmt.predictors.parse import ParsePredictor, TokParsePredictor, \
                                       BpeParsePredictor
from cam.sgnmt.decoding import combination
from cam.sgnmt.decoding.astar import AstarDecoder
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.bigramgreedy import BigramGreedyDecoder
from cam.sgnmt.decoding.bow import BOWDecoder
from cam.sgnmt.decoding.bucket import BucketDecoder
from cam.sgnmt.decoding.core import UnboundedVocabularyPredictor
from cam.sgnmt.decoding.core import Hypothesis
from cam.sgnmt.decoding.dfs import DFSDecoder
from cam.sgnmt.decoding.flip import FlipDecoder
from cam.sgnmt.decoding.greedy import GreedyDecoder
from cam.sgnmt.decoding.heuristics import GreedyHeuristic, \
                                         PredictorHeuristic, \
                                         ScorePerWordHeuristic, \
                                         StatsHeuristic, \
                                         LastTokenHeuristic
from cam.sgnmt.decoding.multisegbeam import MultisegBeamDecoder
from cam.sgnmt.decoding.restarting import RestartingDecoder
from cam.sgnmt.decoding.sepbeam import SepBeamDecoder
from cam.sgnmt.decoding.syntaxbeam import SyntaxBeamDecoder
from cam.sgnmt.decoding.mbrbeam import MBRBeamDecoder
from cam.sgnmt.decoding.syncbeam import SyncBeamDecoder
from cam.sgnmt.decoding.combibeam import CombiBeamDecoder
from cam.sgnmt.output import TextOutputHandler, \
                             NBestOutputHandler, \
                             NgramOutputHandler, \
                             TimeCSVOutputHandler, \
                             FSTOutputHandler, \
                             StandardFSTOutputHandler
from cam.sgnmt.predictors.automata import FstPredictor, \
                                         RtnPredictor, \
                                         NondeterministicFstPredictor
from cam.sgnmt.predictors.bow import BagOfWordsPredictor, \
                                     BagOfWordsSearchPredictor
from cam.sgnmt.predictors.ffnnlm import NPLMPredictor
from cam.sgnmt.predictors.forced import ForcedPredictor, ForcedLstPredictor
from cam.sgnmt.predictors.grammar import RuleXtractPredictor
from cam.sgnmt.predictors.length import WordCountPredictor, NBLengthPredictor, \
    ExternalLengthPredictor, NgramCountPredictor, UnkCountPredictor, \
    NgramizePredictor, WeightNonTerminalPredictor
from cam.sgnmt.predictors.structure import BracketPredictor, OSMPredictor, \
                                           ForcedOSMPredictor
from cam.sgnmt.predictors.misc import UnboundedAltsrcPredictor, \
                                      AltsrcPredictor, \
                                      UnboundedRankPredictor, \
                                      RankPredictor, \
                                      UnboundedGluePredictor, \
                                      GluePredictor
from cam.sgnmt.predictors.vocabulary import IdxmapPredictor, \
                                            UnboundedIdxmapPredictor, \
                                            MaskvocabPredictor, \
                                            UnboundedMaskvocabPredictor, \
                                            UnkvocabPredictor, \
                                            SkipvocabPredictor
from cam.sgnmt.predictors.ngram import SRILMPredictor, KenLMPredictor
from cam.sgnmt.predictors.tf_t2t import T2TPredictor, \
                                        EditT2TPredictor, \
                                        BosLimitT2TPredictor, \
                                        FertilityT2TPredictor
from cam.sgnmt.predictors.tf_nizza import NizzaPredictor, LexNizzaPredictor
from cam.sgnmt.predictors.tokenization import Word2charPredictor, FSTTokPredictor
from cam.sgnmt.tf.interface import tf_get_nmt_predictor, tf_get_nmt_vanilla_decoder, \
    tf_get_rnnlm_predictor, tf_get_default_nmt_config, tf_get_rnnlm_prefix


args = None
"""This variable is set to the global configuration when 
base_init().
"""

def base_init(new_args):
    """This function should be called before accessing any other
    function in this module. It initializes the `args` variable on 
    which all the create_* factory functions rely on as configuration
    object, and it sets up global function pointers and variables for
    basic things like the indexing scheme, logging verbosity, etc.

    Args:
        new_args: Configuration object from the argument parser.
    """
    global args
    args = new_args
    # UTF-8 support
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        logging.warn("SGNMT is tested with Python 2.7, but you are using "
                     "Python 3. Expect the unexpected or switch to 2.7.")
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
    # Set reserved word IDs
    if args.indexing_scheme == 'blocks':
        utils.switch_to_blocks_indexing()
    elif args.indexing_scheme == 'tf':
        utils.switch_to_tf_indexing()
    elif args.indexing_scheme == 't2t':
        utils.switch_to_t2t_indexing()
    # Log summation (how to compute log(exp(l1)+exp(l2)) for log values l1,l2)
    if args.log_sum == 'tropical':
        utils.log_sum = utils.log_sum_tropical_semiring
    ui.validate_args(args)


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


def _parse_config_param(field, default):
    """This method parses arguments which specify model configurations.
    It can point directly to a configuration file, or it can contain
    direct settings such as 'param1=x,param2=y'
    """
    add_config = ui.parse_param_string(_get_override_args(field))
    for (k,v) in default.iteritems():
        if k in add_config:
            if type(v) is type(None):
                default[k] = add_config[k]
            else:
                default[k] = type(v)(add_config[k])
        else:
            logging.debug('Unknown key {}, not adding to config'.format(k))
    return default


def add_predictors(decoder):
    """Adds all enabled predictors to the ``decoder``. This function 
    makes heavy use of the global ``args`` which contains the
    SGNMT configuration. Particularly, it reads out ``args.predictors``
    and adds appropriate instances to ``decoder``.
    TODO: Refactor this method as it is waaaay tooooo looong
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add predictors to this instance with
            ``add_predictor()``
    """
    preds = utils.split_comma(args.predictors)
    if not preds:
        logging.fatal("Require at least one predictor! See the --predictors "
                      "argument for more information.")
    weights = None
    if args.predictor_weights:
        weights = utils.split_comma(args.predictor_weights)
        if len(preds) != len(weights):
            logging.fatal("Specified %d predictors, but %d weights. Please "
                      "revise the --predictors and --predictor_weights "
                      "arguments" % (len(preds), len(weights)))
            return
    
    pred_weight = 1.0
    try:
        for idx, pred in enumerate(preds): # Add predictors one by one
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

            # Create predictor instances for the string argument ``pred``
            if pred == "nmt":
                nmt_engine = _get_override_args("nmt_engine")
                if nmt_engine == 'blocks':
                    nmt_config = _parse_config_param(
                        "nmt_config", blocks_get_default_nmt_config())
                    p = blocks_get_nmt_predictor(
                        args, _get_override_args("nmt_path"), nmt_config)
                elif nmt_engine == 'tensorflow':
                    nmt_config = _parse_config_param(
                        "nmt_config", tf_get_default_nmt_config())
                    p = tf_get_nmt_predictor(
                        args, _get_override_args("nmt_path"), nmt_config)
                elif nmt_engine != 'none':
                    logging.fatal("NMT engine %s is not supported (yet)!" % nmt_engine)
            elif pred == "nizza":
                p = NizzaPredictor(_get_override_args("pred_src_vocab_size"),
                                   _get_override_args("pred_trg_vocab_size"),
                                   _get_override_args("nizza_model"),
                                   _get_override_args("nizza_hparams_set"),
                                   _get_override_args("nizza_checkpoint_dir"),
                                   n_cpu_threads=args.n_cpu_threads)
            elif pred == "lexnizza":
                p = LexNizzaPredictor(_get_override_args("pred_src_vocab_size"),
                                      _get_override_args("pred_trg_vocab_size"),
                                      _get_override_args("nizza_model"),
                                      _get_override_args("nizza_hparams_set"),
                                      _get_override_args("nizza_checkpoint_dir"),
                                      n_cpu_threads=args.n_cpu_threads,
                                      alpha=args.lexnizza_alpha,
                                      beta=args.lexnizza_beta,
                                      trg2src_model_name=
                                          args.lexnizza_trg2src_model, 
                                      trg2src_hparams_set_name=
                                          args.lexnizza_trg2src_hparams_set,
                                      trg2src_checkpoint_dir=
                                          args.lexnizza_trg2src_checkpoint_dir,
                                      shortlist_strategies=
                                          args.lexnizza_shortlist_strategies,
                                      max_shortlist_length=
                                          args.lexnizza_max_shortlist_length,
                                      min_id=args.lexnizza_min_id)
            elif pred == "t2t":
                p = T2TPredictor(_get_override_args("pred_src_vocab_size"),
                                 _get_override_args("pred_trg_vocab_size"),
                                 _get_override_args("t2t_model"),
                                 _get_override_args("t2t_problem"),
                                 _get_override_args("t2t_hparams_set"),
                                 args.t2t_usr_dir,
                                 _get_override_args("t2t_checkpoint_dir"),
                                 t2t_unk_id=_get_override_args("t2t_unk_id"),
                                 n_cpu_threads=args.n_cpu_threads,
                                 max_terminal_id=args.syntax_max_terminal_id,
                                 pop_id=args.syntax_pop_id)
            elif pred == "boslimitt2t":
                p = BosLimitT2TPredictor(_get_override_args("pred_src_vocab_size"),
                                 _get_override_args("pred_trg_vocab_size"),
                                 _get_override_args("t2t_model"),
                                 _get_override_args("t2t_problem"),
                                 _get_override_args("t2t_hparams_set"),
                                 args.t2t_usr_dir,
                                 _get_override_args("t2t_checkpoint_dir"),
                                 t2t_unk_id=_get_override_args("t2t_unk_id"),
                                 n_cpu_threads=args.n_cpu_threads,
                                 max_terminal_id=args.syntax_max_terminal_id,
                                 pop_id=args.syntax_pop_id)
            elif pred == "editt2t":
                p = EditT2TPredictor(_get_override_args("pred_src_vocab_size"),
                                     _get_override_args("pred_trg_vocab_size"),
                                     _get_override_args("t2t_model"),
                                     _get_override_args("t2t_problem"),
                                     _get_override_args("t2t_hparams_set"),
                                     args.trg_test,
                                     args.beam,
                                     args.t2t_usr_dir,
                                     _get_override_args("t2t_checkpoint_dir"),
                                     t2t_unk_id=_get_override_args("t2t_unk_id"),
                                     n_cpu_threads=args.n_cpu_threads,
                                     max_terminal_id=args.syntax_max_terminal_id,
                                     pop_id=args.syntax_pop_id)
            elif pred == "fertt2t":
                p = FertilityT2TPredictor(
                                 _get_override_args("pred_src_vocab_size"),
                                 _get_override_args("pred_trg_vocab_size"),
                                 _get_override_args("t2t_model"),
                                 _get_override_args("t2t_problem"),
                                 _get_override_args("t2t_hparams_set"),
                                 args.t2t_usr_dir,
                                 _get_override_args("t2t_checkpoint_dir"),
                                 n_cpu_threasd=args.n_cpu_threads,
                                 max_terminal_id=args.syntax_max_terminal_id,
                                 pop_id=args.syntax_pop_id)
            elif pred == "bracket":
                p = BracketPredictor(args.syntax_max_terminal_id,
                                     args.syntax_pop_id,
                                     max_depth=args.syntax_max_depth,
                                     extlength_path=args.extlength_path)
            elif pred == "osm":
                p = OSMPredictor(args.osm_type)
            elif pred == "forcedosm":
                p = ForcedOSMPredictor(args.trg_test)
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
                                _get_override_args("pred_trg_vocab_size"))
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
                                _get_override_args("pred_trg_vocab_size"))
            elif pred == "forcedlst":
                feat_name = _get_override_args("forcedlst_sparse_feat")
                p = ForcedLstPredictor(args.trg_test,
                                       args.use_nbest_weights,
                                       args.forcedlst_match_unk,
                                       feat_name if feat_name else None)
            elif pred == "rtn":
                p = RtnPredictor(args.rtn_path,
                                 args.use_rtn_weights,
                                 args.normalize_rtn_weights,
                                 to_log=args.fst_to_log,
                                 minimize_rtns=args.minimize_rtns,
                                 rmeps=args.remove_epsilon_in_rtns)
            elif pred == "srilm":
                p = SRILMPredictor(args.lm_path, 
                                   _get_override_args("ngramc_order"),
                                   args.srilm_convert_to_ln)
            elif pred == "kenlm":
                p = KenLMPredictor(args.lm_path)
            elif pred == "nplm":
                p = NPLMPredictor(args.nplm_path, args.normalize_nplm_probs)
            elif pred == "rnnlm":
                p = tf_get_rnnlm_predictor(_get_override_args("rnnlm_path"),
                                           _get_override_args("rnnlm_config"),
                                           tf_get_rnnlm_prefix())
            elif pred == "wc":
                p = WordCountPredictor(args.wc_word,
                                       args.wc_nonterminal_penalty,
                                       args.syntax_nonterminal_ids,
                                       args.syntax_min_terminal_id,
                                       args.syntax_max_terminal_id,
                                       _get_override_args("pred_trg_vocab_size"))
            elif pred == "ngramc":
                p = NgramCountPredictor(_get_override_args("ngramc_path"),
                                        _get_override_args("ngramc_order"),
                                        args.ngramc_discount_factor)
            elif pred == "unkc":
                p = UnkCountPredictor(
                     _get_override_args("pred_src_vocab_size"), 
                     utils.split_comma(args.unk_count_lambdas, float))
            elif pred == "length":
                length_model_weights = utils.split_comma(
                    args.length_model_weights, float)
                p = NBLengthPredictor(args.src_test_raw, 
                                      length_model_weights, 
                                      args.use_length_point_probs,
                                      args.length_model_offset)
            elif pred == "extlength":
                p = ExternalLengthPredictor(args.extlength_path)
            elif pred == "lrhiero":
                fw = None
                if args.grammar_feature_weights:
                    fw = utils.split_comma(args.grammar_feature_weights, float)
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
            for _,wrapper in enumerate(wrappers):
                # Embed predictor ``p`` into wrapper predictors if necessary
                # TODO: Use wrapper_weights
                if wrapper == "idxmap":
                    src_path = _get_override_args("src_idxmap")
                    trg_path = _get_override_args("trg_idxmap")
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedIdxmapPredictor(src_path, trg_path, p, 1.0) 
                    else: # idxmap predictor for bounded predictors
                        p = IdxmapPredictor(src_path, trg_path, p, 1.0)
                elif wrapper == "maskvocab":
                    words = utils.split_comma(args.maskvocab_words, int)
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedMaskvocabPredictor(words, p) 
                    else: # idxmap predictor for bounded predictors
                        p = MaskvocabPredictor(words, p)
                elif wrapper == "weightnt":
                    p = WeightNonTerminalPredictor(
                        p, 
                        args.syntax_nonterminal_factor,
                        args.syntax_nonterminal_ids,
                        args.syntax_min_terminal_id,
                        args.syntax_max_terminal_id,
                        _get_override_args("pred_trg_vocab_size"))
                elif wrapper == "parse":
                    if args.parse_tok_grammar:
                        if args.parse_bpe_path:
                            p = BpeParsePredictor(
                                args.syntax_path,
                                args.syntax_bpe_path,
                                p,
                                args.syntax_word_out,
                                args.normalize_fst_weights,
                                norm_alpha=args.syntax_norm_alpha,
                                beam_size=args.syntax_internal_beam,
                                max_internal_len=args.syntax_max_internal_len,
                                allow_early_eos=args.syntax_allow_early_eos,
                                consume_out_of_class=args.syntax_consume_ooc,
                                terminal_restrict=args.syntax_terminal_restrict,
                                internal_only_restrict=args.syntax_internal_only,
                                eow_ids=args.syntax_eow_ids,
                                terminal_ids=args.syntax_terminal_ids)
                        else:
                            p = TokParsePredictor(
                                args.syntax_path,
                                p,
                                args.syntax_word_out,
                                args.normalize_fst_weights,
                                norm_alpha=args.syntax_norm_alpha,
                                beam_size=args.syntax_internal_beam,
                                max_internal_len=args.syntax_max_internal_len,
                                allow_early_eos=args.syntax_allow_early_eos,
                                consume_out_of_class=args.syntax_consume_ooc)
                    else:
                        p = ParsePredictor(
                            p,
                            args.normalize_fst_weights,
                            beam_size=args.syntax_internal_beam,
                            max_internal_len=args.syntax_max_internal_len,
                            nonterminal_ids=args.syntax_nonterminal_ids)
                elif wrapper == "altsrc":
                    src_test = _get_override_args("altsrc_test")
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedAltsrcPredictor(src_test, p)
                    else: # altsrc predictor for bounded predictors
                        p = AltsrcPredictor(src_test, p)
                elif wrapper == "rank":
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedRankPredictor(p)
                    else: # rank predictor for bounded predictors
                        p = RankPredictor(p)
                elif wrapper == "glue":
                    if isinstance(p, UnboundedVocabularyPredictor): 
                        p = UnboundedGluePredictor(args.max_len_factor, p)
                    else: # glue predictor for bounded predictors
                        p = GluePredictor(args.max_len_factor, p)
                elif wrapper == "word2char":
                    map_path = _get_override_args("word2char_map")
                    # word2char always wraps unbounded predictors
                    p = Word2charPredictor(map_path, p)
                elif wrapper == "skipvocab":
                    # skipvocab always wraps unbounded predictors
                    p = SkipvocabPredictor(args.skipvocab_max_id, 
                                           args.skipvocab_stop_size, 
                                           args.beam, 
                                           p)
                elif wrapper == "fsttok":
                    fsttok_path = _get_override_args("fsttok_path")
                    # fsttok always wraps unbounded predictors
                    p = FSTTokPredictor(fsttok_path,
                                        args.fst_unk_id,
                                        args.fsttok_max_pending_score,
                                        p)
                elif wrapper == "ngramize":
                    # ngramize always wraps bounded predictors
                    p = NgramizePredictor(args.min_ngram_order, 
                                          args.max_ngram_order,
                                          args.max_len_factor, p)
                elif wrapper == "unkvocab":
                    # unkvocab always wraps bounded predictors
                    p = UnkvocabPredictor(args.trg_vocab_size, p)
                else:
                    logging.fatal("Predictor wrapper '%s' not available. "
                                  "Please double-check --predictors for "
                                  "spelling errors." % wrapper)
                    decoder.remove_predictors()
                    return
            decoder.add_predictor(pred, p, pred_weight)
            logging.info("Initialized predictor {} (weight: {})".format(
                             pred, pred_weight))
    except IOError as e:
        logging.fatal("One of the files required for setting up the "
                      "predictors could not be read: %s" % e)
        decoder.remove_predictors()
    except AttributeError as e:
        logging.fatal("Invalid argument for one of the predictors: %s."
                       "Stack trace: %s" % (e, traceback.format_exc()))
        decoder.remove_predictors()
    except NameError as e:
        logging.fatal("Could not find external library: %s. Please make sure "
                      "that your PYTHONPATH and LD_LIBRARY_PATH contains all "
                      "paths required for the predictors. Stack trace: %s" % 
                      (e, traceback.format_exc()))
        decoder.remove_predictors()
    except ValueError as e:
        logging.fatal("A number format error occurred while configuring the "
                      "predictors: %s. Please double-check all integer- or "
                      "float-valued parameters such as --predictor_weights and"
                      " try again. Stack trace: %s" % (e, traceback.format_exc()))
        decoder.remove_predictors()
    except Exception as e:
        logging.fatal("An unexpected %s has occurred while setting up the pre"
                      "dictors: %s Stack trace: %s" % (sys.exc_info()[0],
                                                       e,
                                                       traceback.format_exc()))
        decoder.remove_predictors()


def create_decoder():
    """Creates the ``Decoder`` instance. This specifies the search 
    strategy used to traverse the space spanned by the predictors. This
    method relies on the global ``args`` variable.
    
    TODO: Refactor to avoid long argument lists
    
    Returns:
        Decoder. Instance of the search strategy
    """
    # Create decoder instance and add predictors
    decoder = None
    try:
        if args.decoder == "greedy":
            decoder = GreedyDecoder(args)
        elif args.decoder == "beam":
            decoder = BeamDecoder(args)
        elif args.decoder == "multisegbeam":
            decoder = MultisegBeamDecoder(args,
                                          args.hypo_recombination,
                                          args.beam,
                                          args.multiseg_tokenizations,
                                          args.early_stopping,
                                          args.max_word_len)
        elif args.decoder == "syncbeam":
            decoder = SyncBeamDecoder(args)
        elif args.decoder == "mbrbeam":
            decoder = MBRBeamDecoder(args)
        elif args.decoder == "sepbeam":
            decoder = SepBeamDecoder(args)
        elif args.decoder == "syntaxbeam":
            decoder = SyntaxBeamDecoder(args)
        elif args.decoder == "combibeam":
            decoder = CombiBeamDecoder(args)
        elif args.decoder == "dfs":
            decoder = DFSDecoder(args)
        elif args.decoder == "restarting":
            decoder = RestartingDecoder(args,
                                        args.hypo_recombination,
                                        args.max_node_expansions,
                                        args.low_decoder_memory,
                                        args.restarting_node_score,
                                        args.stochastic_decoder,
                                        args.decode_always_single_step)
        elif args.decoder == "bow":
            decoder = BOWDecoder(args)
        elif args.decoder == "flip":
            decoder = FlipDecoder(args)
        elif args.decoder == "bigramgreedy":
            decoder = BigramGreedyDecoder(args)
        elif args.decoder == "bucket":
            decoder = BucketDecoder(args,
                                    args.hypo_recombination,
                                    args.max_node_expansions,
                                    args.low_decoder_memory,
                                    args.beam,
                                    args.pure_heuristic_scores,
                                    args.decoder_diversity_factor,
                                    args.early_stopping,
                                    args.stochastic_decoder,
                                    args.bucket_selector,
                                    args.bucket_score_strategy,
                                    args.collect_statistics)
        elif args.decoder == "astar":
            decoder = AstarDecoder(args)
        elif args.decoder == "vanilla":
            decoder = construct_nmt_vanilla_decoder()
            args.predictors = "vanilla"
        else:
            logging.fatal("Decoder %s not available. Please double-check the "
                          "--decoder parameter." % args.decoder)
    except Exception as e:
        logging.fatal("An %s has occurred while initializing the decoder: %s"
                      " Stack trace: %s" % (sys.exc_info()[0],
                                            e,
                                            traceback.format_exc()))
    if decoder is None:
        sys.exit("Could not initialize decoder.")
    add_predictors(decoder)
    # Add heuristics for search strategies like A*
    if args.heuristics:
        add_heuristics(decoder)
    return decoder


def construct_nmt_vanilla_decoder():
    """Creates the vanilla NMT decoder which bypasses the predictor 
    framework. It uses the template methods ``get_nmt_vanilla_decoder``
    for uniform access to the blocks or tensorflow frameworks.
    
    Returns:
        NMT vanilla decoder using all specified NMT models, or None if
        an error occurred.
    """
    is_nmt = ["nmt" == p for p in args.predictors.split(",")]
    n = len(is_nmt)
    if not all(is_nmt):
        logging.fatal("Vanilla decoder can only be used with nmt predictors")
        return None
    nmt_specs = []
    if args.nmt_engine == 'blocks':
        get_default_nmt_config = blocks_get_default_nmt_config
        get_nmt_vanilla_decoder = blocks_get_nmt_vanilla_decoder
    elif args.nmt_engine == 'tensorflow':
        get_default_nmt_config = tf_get_default_nmt_config
        get_nmt_vanilla_decoder = tf_get_nmt_vanilla_decoder
    for _ in xrange(n): 
        nmt_specs.append((_get_override_args("nmt_path"),
                          _parse_config_param("nmt_config",
                                              get_default_nmt_config())))
    return get_nmt_vanilla_decoder(args, nmt_specs)


def add_heuristics(decoder):
    """Adds all enabled heuristics to the ``decoder``. This is relevant
    for heuristic based search strategies like A*. This method relies 
    on the global ``args`` variable and reads out ``args.heuristics``.
    
    Args:
        decoder (Decoder):  Decoding strategy, see ``create_decoder()``.
            This method will add heuristics to this instance with
            ``add_heuristic()``
    """
    if args.heuristic_predictors == 'all':
        h_predictors = decoder.predictors
    else:
        h_predictors = [decoder.predictors[int(idx)]
                       for idx in utils.split_comma(args.heuristic_predictors)]
    decoder.set_heuristic_predictors(h_predictors)
    for name in utils.split_comma(args.heuristics):
        if name == 'greedy':
            decoder.add_heuristic(GreedyHeuristic(args,
                                                  args.cache_heuristic_estimates))
        elif name == 'predictor':
            decoder.add_heuristic(PredictorHeuristic())
        elif name == 'stats':
            decoder.add_heuristic(StatsHeuristic(args.heuristic_scores_file,
                                                 args.collect_statistics))
        elif name == 'scoreperword':
            decoder.add_heuristic(ScorePerWordHeuristic())
        elif name == 'lasttoken':
            decoder.add_heuristic(LastTokenHeuristic())
        else:
            logging.fatal("Heuristic %s not available. Please double-check "
                          "the --heuristics parameter." % name)


def create_output_handlers():
    """Creates the output handlers defined in the ``io`` module. 
    These handlers create output files in different formats from the
    decoding results.
    
    Args:
        args: Global command line arguments.
    
    Returns:
        list. List of output handlers according --outputs
    """
    if not args.outputs:
        return []
    trg_map = {} if utils.trg_cmap else utils.trg_wmap
    outputs = []
    for name in utils.split_comma(args.outputs):
        if '%s' in args.output_path:
            path = args.output_path % name
        else:
            path = args.output_path
        if name == "text":
            outputs.append(TextOutputHandler(path, trg_map))
        elif name == "nbest":
            outputs.append(NBestOutputHandler(path, 
                                              utils.split_comma(args.predictors),
                                              trg_map))
        elif name == "ngram":
            outputs.append(NgramOutputHandler(path,
                                              args.min_ngram_order,
                                              args.max_ngram_order))
        elif name == "timecsv":
            outputs.append(TimeCSVOutputHandler(path, 
                                                utils.split_comma(args.predictors)))
        elif name == "fst":
            outputs.append(FSTOutputHandler(path,
                                            args.fst_unk_id))
        elif name == "sfst":
            outputs.append(StandardFSTOutputHandler(path,
                                                    args.fst_unk_id))
        else:
            logging.fatal("Output format %s not available. Please double-check"
                          " the --outputs parameter." % name)
    return outputs


def get_sentence_indices(range_param, src_sentences):
    """Helper method for ``do_decode`` which returns the indices of the
    sentence to decode
    
    Args:
        range_param (string): ``--range`` parameter from config
        src_sentences (list):  A list of strings. The strings are the
                               source sentences with word indices to 
                               translate (e.g. '1 123 432 2')
    """
    ids = []
    if args.range:
        try:
            if ":" in args.range:
                from_idx,to_idx = args.range.split(":")
            else:
                from_idx = int(args.range)
                to_idx = from_idx
            ids = xrange(int(from_idx)-1, int(to_idx))
        except Exception as e:
            logging.info("The --range does not seem to specify a numerical "
                         "range (%s). Interpreting as file name.." % e)
            tmp_path = "%s/sgnmt-tmp.%s" % (os.path.dirname(args.range), 
                                            uuid.uuid4())
            logging.debug("Temporary range file: %s" % tmp_path)
            while True:
                try:
                    os.rename(args.range, tmp_path)
                    with open(tmp_path) as tmp_f:
                        all_ids = [i.strip() for i in tmp_f]
                    next_id = None
                    if all_ids:
                        next_id = all_ids[0]
                        all_ids = all_ids[1:]
                    with open(tmp_path, "w") as tmp_f:
                        tmp_f.write("\n".join(all_ids))
                    os.rename(tmp_path, args.range)
                    if next_id is None:
                        return
                    logging.debug("Fetched ID %s and updated %s"
                                  % (next_id, args.range))
                    yield int(next_id)-1
                except Exception as e:
                    logging.debug("Could not fetch sentence ID from %s (%s). "
                                  "Trying again in 2 seconds..." 
                                  % (args.range, e))
                    time.sleep(2)
    else:
        if src_sentences is False:
           logging.fatal("Input method dummy requires --range")
        else:
            ids = xrange(len(src_sentences))
    for i in ids:
        yield i


def _get_text_output_handler(output_handlers):
    """Returns the text output handler if in output_handlers, or None."""
    for output_handler in output_handlers:
        if isinstance(output_handler, TextOutputHandler):
            return output_handler
    return None


def _postprocess_complete_hypos(hypos):
    """This function applies the following operations on the list of
    complete hypotheses returned by the Decoder:

      - </s> removal
      - Apply --nbest parameter if necessary
      - Applies combination_scheme on full hypotheses, reorder list

    Args:
      hypos (list): List of complete hypotheses

    Returns:
      list. Postprocessed hypotheses.
    """
    if args.remove_eos:
        for hypo in hypos:
            if (hypo.trgt_sentence 
                    and hypo.trgt_sentence[-1] == utils.EOS_ID):
                hypo.trgt_sentence = hypo.trgt_sentence[:-1]
    if args.nbest > 0:
        hypos = hypos[:args.nbest]
    kwargs={'full': True}
    if args.combination_scheme != 'sum': 
        if args.combination_scheme == 'length_norm':
            breakdown_fn = combination.breakdown2score_length_norm
        elif args.combination_scheme == 'bayesian_loglin':
            breakdown_fn = combination.breakdown2score_bayesian_loglin
        elif args.combination_scheme == 'bayesian':
            breakdown_fn = combination.breakdown2score_bayesian  
        elif args.combination_scheme == 'bayesian_state_dependent':
            breakdown_fn = combination.breakdown2score_bayesian_state_dependent  
            kwargs['lambdas'] = CombiBeamDecoder.get_domain_task_weights(
                args.bayesian_domain_task_weights)
        else:
            logging.warn("Unknown combination scheme '%s'" 
                         % args.combination_scheme)
        for hypo in hypos:
            hypo.total_score = breakdown_fn(
                    hypo.total_score, hypo.score_breakdown, **kwargs)
        hypos.sort(key=lambda hypo: hypo.total_score, reverse=True)
    return hypos


def _generate_dummy_hypo(predictors):
    return Hypothesis([utils.UNK_ID], 0.0, [[(0.0, w) for _, w in predictors]]) 


def do_decode(decoder, 
              output_handlers, 
              src_sentences):
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
        logging.fatal("Terminated due to an error in the "
                      "predictor configuration.")
        return
    all_hypos = []
    text_output_handler = _get_text_output_handler(output_handlers)
    if text_output_handler:
        text_output_handler.open_file()
    start_time = time.time()
    logging.info("Start time: %s" % start_time)
    sen_indices = []
    for sen_idx in get_sentence_indices(args.range, src_sentences):
        decoder.set_current_sen_id(sen_idx)
        try:
            if src_sentences is False:
                src = "0"
                logging.info("Next sentence (ID: %d)" % (sen_idx + 1))
            else:
                src = src_sentences[sen_idx]
            if len(src) > 0 and args.per_sentence_predictor_weights:
                # change predictor weights per-sentence
                weights = src[-1].split(',')
                if len(weights) > 1:
                    weights = [float(x) for x in weights]
                    src = src[:-1]
                    logging.info('Changing predictor weights to {}'.format(
                        weights))
                    decoder.change_predictor_weights(weights)
                else:
                    logging.info(
                        'No weights read in {} - leaving unchanged'.format(
                            src))
            logging.info("Next sentence (ID: %d): %s" % (sen_idx + 1, ' '.join(src)))
            src = [int(x) for x in src]
            start_hypo_time = time.time()
            decoder.apply_predictors_count = 0
            hypos = [hypo 
                     for hypo in decoder.decode(utils.apply_src_wmap(src))
                        if hypo.total_score > args.min_score]
            if not hypos:
                logging.error("No translation found for ID %d!" % (sen_idx+1))
                logging.info("Stats (ID: %d): score=<not-found> "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time))
                hypos = [_generate_dummy_hypo(decoder.predictors)]
            hypos = _postprocess_complete_hypos(hypos)
            if utils.trg_cmap:
                hypos = [h.convert_to_char_level(utils.trg_cmap) for h in hypos]
            logging.info("Decoded (ID: %d): %s" % (
                    sen_idx+1,
                    utils.apply_trg_wmap(hypos[0].trgt_sentence, 
                                         {} if utils.trg_cmap else utils.trg_wmap)))
            logging.info("Stats (ID: %d): score=%f "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        hypos[0].total_score,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time))
            all_hypos.append(hypos)
            sen_indices.append(sen_idx)
            try:
                # Write text output as we go
                if text_output_handler:
                    text_output_handler.write_hypos([hypos])
            except IOError as e:
                logging.error("I/O error %d occurred when creating output files: %s"
                            % (sys.exc_info()[0], e))
        except ValueError as e:
            logging.error("Number format error at sentence id %d: %s, "
                          "Stack trace: %s" % (sen_idx+1, 
                                               e,
                                               traceback.format_exc()))
        except AttributeError as e:
            logging.fatal("Attribute error at sentence id %d: %s. This often "
                          "indicates an error in the predictor configuration "
                          "which could not be detected in initialisation. "
                          "Stack trace: %s" 
                          % (sen_idx+1, e, traceback.format_exc()))
        except Exception as e:
            logging.error("An unexpected %s error has occurred at sentence id "
                          "%d: %s, Stack trace: %s" % (sys.exc_info()[0],
                                                       sen_idx+1,
                                                       e,
                                                       traceback.format_exc()))
    logging.info("Decoding finished. Time: %.2f" % (time.time() - start_time))
    try:
        for output_handler in output_handlers:
            if output_handler == text_output_handler:
                output_handler.close_file()
            else:
                output_handler.write_hypos(all_hypos, sen_indices)
    except IOError as e:
        logging.error("I/O error %s occurred when creating output files: %s"
                      % (sys.exc_info()[0], e))

