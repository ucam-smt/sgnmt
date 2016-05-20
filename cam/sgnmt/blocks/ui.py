"""This module handles configuration and user interface when using 
blocks. ``yaml`` and ``ArgumentParser`` are used for parsing config
files and command line arguments.
"""

import argparse
import yaml
import logging

from cam.sgnmt.blocks.machine_translation import configurations


def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


def parse_args(parser):
    """http://codereview.stackexchange.com/questions/79008/parse-a-config-file-
    and-add-to-command-line-arguments-using-argparse-in-python """
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def get_train_parser():
    """Get the parser object for NMT training configuration. """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument("--proto",  default="get_config_gnmt",
                        help="Prototype config to use for config")
    parser.add_argument("--bokeh",  default=False, action="store_true",
                        help="Use bokeh server for plotting")
    parser.add_argument("--reshuffle",  default=False, action="store_true",
                        help="Reshuffle before each epoch")
    default_config = configurations.get_config_gnmt()
    nmt_help_texts = get_nmt_config_help()
    for k in default_config:
        arg_type = type(default_config[k])
        if arg_type == bool:
            arg_type = 'bool'
        parser.add_argument(
                    "--%s" % k,
                    default=default_config[k],
                    type=arg_type,
                    help=nmt_help_texts[k])
    return parser


def get_parser():
    """Get the parser object which is used to build the configuration
    argument ``args``. This is a helper method for ``get_args()``
    
    Returns:
        ArgumentParser. The pre-filled parser object
    """
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    
    ## General options
    group = parser.add_argument_group('General options')
    group.add_argument('--config_file', 
                        help="Configuration file in standard .ini format. NOTE:"
                        " Configuration file overrides command line arguments",
                        type=argparse.FileType(mode='r'))
    group.add_argument("--verbosity", default="info",
                        choices=['debug', 'info', 'warn', 'error'],
                        help="Log level: debug,info,warn,error")
    group.add_argument("--min_score", default=-1000000.0, type=float,
                        help="Delete all complete hypotheses with total scores"
                        " smaller than this value")
    group.add_argument("--range", default="",
                        help="Defines the range of sentences to be processed. "
                        "Syntax is equal to HiFSTs printstrings and lmerts "
                        "idxrange parameter: <start-idx>:<end-idx> (both "
                        "inclusive, start with 1). E.g. 2:5 means: skip the "
                        "first sentence, process next 4 sentences")
    group.add_argument("--src_test", default="test_en",
                        help="Path to source test set. This is expected to be "
                        "a plain text file with one source sentence in each "
                        "line. Words need to be indexed, i.e. use word IDs "
                        "instead of their string representations.")
    group.add_argument("--en_test", default="",
                        help="DEPRECATED: Old name for --src_test")
    group.add_argument("--legacy_indexing", default=False, type='bool',
                        help="Defines the set of reserved word indices. The "
                        "standard convention is:\n"
                        "0: unk/eps, 1: <s>, 2: </s>.\n"
                        "Older systems use the TensorFlow scheme:"
                        "0: pad, 1: <s>, 2: </s>, 3: unk.\n"
                        "Set this parameter to true to use the old scheme.")
    group.add_argument("--input_method", default="file",
                        choices=['file', 'shell', 'stdin'],
                        help="This parameter controls how the input to GNMT "
                        "is provided. GNMT supports three modes:\n\n"
                        "* 'file': Read test sentences from a plain text file"
                            "specified by --src_test.\n"
                        "* 'shell': Start GNMT in an interactive shell.\n"
                        "* 'stdin': Test sentences are read from stdin\n\n"
                        "In shell and stdin mode you can change GNMT options "
                        "on the fly: Beginning a line with the string '!sgnmt '"
                        " signals GNMT directives instead of sentences to "
                        "translate. E.g. '!sgnmt config predictor_weights "
                        "0.2,0.8' changes the current predictor weights. "
                        "'!sgnmt help' lists all available directives. Using "
                        "GNMT directives is particularly useful in combination"
                        " with MERT to avoid start up times between "
                        "evaluations. Note that input sentences still have to "
                        "be written using word ids in all cases.")
    group.add_argument("--log_sum",  default="log",
                        choices=['tropical', 'log'],
                        help="Controls how to compute the sum in the log "
                        "space, i.e. how to compute log(exp(l1)+exp(l2)) for "
                        "log values l1,l2.\n\n"
                        "* 'tropical': approximate with max(l1,l2)\n"
                        "* 'log': Use logsumexp in scipy")
    
    ## Decoding options
    group = parser.add_argument_group('Decoding options')
    group.add_argument("--beam", default=12, type=int,
                        help="Size of beam. Only used if --decoder is set to "
                        "'beam' or 'astar'. For 'astar' it limits the capacity"
                        " of the queue. Use --beam 0 for unlimited capacity.")
    group.add_argument("--decoder", default="beam",
                        choices=['greedy',
                                 'beam',
                                 'dfs',
                                 'restarting',
                                 'astar',
                                 'vanilla'],
                        help="Strategy for traversing the search space which "
                        "is spanned by the predictors.\n\n"
                        "* 'greedy': Greedy decoding (similar to beam=1)\n"
                        "* 'beam': beam search like in Bahdanau et al, 2015\n"
                        "* 'dfs': Depth-first search. This should be used for "
                        "exact decoding or the complete enumeration of the "
                        "search space, but it cannot be used if the search "
                        "space is too large (like for unrestricted NMT) as "
                        "it performs exhaustive search. If you have not only "
                        "negative predictor scores, set --early_stopping to "
                        "false.\n"
                        "* 'restarting': Like DFS but with better admissible "
                        "pruning behavior.\n"
                        "* 'astar': A* search. The heuristic function is "
                        "configured using the --heuristics options.\n"
                        "* 'vanilla': Original blocks beam decoder. This "
                        "bypasses the predictor framework and directly "
                        "performs pure NMT beam decoding on the GPU. Use this "
                        "when you do pure NMT decoding as this is usually "
                        "faster then using a single nmt predictor as the "
                        "search can be parallelized on the GPU.")
    group.add_argument("--max_node_expansions", default=0, type=int,
                        help="This parameter allows to limit the total number "
                        "of search space expansions for a single sentence. "
                        "Currently, this parameter is only supported by the "
                        "'dfs' and 'restarting' decoder. 1000 is a good value "
                        "for very gentle pruning, 0 means no limitation")
    group.add_argument("--early_stopping", default=True, type='bool',
                        help="Use this parameter if you are only interested in"
                        "the first best decoding result. This option has a "
                        "different effect depending on the used --decoder. For"
                        " the beam decoder, it means stopping decoding when "
                        "the best active hypothesis ends with </s>. If false, "
                        "do not stop until all hypotheses end with EOS. For "
                        "the dfs and restarting decoders, early stopping "
                        "enables admissible pruning of branches when the "
                        "accumulated score already exceeded the currently best"
                        "score. DO NOT USE early stopping in combination with "
                        "the dfs or restarting decoder when your predictors "
                        "can produce positive scores!")
    
    ## Output options
    group = parser.add_argument_group('Output options')
    group.add_argument("--nbest", default=0, type=int,
                        help="Maximum number of hypotheses in the output "
                        "files. Set to 0 to output all hypotheses found by "
                        "the decoder. If you use the beam or astar decoder, "
                        "this option is limited by the beam size.")
    group.add_argument("--output_fst_unk_id", default=999999998, type=int,
                        help="SGNMT uses the ID 0 for UNK. However, this "
                        "clashes with OpenFST when writing FSTs as OpenFST "
                        "reserves 0 for epsilon arcs. Therefore, we use this "
                        "ID for UNK instead. Note that this only applies "
                        "to output FSTs created by the fst or sfst output "
                        "handler. Apart from that, UNK is still represented "
                        "by the ID 0.")
    group.add_argument("--output_path", default="sgnmt-out.%s",
                        help="Path to the output files generated by SGNMT. You "
                        "can use the placeholder %%s for the format specifier")
    group.add_argument("--outputs", default="",
                        help="Comma separated list of output formats: \n\n"
                        "* 'text': First best translations in plain text "
                        "format\n"
                        "* 'nbest': Moses' n-best format with separate "
                        "scores for each predictor.\n"
                        "* 'fst': Translation lattices in OpenFST "
                        "format with sparse tuple arcs.\n"
                        "* 'sfst': Translation lattices in OpenFST "
                        "format with standard arcs (i.e. combined scores).\n\n"
                        "The path to the output files can be specified with "
                        "--output_path")
    group.add_argument("--remove_eos", default=True, type='bool',
                        help="Whether to remove </S> symbol on output.")
    group.add_argument("--heuristics", default="",
                        help="Comma-separated list of heuristics to use in "
                        "heuristic based search like A*.\n\n"
                        "* 'predictor': Predictor specific heuristics. Some "
                        "predictors come with own heuristics - e.g. the fst "
                        "predictor uses the shortest path to the final state."
                        " Using 'predictor' combines the specific heuristics "
                        "of all selected predictors.\n"
                        "* 'greedy': Do greedy decoding to get the heuristic"
                        " costs. This is expensive but accurate.\n"
                        "* 'scoreperword': Using this heuristic normalizes the"
                        " previously accumulated costs by its length. It can "
                        "be used for beam search with normalized scores, using"
                        " a capacity (--beam), no other heuristic, and setting"
                        "--decoder to astar.\n\n"
                        "Note that all heuristics are inadmissible, i.e. A* "
                        "is not guaranteed to find the globally best path.")
    group.add_argument("--heuristic_predictors", default="all",
                        help="Comma separated list of indices of predictors "
                        "considered by the heuristic. For example, if "
                        "--predictors is set to nmt,length,fst then setting "
                        "--heuristic_predictors to 0,2 results in using nmt "
                        "and fst in the heuristics. Use 'all' to use all "
                        "predictors in the heuristics")
    group.add_argument("--cache_heuristic_estimates", default=True, type='bool',
                        help="Whether to cache heuristic future cost "
                        "estimates. This is especially useful with the greedy "
                        "heuristic.")
    
    ## Predictor options
    
    # General
    group = parser.add_argument_group('General predictor options')
    group.add_argument("--predictors", default="nmt",
                        help="Comma separated list of predictors. Predictors "
                        "are scoring modules which define a distribution over "
                        "target words given the history and some side "
                        "information like the source sentence. If vocabulary "
                        "sizes differ among predictors, we fill in gaps with "
                        "predictor UNK scores.:\n\n"
                        "* 'nmt': neural machine translation predictor.\n"
                        "         Options: see machine_translation."
                        "configurations plus proto, nmt_model_selector, "
                        "cache_nmt_posteriors.\n"
                        "* 'srilm': n-gram language model.\n"
                        "          Options: srilm_path, srilm_order\n"
                        "* 'nplm': neural n-gram language model (NPLM).\n"
                        "          Options: nplm_path, normalize_nplm_probs\n"
                        "* 'forced': Forced decoding with one reference\n"
                        "            Options: trg_test\n"
                        "* 'forcedlst': Forced decoding with a Moses n-best "
                        "list (n-best list rescoring)\n"
                        "               Options: trg_test, "
                        "forcedlst_sparse_feat, use_nbest_weights\n"
                        "* 'fst': Deterministic translation lattices\n"
                        "         Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "add_fst_bos_to_eos_weight\n"
                        "* 'nfst': Non-deterministic translation lattices\n"
                        "          Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "add_fst_bos_to_eos_weight\n"
                        "* 'rtn': Recurrent transition networks as created by "
                        "HiFST with late expansion.\n"
                        "         Options: rtn_path, use_rtn_weights, "
                        "minimize_rtns, remove_epsilon_in_rtns, "
                        "normalize_rtn_weights\n"
                        "* 'lrhiero': Direct Hiero (left-to-right Hiero). This "
                        "is a EXPERIMENTAL implementation of LRHiero.\n"
                        "             Options: rules_path, "
                        "grammar_feature_weights, use_grammar_weights\n"
                        "* 'wc': Number of words feature.\n"
                        "        Options: no options.\n"
                        "* 'length': Target sentence length model\n"
                        "            Options: src_test_raw, "
                        "length_model_weights, use_length_point_probs\n"
                        "All predictors can be combined with one or more "
                        "wrapper predictors by adding the wrapper name "
                        "separated by a _ symbol. Following wrappers are "
                        "available:\n"
                        "* 'idxmap': Add this wrapper to predictors which use "
                        "an alternative word map."
                        "            Options: src_idxmap, trg_idxmap\n"
                        "\n"
                        "Note that you can use multiple instances of the same "
                        "predictor. For example, 'nmt,nmt,nmt' can be used "
                        "for ensembling three NMT systems. You can often "
                        "override parts of the predictor configurations for "
                        "subsequent predictors by adding the predictor "
                        "number (e.g. see --nmt_config2 or --fst_path2)")
    group.add_argument("--predictor_weights", default="",
                        help="Predictor weights. Have to be specified "
                        "consistently with --predictor, e.g. if --predictor is"
                        " 'bla_fst,nmt' then set their weights with "
                        "--predictor_weights bla-weight_fst-weight,nmt-weight,"
                        " e.g. '--predictor_weights 0.1_0.3,0.6'. Default "
                        "(empty string) means that each predictor gets "
                        "assigned the weight 1.")
    group.add_argument("--closed_vocabulary_normalization", default="none",
                        choices=['none', 'exact', 'reduced'],
                        help="This parameter specifies the way closed "
                        "vocabulary predictors (e.g. NMT) are normalized. "
                        "Closed vocabulary means that they have a predefined "
                        "vocabulary. Open vocabulary predictors (e.g. fst) can"
                        " potentially produce any word, or have a very large "
                        "vocabulary.\n\n"
                        "* 'none': Use unmodified scores for closed "
                        "vocabulary predictors\n"
                        "* 'exact': Renormalize scores depending on the "
                        "probability mass which they distribute to words "
                        "outside the vocabulary via the UNK probability.\n"
                        "* 'reduced': Normalize to vocabulary defined by the "
                        "open vocabulary predictors at each time step.")
    group.add_argument("--combination_scheme", default="sum",
                        choices=['sum', 'length_norm', 'bayesian'],
                        help="This parameter controls how the combined "
                        "hypothesis score is calculated from the predictor "
                        "scores and weights.\n\n"
                        "* 'sum': The combined score is the weighted sum of "
                        "all predictor scores\n"
                        "* 'length_norm': Renormalize scores by the length of "
                        "hypotheses.\n"
                        "* 'bayesian': Apply the Bayesian LM interpolation "
                        "scheme from Allauzen and Riley to interpolate the "
                        "predictor scores")
    group.add_argument("--apply_combination_scheme_to_partial_hypos", 
                        default=False, type='bool',
                        help="If true, apply the combination scheme specified "
                        "with --combination_scheme after each node expansion. "
                        "If false, apply it only to complete hypotheses at "
                        "the end of decoding")
    
    # Neural predictors
    group = parser.add_argument_group('Neural predictor options')
    group.add_argument("--proto", default="get_config_gnmt",
                        help="Prototype configuration of the NMT model. See "
                        "cam.sgnmt.blocks.machine_translation.configuration "
                        "for available prototypes. However, it is recommended "
                        "to configure GNMT via command line arguments and "
                        "configuration files instead of using this option.")
    group.add_argument("--length_normalization", default=False, type='bool',
                        help="DEPRECATED. Synonym for --combination_scheme "
                        "length_norm. Normalize n-best hypotheses by sentence "
                        "length. Normally improves pure NMT decoding, but "
                        "degrades performance when combined with predictors "
                        "like fst or multiple NMT systems.")
    group.add_argument("--nmt_model_selector", default="bleu",
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
    group.add_argument("--cache_nmt_posteriors", default=False, type='bool',
                        help="This enables the cache in the [F]NMT predictor. "
                        "Normally, the search procedure is responsible to "
                        "avoid applying predictors to the same history twice. "
                        "However, due to the limited NMT vocabulary, two "
                        "different histories might be the same from the NMT "
                        "perspective, e.g. if they are the same up to words "
                        "which are outside the NMT vocabulary. If this "
                        "parameter is set to true, we cache posteriors with "
                        "histories containing UNK and reload them when needed")
    
    # Length predictors
    group = parser.add_argument_group('Length predictor options')
    group.add_argument("--src_test_raw", default="",
                        help="Only required for the 'length' predictor. Path "
                        "to original source test set WITHOUT word indices. "
                        "This is used to extract features for target sentence "
                        "length predictions")
    group.add_argument("--length_model_weights", default="",
                        help="Only required for length predictor. String of "
                        "length model parameters.")
    group.add_argument("--use_length_point_probs", default=False, type='bool',
                        help="If this is true, the length predictor outputs "
                        "probability 1 for all tokens except </S>. For </S> it"
                        " uses the point probability given by the length "
                        "model. If this is set to false, we normalize the "
                        "predictive score by comparing P(l=x) and P(l<x)")
    
    # Forced predictors
    group = parser.add_argument_group('Forced decoding predictor options')
    group.add_argument("--trg_test", default="test_fr",
                        help="Path to target test set (with integer tokens). "
                        "This is only required for the predictors 'forced' "
                        "and 'forcedlst'. For 'forcedlst' this needs to point "
                        "to an n-best list in Moses format.")
    group.add_argument("--fr_test", default="", 
                        help="DEPRECATED. Old name for --trg_test")
    group.add_argument("--forcedlst_sparse_feat", default="", 
                        help="Per default, the forcedlst predictor uses the "
                        "combined score in the Moses nbest list. Alternatively,"
                        " for nbest lists in sparse feature format, you can "
                        "specify the name of the features which should be "
                        "used instead.")
    group.add_argument("--use_nbest_weights", default=False, type='bool',
                        help="Only required for forcedlst predictor. Whether "
                        "to use the scores in n-best lists.")
    
    # Idxmap wrapper
    group = parser.add_argument_group('Indexmap predictor options')
    group.add_argument("--src_idxmap", default="idxmap.en",
                        help="Only required for idxmap wrapper predictor. Path"
                        " to the source side mapping file. The format is "
                        "'<index> <alternative_index>'. The mapping must be "
                        "complete and should be a bijection.")
    group.add_argument("--en_idxmap", default="",
                        help="DEPRECATED. Old name for --src_idxmap")
    group.add_argument("--trg_idxmap", default="idxmap.fr",
                        help="Only required for idxmap wrapper predictor. Path"
                        " to the target side mapping file. The format is "
                        "'<index> <alternative_index>'. The mapping must be "
                        "complete and should be a bijection.")
    group.add_argument("--fr_idxmap", default="",
                        help="DEPRECATED. Old name for --trg_idxmap")

    # Hiero predictor
    group = parser.add_argument_group('Hiero predictor options')
    group.add_argument("--rules_path", default="rules/rules",
                        help="Only required for predictor lrhiero. Path to "
                        "the ruleXtract rules file.")
    group.add_argument("--use_grammar_weights", default=False, type='bool',
                        help="Whether to use weights in the synchronous "
                        "grammar for the lrhiero predictor. If set to false, "
                        "use uniform grammar scores.")
    group.add_argument("--grammar_feature_weights", default='',
                        help="If rules_path points to a factorized rules file "
                        "(i.e. containing rules associated with a number of "
                        "features, not only one score) GNMT uses a weighted "
                        "sum for them. You can specify the weights for this "
                        "summation here (comma-separated) or leave it blank "
                        "to sum them up equally weighted.")
    
    # (NP)LM predictors
    group = parser.add_argument_group('Neural LM predictor options')
    group.add_argument("--srilm_path", default="lm/ngram.lm.gz",
                        help="Path to the ngram LM file in SRILM format")
    group.add_argument("--nplm_path", default="nplm/nplm.gz",
                        help="Path to the NPLM language model")
    group.add_argument("--srilm_order", default=5, type=int,
                        help="Order of ngram for srilm predictor")
    group.add_argument("--normalize_nplm_probs", default=False, type='bool',
                        help="Whether to normalize nplm probabilities over "
                        "the current unbounded predictor vocabulary.")
    
    # Automaton predictors
    group = parser.add_argument_group('FST and RTN predictor options')
    group.add_argument("--fst_path", default="fst/%d.fst",
                        help="Only required for fst and nfst predictor. Sets "
                        "the path to the OpenFST translation lattices. You "
                        "can use the placeholder %%d for the sentence index.")
    group.add_argument("--rtn_path", default="rtn/",
                        help="Only required for rtn predictor. Sets "
                        "the path to the RTN directory as created by HiFST")
    group.add_argument("--add_fst_bos_to_eos_weight", default=False, type='bool',
                        help="This option applies to fst, nfst and rtn "
                        "predictors. Lattices produced by HiFST contain the "
                        "<S> symbol and often have scores on the corresponding"
                        " arc. However, GNMT skips <S> and this score is not "
                        "regarded anywhere. Set this option to true to add the "
                        "<S> score to the </S> arc. This ensures that the "
                        "complete path scores for the [n]fst and rtn "
                        "predictors match the corresponding path weights in "
                        "the original FST as obtained with fstshortestpath.")
    group.add_argument("--fst_to_log", default=True, type='bool',
                        help="Multiply weights in the FST by -1 to transform "
                        "them from tropical semiring into logprobs.")
    group.add_argument("--use_fst_weights", default=False, type='bool',
                        help="Whether to use weights in FSTs for the"
                        "nfst and fst predictor.")
    group.add_argument("--use_rtn_weights", default=False, type='bool',
                        help="Whether to use weights in RTNs.")
    group.add_argument("--minimize_rtns", default=True, type='bool',
                        help="Whether to do determinization, epsilon removal, "
                        "and minimization after each RTN expansion.")
    group.add_argument("--remove_epsilon_in_rtns", default=True, type='bool',
                        help="Whether to remove epsilons after RTN expansion.")
    group.add_argument("--normalize_fst_weights", default=False, type='bool',
                        help="Whether to normalize weights in FSTs. This "
                        "forces the weights on outgoing edges to sum up to 1. "
                        "Applicable to fst and nfst predictor.")
    group.add_argument("--normalize_rtn_weights", default=False, type='bool',
                        help="Whether to normalize weights in RTNs. This "
                        "forces the weights on outgoing edges to sum up to 1. "
                        "Applicable to rtn predictor.")
    
    # Adding arguments for overriding when using same predictor multiple times
    group = parser.add_argument_group('Override options')
    for n,w in [('2', 'second'), ('3', 'third'), ('4', '4-th'), ('5', '5-th'), 
                ('6', '6-th'), ('7', '7-th'), ('8', '8-th'), ('9', '9-th'), 
                ('10', '10-th'), ('11', '11-th'), ('12', '12-th')]:
        group.add_argument("--nmt_config%s" % n,  default="",
                        help="If the --predictors string contains more than "
                        "one nmt predictor, you can specify the configuration "
                        "for the %s one with this parameter. The %s nmt "
                        "predictor inherits all settings under 'Neural model "
                        "configuration' except for the ones in this parameter. "
                        "Usage: --nmt_config%s 'save_to=train%s,enc_embed=400'"
                        % (w, w, n, n))
        group.add_argument("--src_idxmap%s" % n, default="",
                        help="Overrides --src_idxmap for the %s indexmap" % w)
        group.add_argument("--trg_idxmap%s" % n, default="",
                        help="Overrides --trg_idxmap for the %s indexmap" % w)
        group.add_argument("--fst_path%s" % n, default="",
                        help="Overrides --fst_path for the %s fst "
                        "predictor" % w)
        group.add_argument("--forcedlst_sparse_feat%s" % n, default="",
                        help="Overrides --forcedlst_sparse_feat for the %s "
                        "forcedlst predictor" % w)
    
    
    # Add NMT model options
    group = parser.add_argument_group('Neural model configuration')
    default_config = configurations.get_config_gnmt()
    nmt_help_texts = get_nmt_config_help()
    for k in default_config:
        arg_type = type(default_config[k])
        if arg_type == bool:
            arg_type = 'bool'
        group.add_argument("--%s" % k,
                            default=default_config[k],
                            type=arg_type,
                            help=nmt_help_texts[k])
    return parser


def get_args():
    """Get the arguments for the current SGNMT run from both command
    line arguments and configuration files. This method contains all
    available SGNMT options, i.e. configuration is not encapsulated e.g.
    by predictors. Additionally, we add blocks NMT model options as
    parameters to specify how the loaded NMT model was trained. These
    are defined in ``machine_translation.configurations``.
    
    Returns:
        object. Arguments object like for ``ArgumentParser``
    """ 
    parser = get_parser()
    args = parse_args(parser)
    
    # Legacy parameter names
    if args.en_test:
        args.src_test = args.en_test
    if args.fr_test:
        args.trg_test = args.fr_test
    if args.en_idxmap:
        args.src_idxmap = args.en_idxmap
    if args.fr_idxmap:
        args.trg_idxmap = args.fr_idxmap
    if args.length_normalization:
        args.combination_scheme = "length_norm"
    return args


def validate_args(args):
    """Some very rudimental sanity checks for configuration options.
    This method directly prints help messages to the user. In case of fatal
    errors, it terminates using ``logging.fatal()``
    
    Args:
        args (object):  Configuration as returned by ``get_args``
    """
    for depr in ['en_test', 'fr_test',
                 'length_normalization',
                 'en_idxmap', 'fr_idxmap']:
        if getattr(args, depr):
            logging.warn("Using deprecated argument %s." % depr)
    # Validate --range
    if args.range:
        if args.input_method == 'shell':
            logging.warn("The --range parameter can lead to unexpected "
                         "behavior in the 'shell' mode.")
        try:
            f,t = [int(i) for i in args.range.split(":")]
            if f > t:
                logging.fatal("Start index in --range greater than end index.")
            
        except:
            logging.fatal("Wrong format for --range parameter: %s" % args.range)


def get_nmt_config_help():
    """Creates a dictionary with help text for the NMT configuration """

    config = {}
    config['seq_len'] = "Sequences longer than this will be discarded"
    config['enc_nhids'] = "Number of hidden units in encoder GRU"
    config['dec_nhids'] = "Number of hidden units in decoder GRU"
    config['enc_embed'] = "Dimension of the word embedding matrix in encoder"
    config['dec_embed'] = "Dimension of the word embedding matrix in decoder"
    config['saveto'] = "Where to save model, same as 'prefix' in groundhog"
    config['batch_size'] = "Batch size"
    config['sort_k_batches'] = "This many batches will be read ahead and sorted"
    config['step_rule'] = "Optimization step rule"
    config['step_clipping'] = "Gradient clipping threshold"
    config['weight_scale'] = "Std of weight initialization"
    config['weight_noise_ff'] = "Weight noise flag for feed forward layers"
    config['weight_noise_rec'] = "Weight noise flag for recurrent layers"
    config['dropout'] = "Dropout ratio, applied only after readout maxout"
    config['stream'] = "Module name of the stream that will be used"
    config['src_data'] = "Source dataset"
    config['trg_data'] = "Target dataset"
    config['src_vocab_size'] = "Source vocab size, including special tokens"
    config['trg_vocab_size'] = "Target vocab size, including special tokens"
    config['unk_id'] = "UNK id"
    config['bos_token'] = "String representation for <S>"
    config['eos_token'] = "String representation for </S>"
    config['unk_token'] = "String representation for UNK"
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
    config['sampling_freq'] = "Show samples from model after this many updates"
    config['hook_samples'] = "Show this many samples at each sampling"
    config['bleu_val_freq'] = "Validate bleu after this many updates"
    config['val_burn_in'] = "Start bleu validation after this many updates"
    config['store_full_main_loop'] = "Old style archives (not recommended)"
    config['fix_embeddings'] = "Fix embeddings during training"

    return config