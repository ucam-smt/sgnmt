"""This module handles configuration and user interface when using 
blocks. ``yaml`` and ``ArgumentParser`` are used for parsing config
files and command line arguments.

TODO: Remove Blocks dependency
"""

import argparse
import logging
import os

from cam.sgnmt.blocks.nmt import blocks_add_nmt_config


YAML_AVAILABLE = True
try:
    import yaml
except:
    YAML_AVAILABLE = False

def str2bool(v):
    """For making the ``ArgumentParser`` understand boolean values"""
    return v.lower() in ("yes", "true", "t", "1")


def parse_args(parser):
    """http://codereview.stackexchange.com/questions/79008/parse-a-config-file-
    and-add-to-command-line-arguments-using-argparse-in-python """
    args = parser.parse_args()
    if args.config_file:
        if not YAML_AVAILABLE:
            logging.fatal("Install PyYAML in order to use config files.")
            return args
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


def parse_param_string(param):
    """Parses a parameter string such as 'param1=x,param2=y'. Loads 
    config files if specified in the string. If ``param`` points to a
    file, load this file with YAML.
    """
    if not param:
        return {}
    if os.path.isfile(param):
        param = "config_file=%s" % param
    config = {}
    for pair in param.strip().split(","):
        (k,v) = pair.split("=", 1)
        if k == 'config_file':
            if not YAML_AVAILABLE:
                logging.fatal("Install PyYAML in order to use config files.")
            else:
                with open(v) as f:
                    data = yaml.load(f)
                    for config_file_key, config_file_value in data.items():
                        config[config_file_key] = config_file_value
        else:
            config[k] = v
    return config


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


def get_parser():
    """Get the parser object which is used to build the configuration
    argument ``args``. This is a helper method for ``get_args()``
    TODO: Decentralize configuration
    
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
    group.add_argument("--indexing_scheme", default="blocks",
                        choices=['blocks', 'tf', 't2t'],
                        help="This parameter defines the reserved IDs.\n\n"
                        "* 'blocks': eps,unk: 0, <s>: 1, </s>: 2.\n"
                        "* 'tf': unk: 3, <s>: 1, </s>: 2.\n"
                        "* 't2t': unk: 3, <s>: 2, </s>: 1.")
    group.add_argument("--legacy_indexing", default=False, type='bool',
                        help="DEPRECATED: Use --indexing_scheme=tf instead")
    group.add_argument("--input_method", default="file",
                        choices=['dummy', 'file', 'shell', 'stdin'],
                        help="This parameter controls how the input to GNMT "
                        "is provided. GNMT supports three modes:\n\n"
                        "* 'dummy': Use dummy source sentences.\n"
                        "* 'file': Read test sentences from a plain text file"
                            "specified by --src_test.\n"
                        "* 'shell': Start SGNMT in an interactive shell.\n"
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
    group.add_argument("--single_cpu_thread", default=False, type='bool',
                        help="If true, try to prevent libraries like Theano "
                        "or TensorFlow from doing internal multithreading. "
                        "Also, see the OMP_NUM_THREADS environment variable.")
    
    ## Decoding options
    group = parser.add_argument_group('Decoding options')
    group.add_argument("--beam", default=12, type=int,
                        help="Size of beam. Only used if --decoder is set to "
                        "'beam' or 'astar'. For 'astar' it limits the capacity"
                        " of the queue. Use --beam 0 for unlimited capacity.")
    group.add_argument("--decoder", default="beam",
                        choices=['greedy',
                                 'beam',
                                 'multisegbeam',
                                 'syncbeam',
                                 'sepbeam',
                                 'dfs',
                                 'restarting',
                                 'bow',
                                 'flip',
                                 'bucket',
                                 'bigramgreedy',
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
                        "* 'multisegbeam': Beam search for predictors with "
                        "multiple tokenizations ([sub]word/char-levels).\n"
                        "* 'syncbeam': beam search which compares after "
                        "consuming a special synchronization symbol instead "
                        "of after each iteration.\n"
                        "* 'sepbeam': Associates predictors with hypos in "
                        "beam search and applies only one predictor instead "
                        "of all for hypo expansion.\n"
                        "* 'bow': Restarting decoder optimized for bag-of-words "
                        "problems.\n"
                        "* 'flip': This decoder works only for bag problems. "
                        "It traverses the search space by switching two words "
                        "in the hypothesis. Do not use bow predictor.\n"
                        "* 'bucket': Works best for bag problems. Maintains "
                        "buckets for each hypo length and extends a hypo in "
                        "a bucket by one before selecting the next bucket.\n"
                        "* 'bigramgreedy': Works best for bag problems. "
                        "Collects bigram statistics and constructs hypos to "
                        "score by greedily selecting high scoring bigrams. "
                        "Do not use bow predictor with this search strategy.\n"
                        "* 'astar': A* search. The heuristic function is "
                        "configured using the --heuristics options.\n"
                        "* 'vanilla': Original blocks beam decoder. This "
                        "bypasses the predictor framework and directly "
                        "performs pure NMT beam decoding on the GPU. Use this "
                        "when you do pure NMT decoding as this is usually "
                        "faster then using a single nmt predictor as the "
                        "search can be parallelized on the GPU.")
    group.add_argument("--hypo_recombination", default=False, type='bool',
                        help="Activates hypothesis recombination. Has to be "
                        "supported by the decoder. Applicable to beam, "
                        "restarting, bow, bucket")
    group.add_argument("--allow_unk_in_output", default=True, type='bool',
                        help="If false, remove all UNKs in the final "
                        "posteriors. Predictor distributions can still "
                        "produce UNKs, but they have to be replaced by "
                        "other words by other predictors")
    group.add_argument("--max_node_expansions", default=0, type=int,
                        help="This parameter allows to limit the total number "
                        "of search space expansions for a single sentence. "
                        "If this is 0 we allow an unlimited number of "
                        "expansions. If it is negative, the maximum number of "
                        "expansions is this times the length of the source "
                        "sentence. Supporting decoders:\n"
                        "bigramgreedy, bow, bucket, dfs, flip, restarting")
    group.add_argument("--max_len_factor", default=2, type=int,
                        help="Limits the length of hypotheses to avoid "
                        "infinity loops in search strategies for unbounded "
                        "search spaces. The length of any translation is "
                        "limited to max_len_factor times the length of the "
                        "source sentence.")
    group.add_argument("--early_stopping", default=True, type='bool',
                        help="Use this parameter if you are only interested in "
                        "the first best decoding result. This option has a "
                        "different effect depending on the used --decoder. For"
                        " the beam decoder, it means stopping decoding when "
                        "the best active hypothesis ends with </s>. If false, "
                        "do not stop until all hypotheses end with EOS. For "
                        "the dfs and restarting decoders, early stopping "
                        "enables admissible pruning of branches when the "
                        "accumulated score already exceeded the currently best "
                        "score. DO NOT USE early stopping in combination with "
                        "the dfs or restarting decoder when your predictors "
                        "can produce positive scores!")
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
                        "* 'stats': Collect unigram statistics during decoding"
                        "and compare actual hypothesis scores with the product"
                        " of unigram scores of the used words.\n"
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
    group.add_argument("--multiseg_tokenizations", default="",
                        help="This argument must be used when the multisegbeam"
                        " decoder is activated. For each predictor, it defines"
                        " the tokenizations used for it (comma separated). If "
                        "a path to a word map file is provided, the "
                        "corresponding predictor is operating on the pure "
                        "word level. The 'mixed:' prefix activates mixed "
                        "word/character models according Wu et al. (2016). "
                        "the 'eow': prefix assumes to find explicit </w>"
                        "specifiers in the word maps which mark end of words. "
                        "This is suitable for subword units, e.g. bpe.")
    group.add_argument("--cache_heuristic_estimates", default=True, type='bool',
                        help="Whether to cache heuristic future cost "
                        "estimates. This is especially useful with the greedy "
                        "heuristic.")
    group.add_argument("--pure_heuristic_scores", default=False, type='bool',
                        help="If this is set to false, heuristic decoders as "
                        "A* score hypotheses with the sum of the partial hypo "
                        "score plus the heuristic estimates (lik in standard "
                        "A*). Set to true to use the heuristic estimates only")
    group.add_argument("--restarting_node_score", default="difference",
                        choices=['difference',
                                 'absolute',
                                 'constant',
                                 'expansions'],
                        help="This parameter defines the strategy how the "
                        "restarting decoder decides from which node to restart"
                        ".\n\n"
                        "* 'difference': Restart where the difference between "
                        "1-best and 2-best is smallest\n"
                        "* 'absolute': Restart from the unexplored node with "
                        "the best absolute score globally.\n"
                        "* 'constant': Constant node score. Simulates FILO or "
                        "uniform distribution with restarting_stochastic.\n"
                        "* 'expansions': Inverse of the number of expansions "
                        "on the node. Discourages expanding arcs on the same "
                        "node repeatedly.\n")
    group.add_argument("--low_decoder_memory", default=True, type='bool',
                        help="Some decoding strategies support modes which do "
                        "not change the decoding logic, but make use of the "
                        "inadmissible pruning parameters like max_expansions "
                        "to reduce memory consumption. This usually requires "
                        "some  computational overhead for cleaning up data "
                        "structures. Applicable to restarting and bucket "
                        "decoders.")
    group.add_argument("--stochastic_decoder", default=False, type='bool',
                        help="Activates stochastic decoders. Applicable to the "
                        "decoders restarting, bow, bucket")
    group.add_argument("--decode_always_single_step", default=False, type='bool',
                        help="If this is set to true, heuristic depth first "
                        "search decoders like restarting or bow always perform "
                        "a single decoding step instead of greedy decoding. "
                        "Handle with care...")
    group.add_argument("--flip_strategy", default="move",
                        choices=['move', 'flip'],
                        help="Defines the hypothesis transition in the flip "
                        "decoder. 'flip' flips two words, 'move' moves a word "
                        "to a different position")
    group.add_argument("--bucket_selector", default="maxscore",
                        help="Defines the bucket selection strategy for the "
                        "bucket decoder.\n\n"
                        "* 'iter': Rotate through all lengths\n"
                        "* 'iter-n': Rotate through all lengths n times\n"
                        "* 'maxscore': Like iter, but filters buckets with "
                            "hypos worse than a threshold. Threshold is "
                            "increased if no bucket found\n"
                        "* 'score': Select bucket with the highest bucket "
                        "score. The bucket score is determined by the "
                        "bucket_score_strategy\n"
                        "* 'score-end': Start with the bucket with highest bucket "
                            "score, and iterate through all subsequent buckets. \n")
    group.add_argument("--bucket_score_strategy", default="difference",
                        choices=['difference', 'heap', 'absolute', 'constant'],
                        help="Defines how buckets are scored for the "
                        "bucket decoder. Usually, the best hypo in the bucket "
                        "is compared to the global best score of that length "
                        "according --collect_statistics.\n\n"
                        "* 'difference': Difference between both hypos\n"
                        "* 'heap': Use best score on bucket heap directly\n"
                        "* 'absolute': Use best hypo score in bucket directly\n"
                        "* 'constant': Uniform bucket scores.")
    group.add_argument("--collect_statistics", default="best",
                       choices=['best', 'full', 'all'],
                        help="Determines over which hypotheses statistics are "
                        "collected.\n\n"
                        "* 'best': Collect statistics from the current best "
                        "full hypothesis\n"
                        "* 'full': Collect statistics from all full hypos\n"
                        "* 'all': Collect statistics also from partial hypos\n"
                        "Applicable to the bucket decoder, the heuristic "
                        "of the bow predictor, and the heuristic 'stats'.")
    group.add_argument("--heuristic_scores_file", default="",
                       help="The bow predictor heuristic and the stats "
                       "heuristic sum up the unigram scores of words as "
                       "heuristic estimate. This option should point to a "
                       "mapping file from word-id to (unigram) score. If this "
                       "is empty, the unigram scores are collected during "
                       "decoding for each sentence separately according "
                       "--collect_statistics.")
    group.add_argument("--score_lower_bounds_file", default="",
                       help="Admissible pruning in some decoding strategies "
                       "can be improved by providing lower bounds on complete "
                       "hypothesis scores. This is useful to improve the "
                       "efficiency of exhaustive search, with lower bounds "
                       "found by e.g. beam search. The expected file format "
                       "is just a text file with line separated scores for "
                       "each sentence. Supported by the following decoders: "
                       "astar, bigramgreedy, bow, bucket, dfs, flip, restarting")
    group.add_argument("--decoder_diversity_factor", default=-1.0, type=float,
                       help="If this is greater than zero, promote diversity "
                       "between active hypotheses during decoding. The exact "
                       "way of doing this depends on --decoder:\n"
                       "* The 'beam' decoder roughly follows the approach in "
                       "Li and Jurafsky, 2016\n"
                       "* The 'bucket' decoder reorders the hypotheses in a "
                       "bucket by penalizing hypotheses with the number of "
                       "expanded hypotheses from the same parent.")
    group.add_argument("--sync_symbol", default=-1, type=int,
                       help="Used for the syncbeam decoder. Synchronization "
                       "symbol for hypothesis comparision. If negative, use "
                       "the </w> entry in --trg_cmap.")
    group.add_argument("--max_word_len", default=25, type=int,
                       help="Maximum length of a single word. Only applicable "
                       "to the decoders multisegbeam and syncbeam.")

    ## Output options
    group = parser.add_argument_group('Output options')
    group.add_argument("--nbest", default=0, type=int,
                        help="Maximum number of hypotheses in the output "
                        "files. Set to 0 to output all hypotheses found by "
                        "the decoder. If you use the beam or astar decoder, "
                        "this option is limited by the beam size.")
    group.add_argument("--output_fst_unk_id", default=0, type=int,
                        help="DEPRECATED: Old name for --fst_unk_id")
    group.add_argument("--fst_unk_id", default=999999998, type=int,
                        help="SGNMT uses the ID 0 for UNK. However, this "
                        "clashes with OpenFST when writing FSTs as OpenFST "
                        "reserves 0 for epsilon arcs. Therefore, we use this "
                        "ID for UNK instead. Note that this only applies "
                        "to output FSTs created by the fst or sfst output "
                        "handler, or FSTs used by the fsttok wrapper. Apart "
                        "from that, UNK is still represented by the ID 0.")
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
    group.add_argument("--src_wmap", default="",
                        help="Path to the source side word map (Format: <word>"
                        " <id>). This is used to map the words in --src_test "
                        "to their word IDs. If empty, SGNMT expects the input "
                        "words to be in integer representation.")
    group.add_argument("--trg_wmap", default="",
                        help="Path to the target side word map (Format: <word>"
                        " <id>). This is used to generate log output and the "
                        "output formats text and nbest. If empty, we directly "
                        "write word IDs.")
    group.add_argument("--trg_cmap", default="",
                        help="Path to the target side char map (Format: <char>"
                        " <id>). If this is not empty, all output files are "
                        "converted to character-level. The mapping from word "
                        "to character sequence is read from --trg_wmap. The "
                        "char map must contain an entry for </w> which points "
                        "to the word boundary ID.")
    
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
                        "         Options: nmt_config, nmt_path, gnmt_beta, "
                        "nmt_model_selector, cache_nmt_posteriors.\n"
                        "* 't2t': Tensor2Tensor predictor.\n"
                        "         Options: t2t_usr_dir, t2t_model, "
                        "t2t_problem, t2t_hparams_set, t2t_checkpoint_dir\n"
                        "* 'srilm': n-gram language model.\n"
                        "          Options: srilm_path, srilm_order\n"
                        "* 'nplm': neural n-gram language model (NPLM).\n"
                        "          Options: nplm_path, normalize_nplm_probs\n"
                        "* 'rnnlm': RNN language model based on TensorFlow.\n"
                        "          Options: rnnlm_config, rnnlm_path\n"
                        "* 'forced': Forced decoding with one reference\n"
                        "            Options: trg_test\n"
                        "* 'forcedlst': Forced decoding with a Moses n-best "
                        "list (n-best list rescoring)\n"
                        "               Options: trg_test, "
                        "forcedlst_sparse_feat, use_nbest_weights\n"
                        "* 'bow': Forced decoding with one bag-of-words ref.\n"
                        "         Options: trg_test, heuristic_scores_file, "
                        "bow_heuristic_strategies, bow_accept_subsets, "
                        "bow_accept_duplicates, bow_equivalence_vocab_size\n"
                        "* 'bowsearch': Forced decoding with one bag-of-words ref.\n"
                        "         Options: hypo_recombination, trg_test, "
                        "heuristic_scores_file, bow_heuristic_strategies, "
                        "bow_accept_subsets, bow_accept_duplicates, "
                        "bow_equivalence_vocab_size\n"
                        "* 'fst': Deterministic translation lattices\n"
                        "         Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "fst_skip_bos_weight\n"
                        "* 'nfst': Non-deterministic translation lattices\n"
                        "          Options: fst_path, use_fst_weights, "
                        "normalize_fst_weights, fst_to_log, "
                        "fst_skip_bos_weight\n"
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
                        "        Options: wc_word.\n"
                        "* 'unkc': Poisson model for number of UNKs.\n"
                        "          Options: unk_count_lambdas.\n"
                        "* 'ngramc': Number of ngram feature.\n"
                        "            Options: ngramc_path, ngramc_order.\n"
                        "* 'length': Target sentence length model\n"
                        "            Options: src_test_raw, "
                        "length_model_weights, use_length_point_probs\n"
                        "* 'extlength': External target sentence lengths\n"
                        "               Options: extlength_path\n"
                        "All predictors can be combined with one or more "
                        "wrapper predictors by adding the wrapper name "
                        "separated by a _ symbol. Following wrappers are "
                        "available:\n"
                        "* 'idxmap': Add this wrapper to predictors which use "
                        "an alternative word map."
                        "            Options: src_idxmap, trg_idxmap\n"
                        "* 'altsrc': This wrapper loads source sentences from "
                        "an alternative source.\n"
                        "            Options: altsrc_test\n"
                        "* 'unkvocab': This wrapper explicitly excludes "
                        "matching word indices higher than trg_vocab_size "
                        "with UNK scores.\n"
                        "             Options: trg_vocab_size\n"
                        "* 'fsttok': Uses an FST to transduce SGNMT tokens to "
                        "predictor tokens.\n"
                        "             Options: fsttok_path, "
                        "fsttok_max_pending_score, fst_unk_id\n"
                        "* 'word2char': Wraps word-level predictors when SGNMT"
                        " is running on character level.\n"
                        "            Options: word2char_map\n"
                        "* 'skipvocab': Skip a subset of the predictor "
                        "vocabulary.\n"
                        "               Options: skipvocab_max_id, "
                        "skipvocab_stop_size\n"
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
                        choices=['none', 'exact', 'reduced', 'rescale_unk'],
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
                        "* 'rescale_unk': Rescale UNK probabilities and "
                        "leave all other scores unmodified. Results in a "
                        "distribution if predictor scores are stochastic.\n"
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
    group.add_argument("--length_normalization", default=False, type='bool',
                        help="DEPRECATED. Synonym for --combination_scheme "
                        "length_norm. Normalize n-best hypotheses by sentence "
                        "length. Normally improves pure NMT decoding, but "
                        "degrades performance when combined with predictors "
                        "like fst or multiple NMT systems.")
    group.add_argument("--nmt_config", default="",
                        help="Defines the configuration of the NMT model. This "
                        "can either point to a configuration file, or it can "
                        "directly contain the parameters (e.g. 'src_vocab_size"
                        "=1234,trg_vocab_size=2345'). Use 'config_file=' in "
                        "the parameter string to use configuration files "
                        "with the second method.")
    group.add_argument("--nmt_path", default="",
                        help="Defines the path to the NMT model. If empty, "
                        "the model is loaded from the default location which "
                        "depends on the NMT engine")
    group.add_argument("--nmt_engine", default="blocks",
                        choices=['none', 'blocks', 'tensorflow'],
                        help="NMT implementation which should be used. "
                        "Use 'none' to disable NMT support.")
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
    group.add_argument("--gnmt_beta", default=0.0, type=float,
                       help="If this is greater than zero, add a coverage "
                       "penalization term following Googles NMT (Wu et al., "
                       "2016) to the NMT score.")
    group.add_argument("--t2t_usr_dir", default="",
                       help="Available for the t2t predictor. See the "
                       "--t2t_usr_dir argument in tensor2tensor.")
    group.add_argument("--t2t_model", default="transformer",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor model.")
    group.add_argument("--t2t_problem", default="translate_ende_wmt32k",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor problem.")
    group.add_argument("--t2t_hparams_set",
                       default="transformer_base_single_gpu",
                       help="Available for the t2t predictor. Name of the "
                       "tensor2tensor hparams set.")
    group.add_argument("--t2t_checkpoint_dir", default="",
                       help="Available for the t2t predictor. Path to the "
                       "tensor2tensor checkpoint directory. Same as "
                       "--output_dir in t2t_trainer.")
    group.add_argument("--t2t_src_vocab_size", default=30000, type=int,
                        help="T2T source vocabulary size")
    group.add_argument("--t2t_trg_vocab_size", default=30000, type=int,
                        help="T2T target vocabulary size")

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
    group.add_argument("--length_model_offset", default=0, type=int,
                        help="The target sentence length model is applied to "
                        "hypothesis length minus length_model_offst")
    group.add_argument("--extlength_path", default="",
                        help="Only required for the 'extlength' predictor. "
                        "This is the path to the file which specifies the "
                        "length distributions for each sentence. Each line "
                        "consists of blank separated '<length>:<logprob>' "
                        "pairs.")
    
    # UNK count predictors
    group = parser.add_argument_group('Count predictor options')
    group.add_argument("--unk_count_lambdas", default="1.0",
                        help="Model parameters for the UNK count model: comma-"
                        "separated list of lambdas for Poisson distributions. "
                        "The first float specifies the Poisson distribution "
                        "over the number of UNKs in the hypotheses given that "
                        "the number of UNKs on the source side is 0. The last "
                        "lambda specifies the distribution given >=n-1 UNKs "
                        "in the source sentence.")
    group.add_argument("--wc_word", default=-1, type=int,
                       help="If negative, the wc predictor counts all "
                       "words. Otherwise, count only the specific word")
    group.add_argument("--ngramc_path", default="ngramc/%d.txt",
                        help="Only required for ngramc predictor. The ngramc "
                        "predictor counts the number of ngrams and multiplies "
                        "them with the factors defined in the files. The "
                        "format is one ngram per line '<ngram> : <score>'. "
                        "You can use the placeholder %%d for the sentence "
                        "index.")
    group.add_argument("--ngramc_order", default=0, type=int,
                       help="If positive, count only ngrams of the specified "
                       "Order. Otherwise, count all ngrams")
    group.add_argument("--ngramc_discount_factor", default=-1.0, type=float,
                       help="If this is non-negative, discount ngram counts "
                       "by this factor each time the ngram is consumed")
    group.add_argument("--unkc_src_vocab_size", default=30003, type=int,
                        help="Vocabulary size for the unkc predictor.")
    group.add_argument("--skipvocab_max_id", default=30003, type=int,
                        help="All tokens above this threshold are skipped "
                        "by the skipvocab predictor wrapper.")
    group.add_argument("--skipvocab_stop_size", default=1, type=int,
                        help="The internal beam search of the skipvocab "
                        "predictor wrapper stops if the best stop_size "
                         "scores are for in-vocabulary words (ie. with index "
                         "lower or equal skipvocab_max_id")

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
    group.add_argument("--bow_heuristic_strategies", default="remaining",
                       help="Defines the form of heuristic estimates of the "
                       "bow predictor. Comma-separate following values:\n"
                       "* remaining: sum up unigram estimates for all words "
                       "in the bag which haven't been consumed\n"
                       "* consumed: Use the difference between the actual "
                       "hypothesis score and the sum of unigram estimates "
                       "of consumed words as score")
    group.add_argument("--bow_accept_subsets", default=False, type='bool',
                       help="If this is set to false, the bow predictor "
                       "enforces exact correspondence between bag and words "
                       "in complete hypotheses. If false, it ensures that "
                       "hypotheses are consistent with the bag (i.e. do not "
                       "contain words outside the bag) but do not necessarily "
                       "have all words in the bag")
    group.add_argument("--bow_accept_duplicates", default=False, type='bool',
                       help="If this is set to true, the bow predictor "
                       "allows a word in the bag to appear multiple times, "
                       "i.e. the exact count of the word is not enforced. "
                       "Can only be used in conjunction with bow_accept_subsets")
    group.add_argument("--bow_equivalence_vocab_size", default=-1, type=int,
                       help="If positive, bow predictor states are considered "
                       "equal if the the remaining words within that vocab "
                       "and OOVs regarding this vocab are the same. Only "
                       "relevant when using hypothesis recombination")
    group.add_argument("--bow_diversity_heuristic_factor", default=-1.0, type=float,
                       help="If this is greater than zero, promote diversity "
                       "between bags via the bow predictor heuristic. Bags "
                       "which correspond to bags of partial bags of full "
                       "hypotheses are penalized by this factor.")
    
    # Wrappers
    group = parser.add_argument_group('Wrapper predictor options')
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
    group.add_argument("--altsrc_test", default="test_en.alt",
                        help="Only required for altsrc wrapper predictor. Path"
                        " to the alternative source sentences.")
    group.add_argument("--word2char_map", default="word2char.map",
                        help="Only required for word2char wrapper predictor. "
                        "Path to a mapping file from word ID to sequence of "
                        "character IDs (format: <word-id> <char-id1> <char-id2"
                        ">...). All character IDs which do not occur in this "
                        "mapping are treated as word boundary symbols.")
    group.add_argument("--fsttok_path", default="tok.fst",
                        help="For the fsttok wrapper. Defines the path to the "
                        "FSt which transduces sequences of SGNMT tokens (eg. "
                        "characters) to predictor tokens (eg BPEs). FST may "
                        "be non-deterministic and contain epsilons.")
    group.add_argument("--fsttok_max_pending_score", default=5.0, type=float,
                       help="Applicable if an FST used by the fsttok wrapper "
                       "is non-deterministic. In this case, one predictor "
                       "state may correspond to multiple nodes in the FST. "
                       "We prune nodes which are this much worse than the "
                       "best scoring node with the same history.")

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
    group = parser.add_argument_group('(Neural) LM predictor options')
    group.add_argument("--srilm_path", default="lm/ngram.lm.gz",
                        help="Path to the ngram LM file in SRILM format")
    group.add_argument("--srilm_convert_to_ln", default=False,
                        help="Whether to convert srilm scores from log to ln.")
    group.add_argument("--nplm_path", default="nplm/nplm.gz",
                        help="Path to the NPLM language model")
    group.add_argument("--rnnlm_path", default="rnnlm/rnn.ckpt",
                        help="Path to the RNNLM language model")
    group.add_argument("--rnnlm_config", default="rnnlm.ini",
                        help="Defines the configuration of the RNNLM model. This"
                        " can either point to a configuration file, or it can "
                        "directly contain the parameters (e.g. 'src_vocab_size"
                        "=1234,trg_vocab_size=2345'). Use 'config_file=' in "
                        "the parameter string to use configuration files "
                        "with the second method. Use 'model_name=X' in the "
                        "parameter string to use one of the predefined models.")
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
    group.add_argument("--fst_skip_bos_weight", default=True, type='bool',
                        help="This option applies to fst and nfst "
                        "predictors. Lattices produced by HiFST contain the "
                        "<S> symbol and often have scores on the corresponding"
                        " arc. However, GNMT skips <S> and this score is not "
                        "regarded anywhere. Set this option to true to add the "
                        "<S> scores. This ensures that the "
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
                        "predictor inherits all previous settings except for "
                        "the ones in this parameter." % (w, w))
        group.add_argument("--nmt_path%s" % n, default="",
                        help="Overrides --nmt_path for the %s nmt" % w)
        group.add_argument("--nmt_engine%s" % n, default="",
                        help="Overrides --nmt_engine for the %s nmt" % w)
        group.add_argument("--t2t_model%s" % n, default="",
                        help="Overrides --t2t_model for the %s t2t predictor"
                        % w)
        group.add_argument("--t2t_problem%s" % n, default="",
                        help="Overrides --t2t_problem for the %s t2t predictor"
                        % w)
        group.add_argument("--t2t_hparams_set%s" % n, default="",
                        help="Overrides --t2t_hparams_set for the %s t2t "
                        "predictor" % w)
        group.add_argument("--t2t_checkpoint_dir%s" % n, default="",
                        help="Overrides --t2t_checkpoint_dir for the %s t2t "
                        "predictor" % w)
        group.add_argument("--t2t_src_vocab_size%s" % n, default=0, type=int,
                        help="Overrides --t2t_src_vocab_size for the %s t2t "
                        "predictor" % w)
        group.add_argument("--t2t_trg_vocab_size%s" % n, default=0, type=int,
                        help="Overrides --t2t_trg_vocab_size for the %s t2t "
                        "predictor" % w)
        group.add_argument("--rnnlm_config%s" % n,  default="",
                        help="If the --predictors string contains more than "
                        "one rnnlm predictor, you can specify the configuration "
                        "for the %s one with this parameter. The %s rnnlm "
                        "predictor inherits all previous settings except for "
                        "the ones in this parameter." % (w, w))
        group.add_argument("--rnnlm_path%s" % n, default="",
                        help="Overrides --rnnlm_path for the %s nmt" % w)
        group.add_argument("--src_test%s" % n, default="",
                        help="Overrides --src_test for the %s src" % w)                        
        group.add_argument("--altsrc_test%s" % n, default="",
                        help="Overrides --altsrc_test for the %s altsrc" % w)
        group.add_argument("--word2char_map%s" % n, default="",
                        help="Overrides --word2char_map for the %s word2char" % w)
        group.add_argument("--fsttok_path%s" % n, default="",
                        help="Overrides --fsttok_path for the %s fsttok" % w)
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
        group.add_argument("--ngramc_path%s" % n, default="",
                        help="Overrides --ngramc_path for the %s ngramc" % w)
        group.add_argument("--ngramc_order%s" % n, default=0, type=int,
                        help="Overrides --ngramc_order for the %s ngramc" % w)
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
    if args.legacy_indexing:
        args.indexing_scheme = "tf"
    if args.output_fst_unk_id:
        args.fst_unk_id = args.output_fst_unk_id 
    return args


def validate_args(args):
    """Some rudimentary sanity checks for configuration options.
    This method directly prints help messages to the user. In case of fatal
    errors, it terminates using ``logging.fatal()``
    
    Args:
        args (object):  Configuration as returned by ``get_args``
    """
    for depr in ['en_test', 'fr_test',
                 'length_normalization', 'legacy_indexing',
                 'en_idxmap', 'fr_idxmap']:
        if getattr(args, depr):
            logging.warn("Using deprecated argument %s. Please check the "
                         "documentation for the replacement." % depr)
    # Validate --range
    if args.range:
        if args.input_method == 'shell':
            logging.warn("The --range parameter can lead to unexpected "
                         "behavior in the 'shell' mode.")
        if ":" in args.range:
            try:
                f,t = [int(i) for i in args.range.split(":")]
                if f > t:
                    logging.fatal("Start index in range greater than end index")
            except:
                pass # Deal with it later
        
    # Some common pitfalls
    if args.input_method == 'dummy' and args.max_len_factor < 10:
        logging.warn("You are using the dummy input method but a low value "
                     "for max_len_factor (%d). This means that decoding will "
                     "not consider hypotheses longer than %d tokens. Consider "
                     "increasing max_len_factor to the length longest relevant"
                     " hypothesis" % (args.max_len_factor, args.max_len_factor))
    if (args.decoder == "beam" and args.combination_scheme == "length_norm"
                               and args.early_stopping):
        logging.warn("You are using beam search with length normalization but "
                     "with early stopping. All hypotheses found with beam "
                     "search with early stopping have the same length. You "
                     "might want to disable early stopping.")

