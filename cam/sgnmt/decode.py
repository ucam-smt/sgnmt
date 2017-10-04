#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This is the main runner script for SGNMT decoding. 
SGNMT can run in three different modes. The standard mode 'file' reads
sentences to translate from a plain text file. The mode 'stdin' can be
used to parse stdin. The last mode 'shell' enables interactive inter-
action with SGNMT via keyboard. For detailed usage descriptions please
visit the tutorial home page:

http://ucam-smt.github.io/tutorial/sgnmt 
"""

import codecs
import logging
import os
import sys

from cam.sgnmt import utils
from cam.sgnmt.decoding import core
from cam.sgnmt import decode_utils
from cam.sgnmt.ui import get_args, get_parser, validate_args


# UTF-8 support
if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

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


def _update_decoder(decoder, key, val):
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
        decoder = decode_utils.create_decoder(args)
    return decoder


def _process_inputs():
    """Helper method to support multiple input files."""
    inputfiles = [ args.src_test ]
    while True:
        inputfile = getattr(args, "src_test%d" % (len(inputfiles)+1), None)
        if not inputfile:
            break
        inputfiles.append(inputfile)
    # Read all input files
    inputs_tmp = [ [] for i in xrange(len(inputfiles)) ]
    for i in xrange(len(inputfiles)):
        with codecs.open(inputfiles[i], encoding='utf-8') as f:
            for line in f:
                inputs_tmp[i].append(line.strip().split())
    # Gather multiple input sentences for each line
    inputs = []
    for i in xrange(len(inputs_tmp[0])):
        input_lst = []
        for j in xrange(len(inputfiles)):
            input_lst.append(inputs_tmp[j][i])
        inputs.append(input_lst)
    return inputs


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


utils.load_src_wmap(args.src_wmap)
utils.load_trg_wmap(args.trg_wmap)
utils.load_trg_cmap(args.trg_cmap)
decoder = decode_utils.create_decoder(args)
outputs = decode_utils.create_output_handlers()

if args.input_method == 'file':
    # Check for additional input files
    if getattr(args, "src_test2"):
        decode_utils.do_decode(decoder, outputs, _process_inputs())
    else:
        with codecs.open(args.src_test, encoding='utf-8') as f:
            decode_utils.do_decode(decoder,
                                   outputs,
                                   [line.strip().split() for line in f])
elif args.input_method == 'dummy':
    decode_utils.do_decode(decoder, outputs, False)
else: # Interactive mode: shell or stdin
    print("Start interactive mode.")
    print("PID: %d" % os.getpid())
    print("Test sentences are read directly from stdin.")
    print("!sgnmt help lists all available directives")
    print("Quit with ctrl-c or !sgnmt quit")
    quit_sgnmt = False
    sys.stdout.flush()
    while not quit_sgnmt:
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
                        decode_utils.do_decode(
                            decoder, outputs,
                            [line.strip().split() for line in f])
                elif cmd == "quit":
                    quit_sgnmt = True
                elif cmd == "config":
                    if len(input_) == 2:
                        get_parser().print_help()
                    elif len(input_) >= 4:
                        key,val = (input_[2], ' '.join(input_[3:]))
                        setattr(args, key, val) # TODO: non-string args!
                        outputs = decode_utils.create_output_handlers()
                        if not key in ['outputs', 'output_path']:
                            decoder = _update_decoder(decoder, key, val)
                    else:
                        logging.error("Could not parse SGNMT directive")
                else:
                    logging.error("Unknown directive '%s'. Use '!sgnmt help' "
                                  "for help or exit with '!sgnmt quit'" % cmd)
            elif input_[0] == 'quit' or input_[0] == 'exit':
                quit_sgnmt = True
            else: # Sentence to translate
                decode_utils.do_decode(decoder, outputs, [input_])
        except:
            logging.error("Error in last statement: %s" % sys.exc_info()[0])
        sys.stdout.flush()
