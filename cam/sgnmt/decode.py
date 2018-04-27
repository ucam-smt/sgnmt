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

import logging
import os
import sys
import codecs

from cam.sgnmt import utils
from cam.sgnmt import decode_utils
from cam.sgnmt.ui import get_args, get_parser

# Load configuration from command line arguments or configuration file
args = get_args()
decode_utils.base_init(args)


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
    print("!sgnmt quit                   Quit SGNMT")
    print("!sgnmt help                   Print this help")


utils.load_src_wmap(args.src_wmap)
utils.load_trg_wmap(args.trg_wmap)
utils.load_trg_cmap(args.trg_cmap)
decoder = decode_utils.create_decoder()
outputs = decode_utils.create_output_handlers()

if args.input_method == 'file':
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
                elif cmd == "decode":
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
                            decoder = decode_utils.create_decoder(args)
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
