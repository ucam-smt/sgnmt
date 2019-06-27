#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the main runner script for SGNMT decoding. 
SGNMT can run in three different modes. The standard mode 'file' reads
sentences to translate from a plain text file. The mode 'stdin' can be
used to parse stdin. The last mode 'shell' enables interactive inter-
action with SGNMT via keyboard. For detailed usage descriptions please
visit the tutorial home page:

http://ucam-smt.github.io/sgnmt/html/tutorial.html
"""

import logging
import os
import sys
from cmd import Cmd

from cam.sgnmt import utils, io
from cam.sgnmt import decode_utils
from cam.sgnmt.ui import get_args, get_parser, run_diagnostics

# Load configuration from command line arguments or configuration file
args = get_args()
decode_utils.base_init(args)


class SGNMTPrompt(Cmd):

    def default(self, cmd_args):
        """Translate a single sentence."""
        decode_utils.do_decode(
            decoder, outputs,
            [cmd_args.strip()])

    def emptyline(self):
        pass

    def do_translate(self, cmd_args):
        """Translate a single sentence."""
        decode_utils.do_decode(
            decoder, outputs,
            [cmd_args.strip()])

    def do_diagnostics(self, cmd_args):
        """Run diagnostics to check which external libraries are
        available to SGNMT."""
        run_diagnostics()

    def do_config(self, cmd_args):
        """Change SGNMT configuration. Syntax: 'config <key> <value>.
        For most configuration changes the decoder needs to be
        rebuilt.
        """
        global outputs, decoder, args
        split_args = cmd_args.split()
        if len(split_args) < 2:
            print("Syntax: 'config <key> <new-value>'")
        else:
            key, val = (split_args[0], ' '.join(split_args[1:]))
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    if val == "true":
                        val = True
                    elif val == "false":
                        val = False
            setattr(args, key, val)
            print("Setting %s=%s..." % (key, val))
            outputs = decode_utils.create_output_handlers()
            if key in ["wmap", "src_wmap", "trg_wmap", 
                       "preprocessing", "postprocessing", "bpe_codes"]:
                io.initialize(args)
            elif not key in ['outputs', 'output_path']:
                decoder = decode_utils.create_decoder()

    def do_quit(self, cmd_args):
        """Quits SGNMT."""
        raise SystemExit

    def do_EOF(self, line):
        "Quits SGNMT"
        print("quit")
        return True


io.initialize(args)
decoder = decode_utils.create_decoder()
outputs = decode_utils.create_output_handlers()

if args.input_method == 'file':
    if os.access(args.src_test, os.R_OK):
        with open(args.src_test) as f:
            decode_utils.do_decode(decoder,
                                   outputs,
                                   [line.strip() for line in f])
    else:
        logging.fatal("Input file '%s' not readable. Please double-check the "
                      "src_test option or choose an alternative input_method."
                      % args.src_test)
elif args.input_method == 'dummy':
    decode_utils.do_decode(decoder, outputs, False)
elif args.input_method == "stdin":
    decode_utils.do_decode(decoder,
                           outputs,
                           [line.strip() for line in sys.stdin])
else: # Interactive mode: shell
    print("Starting interactive mode...")
    print("PID: %d" % os.getpid())
    print("Display help with 'help'")
    print("Quit with ctrl-d or 'quit'")
    prompt = SGNMTPrompt()
    prompt.prompt = "sgnmt> "
    prompt.cmdloop()

