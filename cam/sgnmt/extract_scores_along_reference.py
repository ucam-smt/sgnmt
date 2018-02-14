#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This script can be used to extract the full predictor posteriors
along a decoding path. This is similar to forced decoding which
also prints out the predictor scores for all other words at each
time step along the reference. This can be useful for tuning predictor
weights.

The script supports most of the command-line arguments of
``decode.py`` except for the following ones:
- decoder (and all arguemnts in the 'Decoding' group)
- heuristics (no heuristic necessary for forced decoding)
- src_test2 (usr altest predictor wrapper for multiple input
  streams)
- outputs (only posteriors,json is produced)
- trg/src_cmap/wmap: Use indexed data only
"""

import logging
import sys
import traceback

from cam.sgnmt.decoding import core
from cam.sgnmt import decode_utils
from cam.sgnmt.ui import get_args

# Load configuration from command line arguments or configuration file
args = get_args()
decode_utils.base_init(args)

class ForcedDecoder(core.Decoder):
    """Forced decoder implementation. The decode() function returns
    the same hypos as the GreedyDecoder with forced predictor. However,
    this implementation keeps track of all posteriors along the way,
    which are dumped to the file system afterwards.
    """
    
    def __init__(self, decoder_args):
        """Initialize the greedy decoder. """
        super(ForcedDecoder, self).__init__(decoder_args)
        self.extracted_data = []

    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        # TODO


decoder = ForcedDecoder(args) 
decode_utils.add_predictors(decoder)
# Update start sentence id if necessary
if args.range:
    idx,_ = args.range.split(":") if (":" in args.range) else (args.range,0)  
    decoder.set_start_sen_id(int(idx)-1) # -1 because indices start with 1

with open(args.src_test) as f:
    src_sentences = [line.strip().split() for line in f]
    for sen_idx in decode_utils.get_sentence_indices(args.range, src_sentences):
        try:
            src = src_sentences[sen_idx]
            # TODO
        except Exception as e:
            logging.error("An unexpected %s error has occurred at sentence id "
                          "%d: %s, Stack trace: %s" % (sys.exc_info()[0],
                                                       sen_idx+1,
                                                       e,
                                                       traceback.format_exc()))

