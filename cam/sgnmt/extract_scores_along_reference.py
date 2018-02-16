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
import time
import numpy as np
import pickle

from cam.sgnmt.decoding import core
from cam.sgnmt import decode_utils
from cam.sgnmt import utils
from cam.sgnmt.ui import get_args
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor

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
        """Initialize the decoder and load target sentences."""
        super(ForcedDecoder, self).__init__(decoder_args)
        self.trg_sentences = load_sentences(decoder_args.trg_test, "target")

    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        trg_sentence = self.trg_sentences[self.current_sen_id] + [utils.EOS_ID]
        score_breakdown = []
        score = 0.0
        all_posteriors = []
        all_unk_scores = []
        for trg_word in trg_sentence:
            self.apply_predictors_count += 1
            breakdown = []
            posteriors = []
            unk_scores = []
            for (p, w) in self.predictors:
                if isinstance(p, UnboundedVocabularyPredictor):
                    posterior = p.predict_next([trg_word])
                else: 
                    posterior = p.predict_next()
                unk_prob = p.get_unk_probability(posterior)
                pred_score = utils.common_get(posterior, trg_word, unk_prob)
                breakdown.append((pred_score, w))
                score += pred_score * w
                posteriors.append(posterior)
                unk_scores.append(unk_prob)
            all_posteriors.append(posteriors)
            all_unk_scores.append(unk_scores)
            score_breakdown.append(breakdown)
            self.consume(trg_word)
        self.add_full_hypo(core.Hypothesis(trg_sentence, score, score_breakdown))
        self.last_meta_data = {
            "src_sentence": np.array(src_sentence + [utils.EOS_ID]),
            "trg_sentence": np.array(trg_sentence),
            "posteriors": all_posteriors,
            "unk_scores": all_unk_scores
        }
        return self.full_hypos

def load_sentences(path, name="source"):
    """Loads sentences from a plain (indexed) text file.

    Args:
        path (string): Path to the text file
        name (string): Name for error messages.
    """
    if not path:
        logging.fatal("Please specify the path to the %s sentences."
                      % name)
    else:
        try:
            with open(path) as f:
                return [map(int, line.strip().split()) for line in f]
        except ValueError:
            logging.fatal("Non-numeric characters in %s sentence file %s"
                          % (name, path))
        except IOError:
            logging.fatal("Could not read %s sentence file %s"
                          % (name, path))


INDENT = 2
SPACE = " "
NEWLINE = "\n"

def to_json(o, level=0):
    """ Adapted from

    https://stackoverflow.com/questions/21866774/pretty-print-json-dumps 
    """
    ret = ""
    if isinstance(o, dict):
        if level < 2:
            ret += "{" + NEWLINE
            comma = ""
            for k,v in sorted(o.iteritems()):
                ret += comma
                comma = ",\n"
                ret += SPACE * INDENT * (level+1)
                ret += '"' + str(k) + '":' + NEWLINE + SPACE * INDENT * (level+1)
                ret += to_json(v, level + 1)
            ret += NEWLINE + SPACE * INDENT * level + "}"
        else:
            ret += "{" + ", ".join(['"%s": %s' % (str(k), to_json(v, level+1)) 
                   for k,v in sorted(o.iteritems())
               ]) + "}"
    elif isinstance(o, basestring):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + NEWLINE
        comma = ""
        for e in o:
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level+1)
            ret += to_json(e, level + 1)
        ret += NEWLINE + SPACE * INDENT * level + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, np.float32) or isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ', '.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


decoder = ForcedDecoder(args) 
decode_utils.add_predictors(decoder)
# Update start sentence id if necessary
if args.range:
    idx,_ = args.range.split(":") if (":" in args.range) else (args.range,0)  
    decoder.set_start_sen_id(int(idx)-1) # -1 because indices start with 1

if "pickle" in args.outputs:
    out_format = "pickle"
    mode = "wb"
else:
    out_format = "json"
    mode = "w"

logging.info("Output format: %s" % out_format)

if '%s' in args.output_path:
    out_path = args.output_path % out_format
else:
    out_path = args.output_path


with open(out_path, mode) as writer:
    if out_format == "json":
        writer.write("[\n")
        json_comma = ""
    else:
        all_meta_data = []
    src_sentences = load_sentences(args.src_test, "source")
    for sen_idx in decode_utils.get_sentence_indices(args.range, src_sentences):
        try:
            src = src_sentences[sen_idx]
            logging.info("Next sentence (ID: %d): %s" 
                         % (sen_idx + 1, ' '.join(map(str, src))))
            start_hypo_time = time.time()
            decoder.apply_predictors_count = 0
            hypos = [hypo for hypo in decoder.decode(src)]
            logging.info("Decoded (ID: %d): %s" % (
                    sen_idx+1,
                    " ".join(map(str, hypos[0].trgt_sentence))))
            logging.info("Stats (ID: %d): score=%f "
                         "num_expansions=%d "
                         "time=%.2f" % (sen_idx+1,
                                        hypos[0].total_score,
                                        decoder.apply_predictors_count,
                                        time.time() - start_hypo_time))
        except Exception as e:
            logging.error("An unexpected %s error has occurred at sentence id "
                          "%d: %s, Stack trace: %s" % (sys.exc_info()[0],
                                                       sen_idx+1,
                                                       e,
                                                       traceback.format_exc()))
        if out_format == "json":
            writer.write(json_comma + to_json(
                    decoder.last_meta_data).replace("inf", "Infinity"))
            json_comma = ",\n"
        else:
            all_meta_data.append(decoder.last_meta_data)
    if out_format == "json":
        writer.write("\n]")
    else:
        pickle.dump(all_meta_data, writer)

