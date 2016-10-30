"""This module implements word alignment with the standard attentional
NMT architecture according Bahdanau et. al., 2015. It relies on the
predictor framework to perform forced NMT decoding and captures the 
alignment weights during this process. The alignment weights are then
interpreted as alignment link strengths.
"""

from cam.sgnmt.blocks.nmt import get_nmt_model_path
from cam.sgnmt.predictors.blocks_nmt import BlocksNMTPredictor
import numpy as np
import logging

def align_with_nmt(config, args):
    """Main method for NMT based alignment. Aligns the corpus using the
    alignment weights from NMT forced decoding.
    
    Args:
        config (dict): NMT configuration
        args (object): ArgumentParser object containing the command
                       line arguments
    
    Returns:
        list. List of alignments, where alignments are represented as
        numpy matrices containing confidences between 0 and 1.
    """
    predictor = BlocksNMTPredictor(get_nmt_model_path(args.nmt_model_selector,
                                                      config), 
                                   False, 
                                   config)
    alignments = []
    with open(config['src_data']) as src:
        with open(config['trg_data']) as trg:
            src_line = src.readline()
            trg_line = trg.readline()
            while src_line and trg_line:
                logging.info("Align sentence pair %s <-> %s" % (
                         src_line.strip(), trg_line.strip()))
                src_sen = [int(w) for w in src_line.strip().split()]
                trg_sen = [int(w) for w in trg_line.strip().split()]
                src_len = len(src_sen)
                trg_len = len(trg_sen)
                matrix = np.zeros((src_len, trg_len))
                predictor.initialize(src_sen)
                for pos in xrange(trg_len):
                    predictor.predict_next()
                    predictor.consume(trg_sen[pos])
                    matrix[:,pos] = predictor.states['weights'][0][:src_len]
                alignments.append(matrix)
                src_line = src.readline()
                trg_line = trg.readline()
    return alignments
