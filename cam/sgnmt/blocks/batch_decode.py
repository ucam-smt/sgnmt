"""This is a highly optimized version of single NMT beam search
decoding for blocks. It runs completely separated from the rest of
SGNMT and does not use the predictor frameworks or the ``Decoder``
search strategy abstraction.
"""

import logging
import time
import pprint

from cam.sgnmt.blocks.model import NMTModel, LoadNMTUtils
from cam.sgnmt.blocks.nmt import blocks_get_default_nmt_config, \
                                 get_nmt_model_path_best_bleu
from cam.sgnmt.ui import get_batch_decode_parser
from cam.sgnmt import utils
from blocks.search import BeamSearch
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

state_names = ['outputs', 'states', 'weights', 'weighted_averages']

parser = get_batch_decode_parser()
args = parser.parse_args()


def load_sentences(path, _range, src_vocab_size):
    """Loads the source sentences to decode.
    
    Args:
        path (string): path to the plain text file with indexed
                       source sentences
        _range (string): Range argument
        src_vocab_size (int): Source language vocabulary size
    
    Returns:
        list. List of tuples, the first element is the sentence ID and
        the second element is a list of integers representing the
        sentence ending with EOS.
    """
    seqs = []
    seq_id = 1
    with open(path) as f:
        for line in f:
            seq = [int(w) for w in line.strip().split()]
            seqs.append((
                    seq_id,
                    utils.oov_to_unk(seq, src_vocab_size) + [utils.EOS_ID]))
            seq_id += 1
    if _range:
        try:
            if ":" in args.range:
                from_idx,to_idx = args.range.split(":")
            else:
                from_idx = int(args.range)
                to_idx = from_idx
            return seqs[int(from_idx)-1:int(to_idx)]
        except Exception as e:
            logging.fatal("Invalid value for --range: %s" % e)
    return seqs


def compute_encoder(sentences, attendeds, initial_states):
    """Computes the contexts and initial states for the given sentences
    and adds them to the contexts and initial_states list
    """
    contexts, states, _ = beam_search.compute_initial_states_and_contexts(
                                        {nmt_model.sampling_input: sentences})
    attendeds.extend(np.transpose(contexts['attended'], (1, 0, 2)))
    for n in state_names:
        initial_states[n].extend(states[n])


class Pipeline(object):
    
    def __init__(self, src_sentences):
        self.src_sentences = src_sentences
        self.pos = 0
        

# MAIN ENTRY POINT

# Get configuration
config = blocks_get_default_nmt_config()
for k in dir(args):
    if k in config:
        config[k] = getattr(args, k)
logging.info("Model options:\n{}".format(pprint.pformat(config)))

nmt_model = NMTModel(config)
nmt_model.set_up()

loader = LoadNMTUtils(get_nmt_model_path_best_bleu(config),
                      config['saveto'],
                      nmt_model.search_model)
loader.load_weights()

src_sentences = load_sentences(args.src_test,
                               args.range,
                               config['src_vocab_size'])
n_sentences = len(src_sentences)

logging.info("%d source sentences loaded. Initialize decoding.." 
                    % n_sentences)

beam_search = BeamSearch(samples=nmt_model.samples)
beam_search.compile()
enc_max_words = args.enc_max_words
dec_batch_size = args.dec_batch_size

# Compute contexts and initial states
start_time = time.time()
logging.info("Start time: %s" % start_time)

logging.info("Sort sentences, longest sentence first...")
src_sentences.sort(key=lambda x: len(x[1]), reverse=True)

logging.info("Compute all initial states and contexts...")

attendeds = []
initial_states = {n: [] for n in state_names}
start_pos = 0
cur_len = len(src_sentences[0][1])
cur_n_words = cur_len
for pos in xrange(1, n_sentences):
    this_len = len(src_sentences[pos][1])
    if this_len != cur_len or cur_len + cur_n_words <= enc_max_words:
        # Construct batch and compute contexts and initial states
        compute_encoder(src_sentences[start_pos:pos],
                        attendeds,
                        initial_states)
        cur_len = this_len
        start_pos = pos
        cur_n_words = 0
    cur_n_words += cur_len

compute_encoder(src_sentences[start_pos:],
                attendeds,
                initial_states)
