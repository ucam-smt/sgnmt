"""This module bypasses the normal predictor framework and decodes 
directly with ``blocks.search.BeamSearch``. This is the original beam
search implementation in the blocks library, and is much faster than
going through the NMT predictor as it parallelizes expanding the active 
hypotheses on the GPU. However, its less flexible because you can only
do pure single NMT decoding.
"""

from blocks.search import BeamSearch

from cam.sgnmt import utils
from cam.sgnmt.blocks.model import NMTModel, LoadNMTUtils
from cam.sgnmt.blocks.sparse_search import SparseBeamSearch
from cam.sgnmt.decoding.core import Decoder
from cam.sgnmt.decoding.core import Hypothesis
from cam.sgnmt.misc.sparse import FlatSparseFeatMap
import numpy as np


class BlocksNMTVanillaDecoder(Decoder):
    """Adaptor class for blocks.search.BeamSearch. We implement the
    ``Decoder`` class but ignore functionality for predictors or
    heuristics. Instead, we pass through decoding directly to the 
    blocks beam search module. This is fast, but breaks with the
    predictor framework. It can only be used for pure single system
    NMT decoding. Note that this decoder supports sparse feat maps
    on both source and target side.
    """
    
    def __init__(self, nmt_model_path, config, decoder_args):
        """Set up the NMT model used by the decoder.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
            config (dict): NMT configuration
            decoder_args (object): Decoder configuration passed through
                                   from configuration API.
        """
        super(BlocksNMTVanillaDecoder, self).__init__(decoder_args)
        self.config = config
        self.set_up_decoder(nmt_model_path)
        self.src_eos = self.src_sparse_feat_map.word2dense(utils.EOS_ID)
    
    def set_up_decoder(self, nmt_model_path):
        """This method uses the NMT configuration in ``self.config`` to
        initialize the NMT model. This method basically corresponds to 
        ``blocks.machine_translation.main``.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
        """
        self.nmt_model = NMTModel(self.config)
        self.nmt_model.set_up()
        loader = LoadNMTUtils(nmt_model_path,
                              self.config['saveto'],
                              self.nmt_model.search_model)
        loader.load_weights()
        self.src_sparse_feat_map = self.config['src_sparse_feat_map'] \
                if self.config['src_sparse_feat_map'] else FlatSparseFeatMap()
        if self.config['trg_sparse_feat_map']:
            self.trg_sparse_feat_map = self.config['trg_sparse_feat_map']
            self.beam_search = SparseBeamSearch(
                                 samples=self.nmt_model.samples, 
                                 trg_sparse_feat_map=self.trg_sparse_feat_map) 
        else:
            self.trg_sparse_feat_map = FlatSparseFeatMap()
            self.beam_search = BeamSearch(samples=self.nmt_model.samples)
    
    def decode(self, src_sentence):
        """Decodes a single source sentence with the original blocks
        beam search decoder. Does not use predictors. Note that the
        score breakdowns in returned hypotheses are only on the 
        sentence level, not on the word level. For finer grained NMT
        scores you need to use the nmt predictor. ``src_sentence`` is a
        list of source word ids representing the source sentence without
        <S> or </S> symbols. As blocks expects to see </S>, this method
        adds it automatically.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        seq = self.src_sparse_feat_map.words2dense(utils.oov_to_unk(
                src_sentence,
                self.config['src_vocab_size'])) + [self.src_eos]
        if self.src_sparse_feat_map.dim > 1: # sparse src feats
            input_ = np.transpose(
                            np.tile(seq, (self.config['beam_size'], 1, 1)),
                            (2,0,1))
        else: # word ids on the source side
            input_ = np.tile(seq, (self.config['beam_size'], 1))
        trans, costs = self.beam_search.search(
                    input_values={self.nmt_model.sampling_input: input_},
                    max_length=3*len(src_sentence),
                    eol_symbol=utils.EOS_ID,
                    ignore_first_eol=True)
        hypos = []
        max_len = 0
        for idx in xrange(len(trans)):
            max_len = max(max_len, len(trans[idx]))
            hypo = Hypothesis(trans[idx], -costs[idx])
            hypo.score_breakdown = len(trans[idx]) * [[(0.0,1.0)]]
            hypo.score_breakdown[0] = [(-costs[idx],1.0)]
            hypos.append(hypo)
        self.apply_predictors_count = max_len * self.config['beam_size']
        return hypos
    
    def has_predictors(self):
        """Always returns true. """
        return True
