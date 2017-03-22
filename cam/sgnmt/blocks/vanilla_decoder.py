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
import logging
from theano import config


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


class BlocksNMTEnsembleVanillaDecoder(Decoder):
    """Vanilla NMT decoder for examples. This still bypasses the 
    predictor framework but can handle multiple NMT systems. Note that
    this decoder refrains from the vocabulary matching mechanisms used
    in the predictor framework in favor of decoding speed. Therefore,
    this decoder should be used only with models of the same vocabulary
    size.
    """
    
    def __init__(self, nmt_specs, decoder_args):
        """Set up the NMT model used by the decoder.
        
        Args:
            nmt_specs (list):  List of tuples of which the first element
                               is the path to the NMT model (.npz) file
                               and the second is the NMT configuration
            decoder_args (object): Decoder configuration passed through
                                   from configuration API.
        """
        super(BlocksNMTEnsembleVanillaDecoder, self).__init__(decoder_args)
        self.n_networks = len(nmt_specs)
        if self.n_networks < 2:
            logging.fatal("The NMT ensemble vanilla decoder needs at "
                          "least two NMT systems")
        global_config = nmt_specs[0][1]
        self.src_vocab_size = global_config['src_vocab_size']
        self.beam_size = global_config['beam_size']
        self.src_sparse_feat_map = global_config['src_sparse_feat_map'] \
            if global_config['src_sparse_feat_map'] else FlatSparseFeatMap()
        if global_config['trg_sparse_feat_map']:
            logging.fatal("Using sparse feature maps on the target size is "
                          "currently not supported by the ensemble vanilla "
                          "decoder")
        self.trg_sparse_feat_map = FlatSparseFeatMap()  
        self.set_up_decoder(nmt_specs)
        self.src_eos = self.src_sparse_feat_map.word2dense(utils.EOS_ID)
    
    def set_up_decoder(self, nmt_specs):
        """This method sets up a list of NMT models and BeamSearch 
        instances, one for each model in the ensemble. Note that we do
        not use the ``BeamSearch.search`` method for ensemble decoding
        directly.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
        """
        self.nmt_models = []
        self.beam_searches = []
        for nmt_model_path, nmt_config in nmt_specs:
            nmt_model = NMTModel(nmt_config)
            nmt_model.set_up()
            loader = LoadNMTUtils(nmt_model_path,
                                  nmt_config['saveto'],
                                  nmt_model.search_model)
            loader.load_weights()
            self.nmt_models.append(nmt_model)
            self.beam_searches.append(BeamSearch(samples=nmt_model.samples))
    
    def has_predictors(self):
        """Always returns true. """
        return True

    def decode(self, src_sentence):
        """This is a generalization to NMT ensembles of 
        ``BeamSearch.search``.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        for search in self.beam_searches:
            if not search.compiled:
                search.compile()
        seq = self.src_sparse_feat_map.words2dense(utils.oov_to_unk(
                src_sentence,
                self.src_vocab_size)) + [self.src_eos]
        if self.src_sparse_feat_map.dim > 1: # sparse src feats
            input_ = np.transpose(
                            np.tile(seq, (self.beam_size, 1, 1)),
                            (2,0,1))
        else: # word ids on the source side
            input_ = np.tile(seq, (self.beam_size, 1))

        contexts_and_states = []
        for sys_idx in xrange(self.n_networks):
            contexts, states, _ = \
                self.beam_searches[sys_idx].compute_initial_states_and_contexts(
                            {self.nmt_models[sys_idx].sampling_input: input_})
            contexts_and_states.append((contexts, 
                                        states, 
                                        self.beam_searches[sys_idx]))

        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = states['outputs'][None, :]
        all_masks = np.ones_like(all_outputs, dtype=config.floatX)
        all_costs = np.zeros_like(all_outputs, dtype=config.floatX)

        for i in range(3*len(src_sentence)):
            if all_masks[-1].sum() == 0:
                break
            logprobs_lst = []
            for contexts, states, search in contexts_and_states:
                logprobs_lst.append(search.compute_logprobs(contexts, states))
            
            logprobs = np.sum(logprobs_lst, axis=0)
            next_costs = (all_costs[-1, :, None] +
                          logprobs * all_masks[-1, :, None])
            (finished,) = np.where(all_masks[-1] == 0)
            next_costs[finished, :utils.EOS_ID] = np.inf
            next_costs[finished, utils.EOS_ID + 1:] = np.inf

            # The `i == 0` is required because at the first step the beam
            # size is effectively only 1.
            (indexes, outputs), chosen_costs = BeamSearch._smallest(
                next_costs, self.beam_size, only_first_row=i == 0)

            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]
            
            # Rearrange everything
            for contexts, states, search in contexts_and_states:
                for name in states:
                    states[name] = states[name][indexes]
                states.update(search.compute_next_states(contexts, 
                                                         states, 
                                                         outputs))
            
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != utils.EOS_ID
            if i == 0:
                mask[:] = 1
            all_masks = np.vstack([all_masks, mask[None, :]])

        all_outputs = all_outputs[1:]
        all_masks = all_masks[:-1]
        all_costs = all_costs[1:] - all_costs[:-1]
        result = all_outputs, all_masks, all_costs
        trans, costs = BeamSearch.result_to_lists(result)
        hypos = []
        max_len = 0
        for idx in xrange(len(trans)):
            max_len = max(max_len, len(trans[idx]))
            hypo = Hypothesis(trans[idx], -costs[idx])
            hypo.score_breakdown = len(trans[idx]) * [[(0.0,1.0)]]
            hypo.score_breakdown[0] = [(-costs[idx],1.0)]
            hypos.append(hypo)
        self.apply_predictors_count = max_len * self.beam_size
        return hypos


