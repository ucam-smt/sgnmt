"""This module bypasses the normal predictor framework and decodes 
directly with ``blocks.search.BeamSearch``. This is the original beam
search implementation in the blocks library, and is much faster than
going through the NMT predictor as it parallelizes expanding the active 
hypotheses on the GPU. However, its less flexible because you can only
do pure single NMT decoding.
"""

from cam.sgnmt import utils
import logging
import numpy as np

from collections import Counter
from theano import tensor
from toolz import merge
import cam.sgnmt.decoding.core
from cam.sgnmt.decoding.decoder import Hypothesis

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model
from blocks.select import Selector
from blocks.search import BeamSearch

from cam.sgnmt.predictors.blocks_neural import LoadNMT
from cam.sgnmt.blocks.machine_translation.model import BidirectionalEncoder, \
                                                      Decoder


class BlocksVanillaDecoder(cam.sgnmt.decoding.core.Decoder):
    """Adaptor class for blocks.search.BeamSearch. We implement the
    ``Decoder`` class but ignore functionality for predictors or
    heuristics. Instead, we pass through decoding directly to the 
    blocks beam search module. This is fast, but breaks with the
    predictor framework. It can only be used for pure single system
    NMT decoding.
    """
    
    def __init__(self, nmt_model_path, config):
        """Set up the NMT model used by the decoder.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
            config (dict): NMT configuration
        """
        super(BlocksVanillaDecoder, self).__init__()
        self.config = config
        self.set_up_decoder(nmt_model_path)
    
    def set_up_decoder(self, nmt_model_path):
        """This method uses the NMT configuration in ``self.config`` to
        initialize the NMT model. This method basically corresponds to 
        ``blocks.machine_translation.main``.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
        """
        # Create Theano variables
        logging.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        sampling_input = tensor.lmatrix('input')
    
        # Construct model
        logging.info('Building RNN encoder-decoder')
        encoder = BidirectionalEncoder(self.config['src_vocab_size'],
                                       self.config['enc_embed'],
                                       self.config['enc_nhids'])
        decoder = Decoder(self.config['trg_vocab_size'],
                          self.config['dec_embed'],
                          self.config['dec_nhids'],
                          self.config['enc_nhids'] * 2)
        cost = decoder.cost(encoder.apply(source_sentence,
                                          source_sentence_mask),
                            source_sentence_mask,
                            target_sentence,
                            target_sentence_mask)
    
        logging.info('Creating computational graph')
        cg = ComputationGraph(cost)
    
        # Initialize model (TODO: do i really need this?)
        logging.info('Initializing model')
        encoder.weights_init = decoder.weights_init = IsotropicGaussian(
            self.config['weight_scale'])
        encoder.biases_init = decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()
    
        # apply dropout for regularization (TODO: remove?)
        if self.config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logging.info('Applying dropout')
            dropout_inputs = [x for x in cg.intermediary_variables
                                if x.name == 'maxout_apply_output']
            cg = apply_dropout(cg, dropout_inputs, self.config['dropout'])
    
        # Apply weight noise for regularization (TODO: remove?)
        if self.config['weight_noise_ff'] > 0.0:
            logging.info('Applying weight noise to ff layers')
            enc_params = Selector(encoder.lookup).get_params().values()
            enc_params += Selector(encoder.fwd_fork).get_params().values()
            enc_params += Selector(encoder.back_fork).get_params().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_params().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_params().values()
            dec_params += Selector(decoder.state_init).get_params().values()
            cg = apply_noise(cg,
                             enc_params+dec_params,
                             self.config['weight_noise_ff'])
    
        # Print shapes
        shapes = [param.get_value().shape for param in cg.parameters]
        logging.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logging.info('    {:15}: {}'.format(shape, count))
        logging.info("Total number of parameters: {}".format(len(shapes)))
    
        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logging.info("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logging.info('    {:15}: {}'.format(value.get_value().shape, name))
        logging.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))
    
        # Set up training model
        logging.info("Building model")
    
        # Set extensions
        logging.info("Initializing extensions")
    
        # Set up beam search and sampling computation graphs if necessary
        logging.info("Building sampling model")
        sampling_representation = encoder.apply(
                                    sampling_input,
                                    tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
            
        # Compare with blocks.machine_translation.BleuValidator.__init__
        self.source_sentence = sampling_input
        self.samples = samples
        self.model = search_model
        self.normalize = True
        self.verbose = self.config.get('val_set_out', None)

        # Reload model if necessary
        if self.config['reload']:
            loader = LoadNMT(nmt_model_path,
                             self.config['saveto'],
                             search_model)
            loader.load_weights()
            
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(samples=samples)
    
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
        seq = self._oov_to_unk(src_sentence, 
                               self.config['src_vocab_size'], 
                               utils.UNK_ID) + [utils.EOS_ID]
        input_ = np.tile(seq, (self.config['beam_size'], 1))
        trans, costs = self.beam_search.search(
                    input_values={self.source_sentence: input_},
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

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]
    
    
    def has_predictors(self):
        """Always returns true. """
        return True
