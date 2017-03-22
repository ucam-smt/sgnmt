"""This file contains the ``NMTModel`` class which is the main 
interface to the NMT implementations in Blocks. All main runner scripts
(``train.py``, ``decode.py``, and ``align.py``) access Blocks NMT 
models through this class. According to the NMT config, it composes the
final NMT model using an encoder, attention, and decoder network which
are implemented in the modules ``encoder.py``, ``attention.py``, and
``decoder.py``.
"""

from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model
from collections import Counter
import logging
import os

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.select import Selector
from theano import tensor
from toolz import merge

from cam.sgnmt.blocks.checkpoint import SaveLoadUtils
from cam.sgnmt.blocks.decoder import Decoder, NoLookupDecoder
from cam.sgnmt.blocks.encoder import BidirectionalEncoder, NoLookupEncoder, \
    HierarchicalAnnotator, EncoderWithAnnotators, DeepBidirectionalEncoder


class LoadNMTUtils(SaveLoadUtils):
    """Loads parameters log and iterations state. This class is adapted
    from the ``LoadNMT`` class in the blocks example and contains
    some copied code. Note that we do not use BLOCKS_DELIMITER.
    Instead, we always use '-' to keep back compatibility with older
    models.
    """

    def __init__(self, nmt_model_path, saveto, model, **kwargs):
        """Initializes the path to the training directory and the NMT
        model.
        
        Args:
            nmt_model_path (string): Path to the model npz file
            saveto (string): Path to the NMT training directory (see
                             the saveto field in the NMT configuration
        """
        self.nmt_model_path = nmt_model_path
        self.folder = saveto
        self.model = model
        super(LoadNMTUtils, self).__init__(**kwargs)

    def load_weights(self):
        """Load the model parameters from the model file. Compare with
        ``blocks.machine_translation.LoadNMT``.
        """
        if not os.path.exists(self.path_to_folder):
            logging.info("No dump found")
            return
        logging.info("Loading the model from {}".format(self.nmt_model_path))
        try:
            logging.debug(" ...loading model parameters")
            params_all = self.load_parameters()
            params_this = self.model.get_parameter_dict()
            missing = set(params_this.keys()) - set(params_all.keys())
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    if params_this[pname].get_value().shape != val.shape:
                        logging.warning(
                            " Dimension mismatch {}-{} for {}"
                            .format(params_this[pname].get_value().shape,
                                    val.shape, pname))

                    params_this[pname].set_value(val)
                    logging.debug(" Loaded to CG {:15}: {}"
                                .format(val.shape, pname))
                else:
                    logging.warning(
                        " Parameter does not exist: {}".format(pname))
            logging.info(
                "Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logging.error(" Error {0}".format(str(e)))

    def load_parameters(self):
        """Currently not used, kept for consistency with blocks
        reference implementation. """
        return self.load_parameter_values(self.nmt_model_path)


class NMTModel:
    """Encapsulates an NMT model. Calling ``set_up`` will initialize
    all the attributes.
    
    Attributes:
        samples (Variable): Samples variable used for search algorithms
        cost (Variable): Model cost
        search_model (Variable): Decoding model
        training_model (Variable): Training model
        sampling_input (Variable): Input variable for the source sentence
        cg (Graph): Computational graph
        encoder (Encoder): Encoder network 
        align_models (dict): Dictionary of alignment models indexed
                             by source sentence length
    """
    
    def __init__(self, config = {}):
        """Stores the configuration but does not initialize the class
        attributes yet.
        
        Args:
            config (dict): NMT configuration
        """
        self.config = config
    
    def set_up(self, config = None, make_prunable = False):
        """Loads and initializes all the theano variables for the
        training model and the decoding model.
        
        Args:
            config (dict): NMT configuration
        """
        if config:
            self.config = config
        else:
            config = self.config
        # Create Theano variables
        logging.debug('Creating theano variables')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence_mask = tensor.matrix('target_mask')
    
        # Construct model (fs439: Add NoLookup options)
        if config['dec_layers'] != 1:
            logging.fatal("Only dec_layers=1 supported.")
        logging.debug('Building RNN encoder-decoder')
        if config['src_sparse_feat_map']:
            if config['enc_layers'] != 1:
                logging.fatal("Only enc_layers=1 supported for sparse "
                              "source features.")
            source_sentence = tensor.tensor3('source')
            self.sampling_input = tensor.tensor3('input')
            encoder = NoLookupEncoder(config['enc_embed'],
                                      config['enc_nhids'])
        else:
            source_sentence = tensor.lmatrix('source')
            self.sampling_input = tensor.lmatrix('input')
            if config['enc_layers'] > 1 and not config['enc_share_weights']:
                encoder = DeepBidirectionalEncoder(config['src_vocab_size'],
                                                   config['enc_embed'],
                                                   config['enc_layers'], 
                                                   config['enc_skip_connections'],
                                                   config['enc_nhids']) 
            else:
                encoder = BidirectionalEncoder(config['src_vocab_size'],
                                               config['enc_embed'],
                                               config['enc_layers'], 
                                               config['enc_skip_connections'],
                                               config['enc_nhids'])
        if config['trg_sparse_feat_map']:
            target_sentence = tensor.tensor3('target')
            decoder = NoLookupDecoder(config['trg_vocab_size'],
                                      config['dec_embed'], 
                                      config['dec_nhids'],
                                      config['att_nhids'],
                                      config['maxout_nhids'],
                                      config['enc_nhids'] * 2,
                                      config['attention'],
                                      config['dec_attention_sources'],
                                      config['dec_readout_sources'],
                                      config['memory'],
                                      config['memory_size'],
                                      config['seq_len'],
                                      config['dec_init'])
        else:
            target_sentence = tensor.lmatrix('target')
            decoder = Decoder(config['trg_vocab_size'],
                              config['dec_embed'], 
                              config['dec_nhids'],
                              config['att_nhids'],
                              config['maxout_nhids'],
                              config['enc_nhids'] * 2,
                              config['attention'],
                              config['dec_attention_sources'],
                              config['dec_readout_sources'],
                              config['memory'],
                              config['memory_size'],
                              config['seq_len'],
                              config['dec_init'],
                              make_prunable=make_prunable)
        if config['annotations'] != 'direct':
            annotators = []
            add_direct = False
            for name in config['annotations'].split(','):
                if name == 'direct':
                    add_direct = True
                elif name == 'hierarchical':
                    annotators.append(HierarchicalAnnotator(encoder))
                else:
                    logging.fatal("Annotation strategy %s unknown" % name)
            encoder = EncoderWithAnnotators(encoder, annotators, add_direct)
        annotations, annotations_mask = encoder.apply(source_sentence,
                                                      source_sentence_mask) 
        self.cost = decoder.cost(annotations,
                                 annotations_mask,
                                 target_sentence, 
                                 target_sentence_mask)
    
        logging.info('Creating computational graph')
        self.cg = ComputationGraph(self.cost)
    
        # Initialize model
        logging.info('Initializing model')
        encoder.weights_init = decoder.weights_init = IsotropicGaussian(
            config['weight_scale'])
        encoder.biases_init = decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        try:
            encoder.bidir.prototype.weights_init = Orthogonal()
        except AttributeError:
            pass # Its fine, no bidirectional encoder
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()
    
        # apply dropout for regularization
        if config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logging.info('Applying dropout')
            dropout_inputs = [x for x in self.cg.intermediary_variables
                              if x.name == 'maxout_apply_output']
            self.cg = apply_dropout(self.cg, dropout_inputs, config['dropout'])
    
        # Apply weight noise for regularization
        if config['weight_noise_ff'] > 0.0:
            logging.info('Applying weight noise to ff layers')
            if encoder.lookup:
                enc_params = Selector(encoder.lookup).get_parameters().values()
            enc_params += Selector(encoder.fwd_fork).get_parameters().values()
            enc_params += Selector(encoder.back_fork).get_parameters().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_parameters().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_parameters().values()
            self.cg = apply_noise(self.cg, 
                                  enc_params+dec_params, 
                                  config['weight_noise_ff'])
    
        # Print shapes
        shapes = [param.get_value().shape for param in self.cg.parameters]
        logging.debug("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logging.debug('    {:15}: {}'.format(shape, count))
        logging.debug("Total number of CG parameters: {}".format(len(shapes)))
    
        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logging.debug("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logging.debug('    {:15}: {}'.format(value.get_value().shape, name))
        logging.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))
    
        # Set up training model
        logging.info("Building model")
        self.training_model = Model(self.cost)
    
        logging.info("Building sampling model")
        src_shape = (self.sampling_input.shape[-2], 
                     self.sampling_input.shape[-1]) # batch_size x sen_length
        sampling_representation,_ = encoder.apply(self.sampling_input,
                                                  tensor.ones(src_shape))
        generated = decoder.generate(src_shape, sampling_representation)
        self.search_model = Model(generated)
        generated_outputs = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
        self.samples = generated_outputs[1]
        self.encoder = encoder
        self.decoder = decoder
    
