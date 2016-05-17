"""This is the only module outside the ``blocks`` package with
dependency on the Blocks framework. It contains the neural machine
translation predictor nmt. Code is partially taken from the neural
machine translation example in blocks.

https://github.com/mila-udem/blocks-examples/tree/master/machine_translation

Note that using this predictor slows down decoding compared to the
original NMT decoding because search cannot be parallelized. However,
it is much more flexible as it can be combined with other predictors.
"""

from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.model import Model
from blocks.search import BeamSearch
from blocks.select import Selector
from collections import Counter
import copy
import logging
import os
import re
from theano import tensor
from toolz import merge

from cam.sgnmt import utils
from cam.sgnmt.blocks.machine_translation.checkpoint import SaveLoadUtils
from cam.sgnmt.blocks.machine_translation.model import BidirectionalEncoder, \
                                                      Decoder
from cam.sgnmt.decoding.core import Predictor
import numpy as np


NEG_INF = float("-inf")


"""Name of the default model file (not checkpoints) """
PARAMS_FILE_NAME = 'params.npz'


"""Pattern for checkpoints created in training for model selection """
BEST_BLEU_PATTERN = re.compile('^best_bleu_params_([0-9]+)_BLEU([.0-9]+).npz$')


def get_nmt_model_path_params(nmt_config):
    """Returns the path to the params.npz. This file usually contains 
    the latest model parameters.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the params.npz
    """
    return '%s/%s' % (nmt_config['saveto'], PARAMS_FILE_NAME)


def get_nmt_model_path_best_bleu(nmt_config):
    """Returns the path to the checkpoint with the best BLEU score. If
    no checkpoint can be found, back up to params.npz.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the checkpoint file with best BLEU score
    """
    best = 0.0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and float(m.group(2)) > best:
            best = float(m.group(2))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


def get_nmt_model_path_most_recent(nmt_config):
    """Returns the path to the most recent checkpoint. If
    no checkpoint can be found, back up to params.npz.
    
    Args:
        nmt_config (dict):  NMT configuration. We will use the field
                            ``saveto`` to get the training directory
    
    Returns:
        string. Path to the most recent checkpoint file
    """
    best = 0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and int(m.group(1)) > best:
            best = int(m.group(1))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


class MyopticSearch(BeamSearch):
    """This class hacks into blocks beam search to leverage off the 
    initialization routines. Note that this has nothing to do with 
    SGNMTs high level decoding in ``cam.sgnmt.decoding``. We basically
    replace the ``search()`` with single_step_``decoding()`` which 
    generates the posteriors for the next word. Thus, it fits in the 
    predictor framework. We try to use ``BeamSearch`` functionality
    wherever possible.
    """
    def __init__(self, samples):
        """Calls the ``BeamSearch`` constructor """
        super(MyopticSearch, self).__init__(samples)


class LoadNMT(SaveLoadUtils):
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
        super(LoadNMT, self).__init__(**kwargs)

    def load_weights(self):
        """Load the model parameters from the model file. Compare with
        ``blocks.machine_translation.LoadNMT``.
        """
        if not os.path.exists(self.path_to_folder):
            logging.info("No dump found")
            return
        logging.info("Loading the model from {}".format(self.nmt_model_path))
        try:
            logging.info(" ...loading model parameters")
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
                " Number of parameters loaded for computation graph: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logging.error(" Error {0}".format(str(e)))

    def load_parameters(self):
        """Currently not used, kept for consistency with blocks
        reference implementation. """
        return self.load_parameter_values(self.nmt_model_path)


class NMTPredictor(Predictor):
    """This is the neural machine translation predictor. The predicted
    posteriors are equal to the distribution generated by the decoder
    network in NMT. This predictor heavily relies on the NMT example in
    blocks.
    """
    
    def __init__(self, nmt_model_path, enable_cache, config):
        """Creates a new NMT predictor.
        
        Args:
            nmt_model_path (string):  Path to the NMT model file (.npz)
            enable_cache (bool):  The NMT predictor usually has a very
                                  limited vocabulary size, and a large
                                  number of UNKs in hypotheses. This
                                  enables reusing already computed
                                  predictor states for hypotheses which
                                  differ only by NMT OOV words.
            config (dict): NMT configuration, see 
                          ``blocks.machine_translation.configurations``
        """
        super(NMTPredictor, self).__init__()
        self.config = copy.deepcopy(config)
        self.enable_cache = enable_cache
        self.set_up_predictor(nmt_model_path)
    
    def set_up_predictor(self, nmt_model_path):
        """Initializes the predictor with the given NMT model. Code 
        following ``blocks.machine_translation.main``. 
        """
        self.src_vocab_size = self.config['src_vocab_size']
        self.trgt_vocab_size = self.config['trg_vocab_size']
        
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
        cost = decoder.cost(
                encoder.apply(source_sentence, source_sentence_mask),
                source_sentence_mask, target_sentence, target_sentence_mask)
    
        logging.info('Creating computational graph')
        cg = ComputationGraph(cost)
    
        # Initialize model (TODO: really necessary?)
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
    
        # Apply dropout for regularization (TODO: remove?)
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
        logging.debug("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logging.debug('    {:15}: {}'.format(shape, count))
        logging.info("Total number of parameters: {}".format(len(shapes)))
    
        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logging.debug("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logging.debug('    {:15}: {}'.format(value.get_value().shape,
                                                 name))
        logging.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))
    
        # Set up training model
        logging.info("Building model")
    
        # Set extensions
        logging.info("Initializing extensions")
    
        # Set up beam search and sampling computation graphs if necessary
        logging.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        search_model = Model(generated)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
            
        # Follows blocks.machine_translation.BleuValidator.__init__
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
        self.search_algorithm = MyopticSearch(samples=samples)
        self.search_algorithm.compile()

    def initialize(self, src_sentence):
        """Runs the encoder network to create the source annotations
        for the source sentence. If the cache is enabled, empty the
        cache.
        
        Args:
            src_sentence (list): List of word ids without <S> and </S>
                                 which represent the source sentence.
        """
        self.reset()
        self.posterior_cache = utils.SimpleTrie()
        self.states_cache = utils.SimpleTrie()
        self.consumed = []
        seq = [w if w < self.src_vocab_size else utils.UNK_ID 
                    for w in src_sentence] + [utils.EOS_ID]
        input_ = np.tile(seq, (1, 1))
        input_values={self.source_sentence: input_}
        self.contexts, self.states, _ = self.search_algorithm.compute_initial_states_and_contexts(
            input_values)
    
    def is_history_cachable(self):
        """Returns true if cache is enabled and history contains UNK """
        if not self.enable_cache:
            return False
        for w in self.consumed:
            if w == utils.UNK_ID:
                return True
        return False

    def predict_next(self):
        """Uses cache or runs the decoder network to get the 
        distribution over the next target words.
        
        Returns:
            np array. Full distribution over the entire NMT vocabulary
            for the next target token.
        """
        use_cache = self.is_history_cachable()
        if use_cache:
            posterior = self.posterior_cache.get(self.consumed)
            if not posterior is None:
                logging.debug("Loaded NMT posterior from cache for %s" % 
                                self.consumed)
                return posterior
        # logprobs are negative log probs, i.e. greater than 0
        logprobs = self.search_algorithm.compute_logprobs(self.contexts,
                                                          self.states)
        posterior = np.multiply(logprobs[0], -1.0)
        if use_cache:
            self.posterior_cache.add(self.consumed, posterior)
        return posterior
        
    def get_unk_probability(self, posterior):
        """Returns the UNK probability defined by NMT. """
        return posterior[utils.UNK_ID] if len(posterior) > utils.UNK_ID else NEG_INF
    
    def consume(self, word):
        """Feeds back ``word`` to the decoder network. This includes 
        embedding of ``word``, running the attention network and update
        the recurrent decoder layer.
        """
        if word >= self.trgt_vocab_size:
            word = utils.UNK_ID
        self.consumed.append(word)
        use_cache = self.is_history_cachable()
        if use_cache:
            s = self.states_cache.get(self.consumed)
            if not s is None:
                logging.debug("Loaded NMT decoder states from cache for %s" % 
                                    self.consumed)
                self.states = copy.deepcopy(s)
                return
        self.states.update(self.search_algorithm.compute_next_states(
                self.contexts, self.states, [word]))
        if use_cache:
            self.states_cache.add(self.consumed, copy.deepcopy(self.states))
    
    def get_state(self):
        """The NMT predictor state consists of the decoder network 
        state, and (for caching) the current history of consumed words
        """
        return self.states,self.consumed
    
    def set_state(self, state):
        """Set the NMT predictor state. """
        self.states,self.consumed = state

    def reset(self):
        """Deletes the source side annotations and decoder state. """
        self.contexts = None
        self.states = None 
        
