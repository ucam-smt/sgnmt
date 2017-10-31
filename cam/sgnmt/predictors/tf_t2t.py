"""This is the interface to the tensor2tensor library.

https://github.com/tensorflow/tensor2tensor

Alternatively, you may use the following fork which has been tested in
combination with SGNMT:

https://github.com/fstahlberg/tensor2tensor

The t2t predictor can read any model trained with tensor2tensor which
includes the transformer model, convolutional models, and RNN-based
sequence models.
"""

import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor

POP = "##POP##"
"""Textual representation of the POP symbol."""

try:
    # Requires tensor2tensor
    from tensor2tensor.utils import trainer_utils
    from tensor2tensor.utils import usr_dir
    from tensor2tensor.utils import registry
    from tensor2tensor.utils import devices
    from tensor2tensor.data_generators.text_encoder import TextEncoder
    from tensor2tensor.data_generators import text_encoder
    import tensorflow as tf
    from tensorflow.python.training import saver
    from tensorflow.python.training import training

    class DummyTextEncoder(TextEncoder):
        """Dummy TextEncoder implementation. The TextEncoder 
        implementation in tensor2tensor reads the vocabulary file in
        the constructor, which is not available inside SGNMT. This
        class can be used to replace the standard TextEncoder 
        implementation with a fixed vocabulary size. Note that this
        encoder cannot be used to translate between raw text and
        integer sequences.
        """

        def __init__(self, vocab_size, pop_id=None):
            super(DummyTextEncoder, self).__init__(num_reserved_ids=None)
            self._vocab_size = vocab_size

        def encode(self, s):
            raise NotImplementedError("Dummy encoder cannot be used to encode.")

        def decode(self, ids):
            raise NotImplementedError("Dummy encoder cannot be used to decode.")

        @property
        def vocab_size(self):
            return self._vocab_size

    # Define flags from the t2t binaries
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("schedule", "train_and_evaluate",
                        "Method of tf.contrib.learn.Experiment to run.")
except ImportError:
    pass # Deal with it in decode.py


T2T_INITIALIZED = False
"""Set to true by _initialize_t2t() after first constructor call."""


def _initialize_t2t(t2t_usr_dir):
    global T2T_INITIALIZED
    if not T2T_INITIALIZED:
        logging.info("Setting up tensor2tensor library...")
        tf.logging.set_verbosity(tf.logging.INFO)
        usr_dir.import_usr_dir(t2t_usr_dir)
        trainer_utils.log_registry()
        T2T_INITIALIZED = True


def log_prob_from_logits(logits):
    """Softmax function."""
    return logits - tf.reduce_logsumexp(logits, keep_dims=True)


class _BaseTensor2TensorPredictor(Predictor):
    """Base class for tensor2tensor based predictors."""

    def __init__(self, t2t_usr_dir, checkpoint_dir, t2t_unk_id, single_cpu_thread):
        """Common initialization for tensor2tensor predictors.

        Args:
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.

        Raises:
            IOError if checkpoint file not found.
        """
        super(_BaseTensor2TensorPredictor, self).__init__()
        if not os.path.isfile("%s/checkpoint" % checkpoint_dir):
            logging.fatal("T2T checkpoint file %s/checkpoint not found!" 
                          % checkpoint_dir)
            raise IOError
        self._single_cpu_thread = single_cpu_thread
        self._t2t_unk_id = t2t_unk_id
        self._checkpoint_dir = checkpoint_dir
        _initialize_t2t(t2t_usr_dir)

    def _session_config(self):
        """Creates the session config with t2t default parameters."""
        graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
        if self._single_cpu_thread:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                graph_options=graph_options,
                log_device_placement=False)
        else:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=0.95)
            config = tf.ConfigProto(
                allow_soft_placement=True,
                graph_options=graph_options,
                gpu_options=gpu_options,
                log_device_placement=False)
        return config

    def create_session(self):
        """Creates a MonitoredSession for this predictor."""
        checkpoint_path = saver.latest_checkpoint(self._checkpoint_dir)
        return training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                config=self._session_config()))

    def get_unk_probability(self, posterior):
        """Fetch posterior[t2t_unk_id] or return NEG_INF if None."""
        if self._t2t_unk_id is None:
            return utils.NEG_INF
        return posterior[self._t2t_unk_id]


def expand_input_dims_for_t2t(t):
    """Expands a plain input tensor for using it in a T2T graph.

    Args:
        t: Tensor

    Returns:
      Tensor `t` expanded by 1 dimension on the left and two dimensions
      on the right.
    """
    t = tf.expand_dims(t, 0) # Because of batch_size
    t = tf.expand_dims(t, -1) # Because of modality
    t = tf.expand_dims(t, -1) # Because of random reason X
    return t


class T2TPredictor(_BaseTensor2TensorPredictor):
    """This predictor implements scoring with Tensor2Tensor models. We
    follow the decoder implementation in T2T and do not reuse network
    states in decoding. We rather compute the full forward pass along
    the current history. Therefore, the decoder state is simply the
    the full history of consumed words.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 single_cpu_thread=False,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Creates a new T2T predictor. The constructor prepares the
        TensorFlow session for predict_next() calls. This includes:
        - Load hyper parameters from the given set (hparams)
        - Update registry, load T2T model
        - Create TF placeholders for source sequence and target pefix
        - Create computation graph for computing log probs.
        - Create a MonitoredSession object, which also handles 
          restoring checkpoints.
        
        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
        """
        super(T2TPredictor, self).__init__(t2t_usr_dir, 
                                           checkpoint_dir, 
                                           t2t_unk_id, 
                                           single_cpu_thread)
        self.consumed = []
        self.src_sentence = []
        self.pop_id = pop_id 
        self.max_terminal_id = max_terminal_id 
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = self._create_hparams(
                src_vocab_size, trg_vocab_size, hparams_set_name, problem_name)
            p_hparams = hparams.problems[0]
            self._inputs_var = tf.placeholder(dtype=tf.int32, 
                                              shape=[None], 
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, 
                                               shape=[None], 
                                               name="sgnmt_targets")
            features = {"problem_choice": tf.constant(0),
                        "input_space_id": tf.constant(p_hparams.input_space_id),
                        "target_space_id": tf.constant(
                            p_hparams.target_space_id),
                        "inputs": expand_input_dims_for_t2t(self._inputs_var),
                        "targets": expand_input_dims_for_t2t(self._targets_var)}
        
            model = registry.model(model_name)(
                hparams,
                tf.estimator.ModeKeys.PREDICT,
                hparams.problems[0],
                0,
                devices.data_parallelism(),
                devices.ps_devices(all_workers=True))
            sharded_logits, _ = model.model_fn(features, 
                                               last_position_only=True)
            self._log_probs = log_prob_from_logits(sharded_logits[0])
            self.mon_sess = self.create_session()

    def _create_hparams(
          self, src_vocab_size, trg_vocab_size, hparams_set_name, problem_name):
        """Creates hparams object.
        
        This method corresponds to create_hparams() in tensor2tensor's
        trainer_utils module, but replaces the feature encoders with
        DummyFeatureEncoder's.
        
        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            hparams_set_name (string): T2T hparams set name.
            problem_name (string): T2T problem name.
        
        Returns:
            hparams object.
        
        Raises:
            LookupError if the problem name is not in the registry or
            uses the old style problem_hparams.
        """
        hparams = registry.hparams(hparams_set_name)()
        problem = registry.problem(problem_name)
        # The following hack is necessary to prevent the problem from creating
        # the default TextEncoders, which would fail due to the lack of a
        # vocabulary file.
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=trg_vocab_size)
        }
        try:
            hparams.add_hparam("max_terminal_id", self.max_terminal_id)
        except:
            if hparams.max_terminal_id != self.max_terminal_id:
                logging.warn("T2T max_terminal_id does not match (%d!=%d)"
                             % (hparams.max_terminal_id, self.max_terminal_id))
        try:
            hparams.add_hparam("closing_bracket_id", self.pop_id)
        except:
            if hparams.closing_bracket_id != self.pop_id:
                logging.warn("T2T closing_bracket_id does not match (%d!=%d)"
                             % (hparams.closing_bracket_id, self.pop_id))
        p_hparams = problem.get_hparams(hparams)
        hparams.problem_instances = [problem]
        hparams.problems = [p_hparams]
        return hparams
                
    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        log_probs_squeezed = log_probs[0, 0, 0, 0, :]
        log_probs_squeezed[text_encoder.PAD_ID] = utils.NEG_INF
        return log_probs_squeezed
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.consumed = []
        self.src_sentence = src_sentence + [text_encoder.EOS_ID]
    
    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)
    
    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1 == state2


class T2TBFSLayerbylayerPredictor(_BaseTensor2TensorPredictor):
    """This predictor can be used to decode with layer-by-layer models.
    The state of the predictor includes the current sequence of target
    roots. We introduce a new end-of-layer symbol </l> which is set to
    0 (t2t PAD) by default. We initially use the </s> score for </l> 
    and set </s> to -inf. When </l> is consumed, we update the 
    predictor state as follows:
      - If only terminals have been consumed since the last </l>, we
        finalize this hypothesis by forcing </s> at the next step.
      - Otherwise, update the current sequence of target roots to the
        sequence consumed since the last </l> and reset the history.
    Decoding with this predictor will generate a sequence of tokens
    which can be split into segments at </l>. Each segment describes
    one layer of the generated tree with increasing depth.
    """

    def __init__(self,
                 root_id,
                 max_terminal_id,
                 terminal_list,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 single_cpu_thread=False,
                 terminal_strategy="skip",
                 max_depth=5,
                 eol="add",
                 pop_id=-1):
        """Creates a new layerbylayer predictor, similar to the
        T2TPredictor constructor.
        
        Args:
            root_id (int): ID of the ROOT token.
            max_terminal_id (int): All IDs larger than this are non-
                                   terminals except for the ones in
                                   `terminal_list`.
            terminal_list (string): Comma separated list of IDs larger
                                    than max_terminal_id which still 
                                    should be treated as terminals.
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            terminal_strategy (string): 'force': Force output to parent if
                                        parent is terminal.
                                        'skip': Like force, but use 0 scores.
                                        Otherwise: Treat terminal parent like
                                        any other token
            max_depth (int): Maximum tree depth.
            eol (string): "pad" or "add". If "pad", we use the global token
                          ID for the end-of-layer (eol) symbol. "add" means
                          adding an entry to the posterior vector, ie. the
                          EOL token ID is equal to trg_vocab_size.
            pop_id (int): If positive, ID of the POP symbol.

        Raises:
            ValueError if root_id is negative.
        """
        super(T2TLayerbylayerPredictor, self).__init__(t2t_usr_dir, 
                                                       checkpoint_dir, 
                                                       t2t_unk_id, 
                                                       single_cpu_thread)
        if root_id < 0:
            logging.fatal("Set layerbylayer_root_id to the correct value!") 
            raise ValueError
        self.max_terminal_id = max_terminal_id
        self.force_terminals = terminal_strategy == "force"
        self.skip_terminals = terminal_strategy == "skip"
        self.max_depth = max_depth
        self.pop_id = pop_id if pop_id >= 0 else None
        if terminal_list:
            self.terminal_list = [int(i) for i in terminal_list.split(",")]
        else:
            self.terminal_list = []
        self.root_id = root_id
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = self._create_hparams(
                src_vocab_size, trg_vocab_size, hparams_set_name, problem_name)
            p_hparams = hparams.problems[0]
            self._inputs_var = tf.placeholder(dtype=tf.int32, 
                                              shape=[None], 
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, 
                                               shape=[None], 
                                               name="sgnmt_targets")
            self._target_roots_var = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name="sgnmt_target_roots")
            features = {"problem_choice": tf.constant(0),
                        "input_space_id": tf.constant(p_hparams.input_space_id),
                        "target_space_id": tf.constant(
                            p_hparams.target_space_id),
                        "inputs": expand_input_dims_for_t2t(self._inputs_var),
                        "targets": expand_input_dims_for_t2t(self._targets_var),
                        "target_roots": expand_input_dims_for_t2t(
                                            self._target_roots_var)}
        
            model = registry.model(model_name)(
                hparams,
                tf.estimator.ModeKeys.PREDICT,
                hparams.problems[0],
                0,
                devices.data_parallelism(),
                devices.ps_devices(all_workers=True))
            sharded_logits, _ = model.model_fn(features, 
                                               last_position_only=True)
            self._log_probs = log_prob_from_logits(sharded_logits[0])
            if eol == "pad":
              self.eol_id = text_encoder.PAD_ID
            elif eol == "add":
              self.eol_id = trg_vocab_size
              self._log_probs = tf.pad(self._log_probs, [[0, 0],
                                                         [0, 0],
                                                         [0, 0],
                                                         [0, 0],
                                                         [0, 1]])
            else:
              logging.fatal("Unkown end-of-layer token ID strategy.")
              raise ValueError
            logging.info("End-of-layer ID: %d" % self.eol_id)
            self.mon_sess = self.create_session()

    def _create_hparams(
          self, src_vocab_size, trg_vocab_size, hparams_set_name, problem_name):
        """Creates hparams object, similar to T2TPredictor._create_hparams."""
        hparams = registry.hparams(hparams_set_name)()
        problem = registry.problem(problem_name)
        # The following hack is necessary to prevent the problem from creating
        # the default TextEncoders, which would fail due to the lack of a
        # vocabulary file.
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=trg_vocab_size, 
                                        pop_id=self.pop_id),
            "target_roots": DummyTextEncoder(vocab_size=trg_vocab_size)
        }
        try:
            hparams.add_hparam("max_terminal_id", self.max_terminal_id)
        except:
            if hparams.max_terminal_id != self.max_terminal_id:
                logging.warn("T2T max_terminal_id does not match (%d!=%d)"
                             % (hparams.max_terminal_id, self.max_terminal_id))
        try:
            hparams.add_hparam("pop_id", self.pop_id)
        except:
            if hparams.pop_id != self.pop_id:
                logging.warn("T2T pop_id does not match (%d!=%d)"
                             % (hparams.pop_id, self.pop_id))
        p_hparams = problem.get_hparams(hparams)
        hparams.problem_instances = [problem]
        hparams.problems = [p_hparams]
        return hparams

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        if not self.pop_id is None:
            num_pop = sum([int(i == self.pop_id) for i in self.consumed])
            cur_target_root = self.target_roots[num_pop]
        # Skip computation if terminal root and skip_terminals
        if self.skip_terminals and not self._is_nonterminal(cur_target_root):
            if self.consumed[-1:] == [cur_target_root]:
                return {self.pop_id: 0.0}
            else:
                return {cur_target_root: 0.0}
        # Run tensorflow to get log probs
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._target_roots_var: self.target_roots,
             self._targets_var: self.consumed + [text_encoder.PAD_ID]})
        log_probs_squeezed = log_probs[0, 0, 0, 0, :]
        # Set EOS or EOL scores depending on self.has_nonterminals
        log_probs_squeezed[text_encoder.PAD_ID] = utils.NEG_INF
        if self.has_nonterminals:
            log_probs_squeezed[self.eol_id] = \
                log_probs_squeezed[text_encoder.EOS_ID]
            log_probs_squeezed[text_encoder.EOS_ID] = utils.NEG_INF
        else:
            log_probs_squeezed[self.eol_id] = utils.NEG_INF
        # Restrict the total number of POPs
        if not self.pop_id is None and num_pop >= len(self.target_roots) - 1:
            log_probs_squeezed[self.pop_id] = utils.NEG_INF
        # Force terminals if max depth is reached
        if self.cur_depth >= self.max_depth:
            log_probs_squeezed = self._set_nts_to_inf(log_probs_squeezed)
        # Force output if force_terminals is enabled
        if self.force_terminals and not self._is_nonterminal(cur_target_root):
            return {cur_target_root: log_probs_squeezed[cur_target_root],
                    self.pop_id: log_probs_squeezed[self.pop_id]}
        return log_probs_squeezed
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset state."""
        self.has_nonterminals = False
        self.consumed = []
        self.cur_depth = 0
        self.target_roots = [self.root_id, text_encoder.EOS_ID]
        self.src_sentence = src_sentence + [text_encoder.EOS_ID]

    def _set_nts_to_inf(self, log_probs):
        """Set scores of all non terminals to -inf."""
        for idx in xrange(self.max_terminal_id + 1, len(log_probs)):
            if not idx in self.terminal_list:
                log_probs[idx] = utils.NEG_INF
        return log_probs

    def _is_nonterminal(self, word):
        if word == text_encoder.EOS_ID:
            return True
        return word > self.max_terminal_id and not word in self.terminal_list
    
    def consume(self, word):
        if word == self.eol_id: # Decode next layer
            self.cur_depth += 1
            if self.cur_depth >= self.max_depth:
                logging.debug("Maximum tree depth reached!")
            if  self.target_roots == self.consumed + [text_encoder.EOS_ID]:
                logging.warn("Repeated tree layers: %s!" % (self.target_roots,))
            self.target_roots = self.consumed + [text_encoder.EOS_ID]
            if self.pop_id is not None:
              self.target_roots = [t for t in self.target_roots 
                                   if t != self.pop_id]
            self.has_nonterminals = False
            self.consumed = []
        else:
            self.consumed.append(word)
            if not self.has_nonterminals:
                self.has_nonterminals = self._is_nonterminal(word)
    
    def get_state(self):
        return self.has_nonterminals, self.consumed, self.target_roots, self.cur_depth
    
    def set_state(self, state):
        self.has_nonterminals, self.consumed, self.target_roots, self.cur_depth = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the target roots and consumed are the same """
        return state1[1] == state2[1] and state1[2] == state2[2]


class T2TDFSLayerbylayerPredictor(_BaseTensor2TensorPredictor):
    """TODO"""

    def __init__(self,
                 root_id,
                 max_terminal_id,
                 terminal_list,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 single_cpu_thread=False,
                 terminal_strategy="skip",
                 max_depth=5,
                 pop_id=-1):
        """TODO"""
        super(T2TDFSLayerbylayerPredictor, self).__init__(t2t_usr_dir, 
                                                       checkpoint_dir, 
                                                       t2t_unk_id, 
                                                       single_cpu_thread)
        if root_id < 0:
            logging.fatal("Set layerbylayer_root_id to the correct value!") 
            raise ValueError
        self.max_terminal_id = max_terminal_id
        self.force_terminals = terminal_strategy == "force"
        self.skip_terminals = terminal_strategy == "skip"
        self.max_depth = max_depth
        self.pop_id = pop_id if pop_id >= 0 else None
        if terminal_list:
            self.terminal_list = [int(i) for i in terminal_list.split(",")]
        else:
            self.terminal_list = []
        self.terminal_list.append(self.pop_id)
        self.root_id = root_id
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = self._create_hparams(
                src_vocab_size, trg_vocab_size, hparams_set_name, problem_name)
            p_hparams = hparams.problems[0]
            self._inputs_var = tf.placeholder(dtype=tf.int32, 
                                              shape=[None], 
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, 
                                               shape=[None], 
                                               name="sgnmt_targets")
            self._target_roots_var = tf.placeholder(dtype=tf.int32, 
                                                    shape=[None], 
                                                    name="sgnmt_target_roots")
            features = {"problem_choice": tf.constant(0),
                        "input_space_id": tf.constant(p_hparams.input_space_id),
                        "target_space_id": tf.constant(
                            p_hparams.target_space_id),
                        "inputs": expand_input_dims_for_t2t(self._inputs_var),
                        "targets": expand_input_dims_for_t2t(self._targets_var),
                        "target_roots": expand_input_dims_for_t2t(
                                            self._target_roots_var)}
        
            model = registry.model(model_name)(
                hparams,
                tf.estimator.ModeKeys.PREDICT,
                hparams.problems[0],
                0,
                devices.data_parallelism(),
                devices.ps_devices(all_workers=True))
            sharded_logits, _ = model.model_fn(features, 
                                               last_position_only=True)
            self._log_probs = log_prob_from_logits(sharded_logits[0])
            self.mon_sess = self.create_session()

    def _create_hparams(
          self, src_vocab_size, trg_vocab_size, hparams_set_name, problem_name):
        """Creates hparams object, similar to T2TPredictor._create_hparams."""
        hparams = registry.hparams(hparams_set_name)()
        problem = registry.problem(problem_name)
        # The following hack is necessary to prevent the problem from creating
        # the default TextEncoders, which would fail due to the lack of a
        # vocabulary file.
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=trg_vocab_size, 
                                        pop_id=self.pop_id),
            "target_roots": DummyTextEncoder(vocab_size=trg_vocab_size)
        }
        try:
            hparams.add_hparam("max_terminal_id", self.max_terminal_id)
        except:
            if hparams.max_terminal_id != self.max_terminal_id:
                logging.warn("T2T max_terminal_id does not match (%d!=%d)"
                             % (hparams.max_terminal_id, self.max_terminal_id))
        try:
            hparams.add_hparam("pop_id", self.pop_id)
        except:
            if hparams.pop_id != self.pop_id:
                logging.warn("T2T pop_id does not match (%d!=%d)"
                             % (hparams.pop_id, self.pop_id))
            pass
        p_hparams = problem.get_hparams(hparams)
        hparams.problem_instances = [problem]
        hparams.problems = [p_hparams]
        return hparams

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        # Force EOS if parent is EOS
        if self.cur_depth == 0:
            return {utils.EOS_ID: self._compute_eos_score()}
        # Run tensorflow to get log probs
        #print("\nPREDICT NEXT START (%d)" % self.cur_depth)
        #print(self._strip_pop(self.layer_targets[self.cur_depth-1]))
        #print(self.layer_targets[self.cur_depth])
        #print("PREDICT NEXT STOP")
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._target_roots_var: self._strip_pop(self.layer_targets[self.cur_depth-1]),
             self._targets_var: self.layer_targets[self.cur_depth] + [text_encoder.PAD_ID]})
        log_probs_squeezed = log_probs[0, 0, 0, 0, :]
        # Go not deeper than max_depth
        if self.cur_depth >= self.max_depth - 1:
            #print("SET NTs to -inf")
            log_probs_squeezed = self._set_nts_to_inf(log_probs_squeezed)
        log_probs_squeezed[text_encoder.PAD_ID] = utils.NEG_INF
        log_probs_squeezed[text_encoder.EOS_ID] = utils.NEG_INF
        return log_probs_squeezed

    def _strip_pop(self, seq):
        return [i for i in seq if i != self.pop_id]
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset state."""
        self.cur_depth = 1
        self.tree_depth = 1
        self.layer_targets = [[self.root_id]]
        self.layer_targets.extend([[] for _ in xrange(self.max_depth)])
        self.src_sentence = src_sentence + [text_encoder.EOS_ID]

    def _compute_eos_score(self):
        eos_score = 0.0
        for depth in xrange(1, self.tree_depth + 1):
            log_probs = self.mon_sess.run(self._log_probs,
                {self._inputs_var: self.src_sentence,
                 self._target_roots_var: self._strip_pop(self.layer_targets[depth-1]) + [text_encoder.EOS_ID],
                 self._targets_var: self.layer_targets[depth] + [text_encoder.PAD_ID]})
            eos_score += log_probs[0, 0, 0, 0, text_encoder.EOS_ID]
        return eos_score

    def _set_nts_to_inf(self, log_probs):
        """Set scores of all non terminals to -inf."""
        for idx in xrange(self.max_terminal_id + 1, len(log_probs)):
            if not idx in self.terminal_list:
                log_probs[idx] = utils.NEG_INF
        return log_probs

    def _is_nonterminal(self, word):
        return word > self.max_terminal_id and not word in self.terminal_list
    
    def consume(self, word):
        self.layer_targets[self.cur_depth].append(word)
        if word == self.pop_id:
            self.cur_depth -= 1
        elif self._is_nonterminal(word):
            self.cur_depth += 1
            self.tree_depth = max(self.tree_depth, self.cur_depth)
        else: # When producing a terminal, propagate to all next layers
            for i in xrange(self.cur_depth + 1, len(self.layer_targets)):
                self.layer_targets[i].extend([word, self.pop_id])
        #print("consumed %d new depth %d" % (word, self.cur_depth))
        #for i in xrange(min(6, len(self.layer_targets))):
        #    print(self.layer_targets[i])
    
    def get_state(self):
        return self.cur_depth, self.tree_depth, self.layer_targets
    
    def set_state(self, state):
        self.cur_depth, self.tree_depth, self.layer_targets = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """TODO"""
        return state1 == state2 
