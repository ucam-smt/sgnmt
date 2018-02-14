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
        self._t2t_unk_id = utils.UNK_ID if t2t_unk_id is None else t2t_unk_id
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
        try:
            checkpoint_path = saver.latest_checkpoint(self._checkpoint_dir)
            return training.MonitoredSession(
                session_creator=training.ChiefSessionCreator(
                    checkpoint_filename_with_path=checkpoint_path,
                    config=self._session_config()))
        except tf.errors.NotFoundError as e:
            logging.fatal("Could not find all variables of the computation "
                "graph in the T2T checkpoint file. This means that the "
                "checkpoint does not correspond to the model specified in "
                "SGNMT. Please double-check pred_src_vocab_size, "
                "pred_trg_vocab_size, and all the t2t_* parameters.")
            raise AttributeError("Could not initialize TF session.")

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
        - Create TF placeholders for source sequence and target prefix
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
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(T2TPredictor, self).__init__(t2t_usr_dir, 
                                           checkpoint_dir, 
                                           t2t_unk_id, 
                                           single_cpu_thread)
        if not model_name or not problem_name or not hparams_set_name:
            logging.fatal(
                "Please specify t2t_model, t2t_problem, and t2t_hparams_set!")
            raise AttributeError
        self.consumed = []
        self.src_sentence = []
        try:
            self.pop_id = int(pop_id) 
        except ValueError:
            logging.warn("t2t predictor only supports single POP IDs. "
                         "Reset to -1")
            self.pop_id = -1
        self.max_terminal_id = max_terminal_id 
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = trainer_utils.create_hparams(hparams_set_name, None)
            if self.pop_id >= 0:
              try:
                hparams.add_hparam("pop_id", self.pop_id)
              except:
                if hparams.pop_id != self.pop_id:
                  logging.warn("T2T pop_id does not match (%d!=%d)"
                    % (hparams.pop_id, self.pop_id))
            self._add_problem_hparams(
                hparams, src_vocab_size, trg_vocab_size, problem_name)
            translate_model = registry.model(model_name)(
                hparams, tf.estimator.ModeKeys.PREDICT)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                               name="sgnmt_targets")
            features = {"inputs": expand_input_dims_for_t2t(self._inputs_var), 
                        "targets": expand_input_dims_for_t2t(self._targets_var)}
            with translate_model._var_store.as_default():
                translate_model.prepare_features_for_infer(features)
                translate_model._fill_problem_hparams_features(features)
                logits, _ = translate_model(features)
                logits = tf.squeeze(logits, [0, 1, 2, 3])
                self._log_probs = log_prob_from_logits(logits)
            self.mon_sess = self.create_session()

    def _add_problem_hparams(
            self, hparams, src_vocab_size, trg_vocab_size, problem_name):
        """Add problem hparams for the problems. 

        This method corresponds to create_hparams() in tensor2tensor's
        trainer_utils module, but replaces the feature encoders with
        DummyFeatureEncoder's.

        Args:
            hparams (Hparams): Model hyper parameters.
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            problem_name (string): T2T problem name.
        
        Returns:
            hparams object.

        Raises:
            LookupError if the problem name is not in the registry or
            uses the old style problem_hparams.
        """
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
        problem = registry.problem(problem_name)
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=trg_vocab_size)
        }
        p_hparams = problem.get_hparams(hparams)
        hparams.problem_instances = [problem]
        hparams.problems = [p_hparams]
        return hparams
                
    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: utils.oov_to_unk(
                 self.consumed + [text_encoder.PAD_ID],
                 self.trg_vocab_size)})
        log_probs[text_encoder.PAD_ID] = utils.NEG_INF
        return log_probs
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.consumed = []
        self.src_sentence = utils.oov_to_unk(
            src_sentence + [text_encoder.EOS_ID], 
            self.src_vocab_size)
   
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


