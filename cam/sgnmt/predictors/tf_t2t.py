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

        def __init__(self, vocab_size):
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


class T2TPredictor(_BaseTensor2TensorPredictor):
    """This predictor implements scoring with Tensor2Tensor models. We
    follow the decoder implementation in T2T and do not reuse network
    states in decoding. We rather compute the full forward pass along
    the current history. Therefore, the decoder state is simply the
    the full history of consumed words.
    """

    
    def __init__(self,
                 t2t_usr_dir,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 single_cpu_thread=False):
        """Creates a new T2T predictor. The constructor prepares the
        TensorFlow session for predict_next() calls. This includes:
        - Load hyper parameters from the given set (hparams)
        - Update registry, load T2T model
        - Create TF placeholders for source sequence and target pefix
        - Create computation graph for computing log probs.
        - Create a MonitoredSession object, which also handles 
          restoring checkpoints.
        
        Args:
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
        """
        super(T2TPredictor, self).__init__(t2t_usr_dir, 
                                           checkpoint_dir, 
                                           t2t_unk_id, 
                                           single_cpu_thread)
        self.consumed = []
        self.src_sentence = []
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
            def expand_input_dims_for_t2t(t):
                t = tf.expand_dims(t, 0) # Because of batch_size
                t = tf.expand_dims(t, -1) # Because of modality
                t = tf.expand_dims(t, -1) # Because of random reason X
                return t
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
