"""This module integrates Nizza alignment models.

https://github.com/fstahlberg/nizza
"""

import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor

try:
    import tensorflow as tf
    from tensorflow.python.training import saver
    from tensorflow.python.training import training
    # Requires nizza
    from nizza import registry
    from nizza.utils import common_utils
except ImportError:
    pass # Deal with it in decode.py


class NizzaPredictor(Predictor):
    """This predictor uses Nizza alignment models to derive a posterior over
    the target vocabulary for the next position. It mainly relies on the
    predict_next_word() implementation of Nizza models.
    """

    def __init__(self, src_vocab_size, trg_vocab_size, model_name, 
                 hparams_set_name, checkpoint_dir, single_cpu_thread,
                 nizza_unk_id=None):
        """Initializes a nizza predictor.

        Args:
            src_vocab_size (int): Source vocabulary size (called inputs_vocab_size
                in nizza)
            trg_vocab_size (int): Target vocabulary size (called targets_vocab_size
                in nizza)
            model_name (string): Name of the nizza model
            hparams_set_name (string): Name of the nizza hyper-parameter set
            checkpoint_dir (string): Path to the Nizza checkpoint directory. The 
                                     predictor will load the top most checkpoint in 
                                     the `checkpoints` file.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            nizza_unk_id (int): If set, use this as UNK id. Otherwise, the
                nizza is assumed to have no UNKs

        Raises:
            IOError if checkpoint file not found.
        """
        super(NizzaPredictor, self).__init__()
        if not os.path.isfile("%s/checkpoint" % checkpoint_dir):
            logging.fatal("Checkpoint file %s/checkpoint not found!" 
                          % checkpoint_dir)
            raise IOError
        self._single_cpu_thread = single_cpu_thread
        self._checkpoint_dir = checkpoint_dir
        self._nizza_unk_id = nizza_unk_id
        self.consumed = []
        self.src_sentence = []
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = registry.get_registered_hparams_set(hparams_set_name)
            hparams.add_hparam("inputs_vocab_size", src_vocab_size)
            hparams.add_hparam("targets_vocab_size", trg_vocab_size)
            run_config = tf.contrib.learn.RunConfig()
            run_config = run_config.replace(model_dir=checkpoint_dir)
            model = registry.get_registered_model(model_name, hparams, run_config)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                               name="sgnmt_targets")
            features = {"inputs": tf.expand_dims(self._inputs_var, 0), 
                        "targets": tf.expand_dims(self._targets_var, 0)}
            mode = tf.estimator.ModeKeys.PREDICT
            precomputed = model.precompute(features, mode, hparams)
            self.log_probs = tf.squeeze(
                model.predict_next_word(features, hparams, precomputed), 0)
            self.mon_sess = self.create_session()

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
        if self._nizza_unk_id is None:
            return utils.NEG_INF
        return posterior[self._nizza_unk_id]

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self.log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [common_utils.PAD_ID]})
        log_probs[common_utils.PAD_ID] = utils.NEG_INF  # Mask padding
        return log_probs
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.consumed = []
        self.src_sentence = src_sentence + [utils.EOS_ID]
    
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

