"""This module contains interpolation strategies. This is commonly 
specified via the --interpolation_strategy parameter.
"""

from cam.sgnmt import utils
import numpy as np
import logging
from abc import abstractmethod

try:
    # This is the TF backend needed for MoE interpolation
    import tensorflow as tf
    from tensorflow.python.training import saver
    from tensorflow.python.training import training
    from tensorflow.contrib.training.python.training import hparam
    # Requires sgnmt_moe
    from sgnmt_moe.model import MOEModel
except ImportError:
    pass # Deal with it in decode.py


class InterpolationStrategy(object):
    """Base class for interpolation strategies."""

    @abstractmethod
    def find_weights(self, pred_weights, non_zero_words, posteriors, unk_probs):
        """Find interpolation weights for the current prediction.

        Args:
            pred_weights (list): A prior predictor weights
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``

        Returns:
            list of floats. The predictor weights for this prediction.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        raise NotImplementedError


class FixedInterpolationStrategy(InterpolationStrategy):
    """Null-object (GoF design pattern) implementation."""

    def find_weights(self, pred_weights, non_zero_words, posteriors, unk_probs):
        """Returns ``pred_weights``."""
        return pred_weights


class MoEInterpolationStrategy(InterpolationStrategy):
    """This class implements a predictor-level Mixture of Experts (MoE)
    model. In this scenario, we have a neural model which predicts 
    predictor weights from the predictor outputs. See the sgnmt_moe 
    project on how to train this gating network with TensorFlow.
    """

    def __init__(self, num_experts, args):
        """Creates the computation graph of the MoE network and loads
        the checkpoint file. Following fields are fetched from ``args``

            moe_config: Comma-separated <key>=<value> pairs specifying
              the MoE network. See the command line arguments of 
              sgnmt_moe for a full description. Available keys:
              vocab_size, embed_size, activation, hidden_layer_size,
              preprocessing.
            moe_checkpoint_dir (string): Checkpoint directory

        Args:
            num_experts (int): Number of predictors under the MoE model
            args (object): SGNMT configuration object
        """
        super(MoEInterpolationStrategy, self).__init__()
        config = dict(el.split("=", 1) for el in args.moe_config.split(";"))
        self._single_cpu_thread = args.single_cpu_thread
        self._checkpoint_dir = args.moe_checkpoint_dir
        self._create_hparams(num_experts, config)
        self.model = MOEModel(self.params)
        logging.info("MoE HParams: %s" % self.params)
        moe_graph = tf.Graph()
        with moe_graph.as_default() as g:
          self.model.initialize()
          self.sess = self._create_session()

    def _create_hparams(self, num_experts, config):
        """Creates self.params."""
        self.params = hparam.HParams(
          vocab_size=int(config.get("vocab_size", "30003")),
          learning_rate=0.001, # Not used
          batch_size=1,
          num_experts=num_experts,
          embed_filename="",
          embed_size=int(config.get("embed_size", "512")),
          activation=config.get("activation", "relu"),
          loss_strategy="rank", # Not used
          hidden_layer_size=int(config.get("hidden_layer_size", "64")),
          preprocessing=config.get("preprocessing", "")
        )

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

    def _create_session(self):
        """Creates a MonitoredSession for this predictor."""
        try:
            checkpoint_path = saver.latest_checkpoint(self._checkpoint_dir)
            return training.MonitoredSession(
                session_creator=training.ChiefSessionCreator(
                    checkpoint_filename_with_path=checkpoint_path,
                    config=self._session_config()))
        except tf.errors.NotFoundError as e:
            logging.fatal("Could not find all variables of the computation "
                "graph in the MoE checkpoint file. This means that the "
                "checkpoint does not correspond to the model specification.")
            raise AttributeError("Could not initialize TF session for MoE.")

    def _create_score_matrix(self, posteriors, unk_probs):
      scores = np.transpose(np.tile(np.array(unk_probs, dtype=np.float32),
                                    (self.params.vocab_size, 1)))
      # Scores has shape [n_predictors, vocab_size], fill it
      for row, posterior in enumerate(posteriors):
          if isinstance(posterior, dict):
              for  w, s in posterior.iteritems():
                  scores[row,int(w)] = s
          else:
              scores[row,:len(posterior)] = np.maximum(-99, posterior)
      return np.expand_dims(scores, axis=0)

    def find_weights(self, pred_weights, non_zero_words, posteriors, unk_probs):
        """Runs the MoE model to find interpolation weights.

        Args:
            pred_weights (list): A prior predictor weights
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``

        Returns:
            list of floats. The predictor weights for this prediction.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        scores = self._create_score_matrix(posteriors, unk_probs)
        weights = self.sess.run(self.model.weights, 
                                feed_dict={self.model.expert_scores: scores})
        return weights[0,:]

