# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains interpolation strategies. This is commonly 
specified via the --interpolation_strategy parameter.
"""

from cam.sgnmt import utils, tf_utils
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
            pred_weights (list): A priori predictor weights
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

    def is_fixed(self):
        return False


class FixedInterpolationStrategy(InterpolationStrategy):
    """Null-object (GoF design pattern) implementation."""

    def find_weights(self, pred_weights, non_zero_words, posteriors, unk_probs):
        """Returns ``pred_weights``."""
        return pred_weights

    def is_fixed(self):
        return True


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
            n_cpu_threads (int): Number of CPU threads for TensorFlow

        Args:
            num_experts (int): Number of predictors under the MoE model
            args (object): SGNMT configuration object
        """
        super(MoEInterpolationStrategy, self).__init__()
        config = dict(el.split("=", 1) for el in args.moe_config.split(";"))
        self._create_hparams(num_experts, config)
        self.model = MOEModel(self.params)
        logging.info("MoE HParams: %s" % self.params)
        moe_graph = tf.Graph()
        with moe_graph.as_default() as g:
          self.model.initialize()
          self.sess = tf_utils.create_session(args.moe_checkpoint_dir,
                                              args.n_cpu_threads)

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

    def _create_score_matrix(self, posteriors, unk_probs):
        scores = np.transpose(np.tile(np.array(unk_probs, dtype=np.float32),
                                      (self.params.vocab_size, 1)))
        # Scores has shape [n_predictors, vocab_size], fill it
        for row, posterior in enumerate(posteriors):
            if isinstance(posterior, dict):
                for  w, s in posterior.items():
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


class EntropyInterpolationStrategy(InterpolationStrategy):
    """The entropy interpolation strategy assigns weights to predictors
    according the entropy of their posteriors to the other posteriors.
    We first build a n x n square matrix of (cross-)entropies between 
    all predictors, and then weight according the row sums. 

    We assume that predictor weights are log probabilities.
    """

    def __init__(self, vocab_size, cross_entropy):
        """Constructor.

        Args:
            vocab_size (int): Vocabulary size
            cross_entropy (bool): If true, use cross entropy to other
                                  predictors. Otherwise, just use
                                  predictor distribution entropy.
        """
        self.vocab_size = vocab_size
        self.cross_entropy = cross_entropy

    def _create_score_matrix(self, posteriors, unk_probs):
        scores = np.transpose(np.tile(np.array(unk_probs),
                                      (self.vocab_size, 1)))
        # Scores has shape [n_predictors, vocab_size], fill it
        for row, posterior in enumerate(posteriors):
            if isinstance(posterior, dict):
                for  w, s in posterior.items():
                    scores[row,int(w)] = s
            else:
                scores[row,:len(posterior)] = np.maximum(-99, posterior)
        return scores

    def find_weights(self, pred_weights, non_zero_words, posteriors, unk_probs):
        logprobs = self._create_score_matrix(posteriors, unk_probs)
        probs = np.exp(logprobs)
        n_preds = len(pred_weights)
        ents = np.zeros((n_preds, n_preds))
        if self.cross_entropy:
            for p_idx in range(n_preds):
                for q_idx in range(n_preds):
                    ents[p_idx, q_idx] = -np.sum(probs[p_idx] * logprobs[q_idx])
        else:
            for p_idx in range(n_preds):
                ents[p_idx, p_idx] = -np.sum(probs[p_idx] * logprobs[p_idx])
        ent_weights = -np.sum(ents, axis=0)
        ent_weights -= np.min(ent_weights)
        ent_weights /= np.sum(ent_weights)
        return np.clip(np.nan_to_num(ent_weights), 0.0, 1.0)
