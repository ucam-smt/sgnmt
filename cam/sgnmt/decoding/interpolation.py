"""This module contains interpolation strategies. This is commonly 
specified via the --interpolation_strategy parameter.
"""

from cam.sgnmt import utils
import numpy as np
import logging
from abc import abstractmethod

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

            moe_preprocessing:

        Args:
            num_experts (int): Number of predictors under the MoE model
            args (object): SGNMT configuration object
        """
        super(MoEInterpolationStrategy, self).__init__()

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
        # TODO implement
        return pred_weights

