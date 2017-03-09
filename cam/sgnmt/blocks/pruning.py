"""This module contains  code for model pruning during training
"""
import logging

from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph
import theano


logger = logging.getLogger(__name__)


class PruningGradientDescent(GradientDescent):

    def __init__(self, prune_every, nmt_model, **kwargs):
        self.prune_every = prune_every
        self.nmt_model = nmt_model
        super(PruningGradientDescent, self).__init__(**kwargs)
    
    def initialize(self):
        # From UpdatesAlgorithm
        logger.info("Initializing the training algorithm")
        update_values = [new_value for _, new_value in self.updates]
        logger.debug("Inferring graph inputs...")
        self.inputs = ComputationGraph(update_values).inputs
        logger.debug("Compiling training function...")
        # Catch the activities
        self.activity_variables = []
        self.activity_variables.append(self.nmt_model.encoder.bidir.children[0])
        self._function = theano.function(
            self.inputs, [], updates=self.updates, **self.theano_func_kwargs)
        logger.info("The training algorithm is initialized")