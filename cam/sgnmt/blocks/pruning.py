"""This module contains code for model pruning during training. This
implements the data-bound neuron removal algorithm descibed in

Unfolding and Shrinking Neural Machine Translation Ensembles
Felix Stahlberg and Bill Byrne
https://arxiv.org/abs/1704.03279

Note that to avoid rebuilding the computation graph after each
prunning operation, we do not remove neurons but set their connections
to zero. To realize speed ups, neurons with zero weights must be
removed in a postprocessing step.
"""

import logging

from blocks.algorithms import GradientDescent
from blocks.graph import ComputationGraph
from blocks.bricks.sequence_generators import SequenceGenerator
from blocks.bricks.base import application
from blocks.utils import dict_union, dict_subset
from blocks.bricks import FeedforwardSequence, Initializable
from blocks.utils import pack
from theano import tensor
import theano
import numpy as np

logger = logging.getLogger(__name__)

INF_DIST = 10000.0

class PrunableLayer(object):
    """This class represents a layer definition loaded from the file
    system and keeps track of neurons which have been removed in the
    layer.
    """

    def __init__(self, 
                 name, 
                 theano_variable, 
                 trg_size, 
                 n_steps, 
                 maxout=False):
        """Creates a new ``PrunableLayer`` instance.
        
        Args:
            name (string): Name of the layer
            theano_variable (TheanoVariable): Variable in the 
                                              computation graph which
                                              stores the neuron
                                              activities in the layer
            trg_size (int): Remove neurons in this layer until this
                            layer size is reached
            n_steps (int): Number of pruning operations
            maxout (bool): Whether this layer is a maxout layer.
        """
        self.name = name
        self.theano_variable = theano_variable
        self.dists = None
        self.activities = None
        self.mask = None
        self.n_obs = 0.0
        self.trg_size = trg_size
        self.n_steps = n_steps
        self.connections = []
        self.pruned_neurons = []
        self.obs = []
        self.maxout = maxout

    def reset(self):
        """Resets the activity records. Can be called after a pruning
        operation to maintain recency.
        """ 
        self.dists = None
        self.activities = None
        self.n_obs = 0.0
        self.obs = []
        logging.info("Layer %s reset" % self.name)

    def initialize_mask(self):
        """Initialize the neuron mask to a upper triangular matrix, ie.
        no neuron has been removed.
        """
        self.mask = np.triu(INF_DIST * np.ones_like(self.dists))

    def derive_step_size(self):
        """Calculate the required step size to reach the target size
        for this layer.
        """
        self.step_size = int((self.get_size() - self.trg_size) / self.n_steps)
        self.step_size += 1

    def get_size(self):
        """Get the size of the layer. """
        return self.mask.shape[0]

    def register_activities(self, activity):
        """Store the layer activities in a training batch and update 
        the distance matrix.
        
        Args:
            activity (array): Neuron activity in the most recent batch
        """
        # TODO: Use Decoder training stream mask!
        x = activity.reshape((-1, activity.shape[-1])).transpose()
        self.obs.append(x)
        # http://stackoverflow.com/questions/15556116/implementing-support-
        # vector-machine-efficiently-computing-gram-matrix-k
        pt_sq_norms = (x ** 2).sum(axis=1)
        dists_sq = np.dot(x, x.T)
        dists_sq *= -2
        dists_sq += pt_sq_norms.reshape(-1, 1)
        dists_sq += pt_sq_norms
        activities_sq = np.sum(np.square(x), axis=1)
        if self.dists is None:
            self.dists = dists_sq
            self.activities = activities_sq
        else:
            self.dists += dists_sq
            self.activities += activities_sq
        self.n_obs += x.shape[1]

    def count_unpruned_neurons(self):
        """Get the number of unpruned neurons in the layer. """
        return self.get_size() - len(self.pruned_neurons)

    def prune(self, params_dict):
        """Perform a single pruning operation. The number of neurons to
        delete depends on the step size and the difference between the
        current and the target layer size.
        
        Args:
            params_dict (dict): Dictionary of numpy arrays
        """
        if self.mask is None:
            self.initialize_mask()
            self.derive_step_size()
        n_to_delete = min(self.count_unpruned_neurons() - self.trg_size,
                          self.step_size)
        if n_to_delete <= 0:
            logging.info("Layer %s already pruned enough" % self.name)
            self.reset()
            return
        self.activities /= self.n_obs
        self.dists /= self.n_obs
        activity_discounts = get_activity_discounts(self)
        search_mat = np.multiply(self.dists, activity_discounts) + self.mask
        idxs_flat = np.argsort(search_mat, None)
        idxs = np.unravel_index(idxs_flat, search_mat.shape)
        min_score = search_mat[idxs[0][0], idxs[1][0]]
        to_delete = []
        for i,j in zip(idxs[0], idxs[1]):
            if i in self.pruned_neurons or j in self.pruned_neurons:
                continue
            max_score = search_mat[i,j]
            if self.activities[i] < self.activities[j]:
                i,j = j,i
            to_delete.append((i, j))
            self.pruned_neurons.append(j)
            self.mask[j,:] = INF_DIST
            self.mask[:,j] = INF_DIST
            if len(to_delete) >= n_to_delete:
                break
        compensate_for_pruning(to_delete, self, params_dict)
        logging.info("%s: Prune %d neurons obs=%d min=%f max=%f" % (
                                                         self.name,
                                                         len(to_delete),
                                                         self.n_obs,
                                                         min_score,
                                                         max_score))

    def sanity_check(self, params_dict):
        """Checks whether the weight matrices are consistent with the
        list of neurons which have been removed so far. All incoming
        and outgoing connections of prunned neurons should be zero.
        
        Args:
            params_dict (dict): Dictionary with numpy arrays
        """
        geps = 0.0
        for conn in self.connections:
            mat = params_dict[conn.mat_name].get_value()
            mat_idxs = self.pruned_neurons
            if conn.start_idx > 0.0:
                offset = int(mat.shape[conn.dim] * conn.start_idx)
                mat_idxs = [idx+offset for idx in mat_idxs]
            if len(mat.shape) == 1:
                eps = np.max(np.absolute(mat[mat_idxs]))
            elif conn.dim == 0:
                eps = np.max(np.absolute(mat[mat_idxs,:]))
            elif conn.dim == 1:
                eps = np.max(np.absolute(mat[:,mat_idxs]))
            geps = max(eps, geps)
        logging.info("Sanity check: max of %d prunned connections: %f" % (
                                                    len(self.pruned_neurons), 
                                                    geps))
            
def _get_activity_discounts_sum(layer):
    """For a pair of neurons, use the sum of their activities to 
    discount their distance.
    
    Args:
        layer (PrunableLayer): The layer which is to be pruned
    
    Returns:
        array. Discount matrix for the layer.
    """ 
    return layer.activities.reshape((-1, 1)) + layer.activities.reshape((1, -1))

def _get_activity_discounts_min(layer):
    """For a pair of neurons, use the minimum of their activities to 
    discount their distance.
    
    Args:
        layer (PrunableLayer): The layer which is to be pruned
    
    Returns:
        array. Discount matrix for the layer.
    """ 
    return np.minimum(layer.activities.reshape((-1, 1)),
                      layer.activities.reshape((1, -1)))

get_activity_discounts = _get_activity_discounts_min
"""We decide which neuron to prune based on (a) their similarity, and
(b) we prefer neurons with small activity. We provide two strategies
for (b): Using the minimum or the sum of the activities in neuron
pairs. The original method uses the minimum.  
    
Args:
    layer (PrunableLayer): The layer which is to be pruned
    
Returns:
    array. Discount matrix for the layer.
""" 

def _compensate_for_pruning_sum(to_delete, layer, params_dict):
    """This is a data-bound version of the approach of Srinivas and
    Babu (2015) which compensates for a neuron removal by adding
    the outgoing weights to the weights of a similar neuron. The 
    parameter dictionary ``params_dict`` is updated accordingly.
    
    Args:
        to_delete (list): List of i,j neuron pairs which were selected
                          for deletion
        layer (PrunableLayer): The layer which we are currently pruning
        params_dict (dict): Dictionary of numpy arrays (weight matrices)
    """
    for i,j in to_delete:
        # Prune neuron j, add output connections to i
        for conn in layer.connections:
            mat = params_dict[conn.mat_name].get_value()
            mat_i = i
            mat_j = j
            if conn.start_idx > 0.0:
                offset = int(mat.shape[conn.dim] * conn.start_idx)
                mat_i += offset
                mat_j += offset
            if conn.direction == "in" and layer.maxout:
                mat = set_zero_in_mat(mat, conn.dim, mat_j*2)
                mat = set_zero_in_mat(mat, conn.dim, mat_j*2+1)
            else:
                mat = set_zero_in_mat(mat, conn.dim, mat_j)
            params_dict[conn.mat_name].set_value(mat)


def _compensate_for_pruning_interpol(to_delete, layer, params_dict):
    """This is the data-bound removal algorithm from Stahlberg and
    Byrne (2017) based on linear interpolations. The parameter
    dictionary ``params_dict`` is updated accordingly.
    
    Args:
        to_delete (list): List of i,j neuron pairs which were selected
                          for deletion
        layer (PrunableLayer): The layer which we are currently pruning
        params_dict (dict): Dictionary of numpy arrays (weight matrices)
    """
    reduced_obs = np.hstack(layer.obs)[:,np.random.randint(
                                                    layer.n_obs, 
                                                    size=50000)].transpose()
    delete_idxs = [j for i,j in to_delete]
    survive_mask = np.ones((layer.get_size(),), dtype=bool)
    survive_mask[layer.pruned_neurons] = False
    survive_idxs = np.where(survive_mask)[0]
    A = reduced_obs[:,survive_idxs]
    y = reduced_obs[:,delete_idxs]
    logging.info("Least square A=%s y=%s" % (A.shape, y.shape))
    weights = np.linalg.lstsq(A, y)[0]
    for conn in layer.connections:
        mat = params_dict[conn.mat_name].get_value()
        work = mat
        if conn.dim == 1:
            work = work.transpose()
        if not layer.maxout:
            offset = int(work.shape[0] * conn.start_idx)
            if len(work.shape) == 2:
                work = work[offset:offset+layer.get_size(), :]
            else:
                work = work[offset:offset+layer.get_size()]
        if conn.direction == "out": 
            work[survive_idxs,:] += np.dot(weights, work[delete_idxs])
        for j in delete_idxs:
            if conn.direction == "in" and layer.maxout:
                work = set_zero_in_mat(work, 0, j*2)
                work = set_zero_in_mat(work, 0, j*2+1)
            else:
                work = set_zero_in_mat(work, 0, j)
        params_dict[conn.mat_name].set_value(mat)


compensate_for_pruning = _compensate_for_pruning_interpol
"""This the the strategy used to compensate for the removal of a 
neuron. It can be set to ``_compensate_for_pruning_interpol`` or
``_compensate_for_pruning_sum``. The first one is based on linear
combinations of the remaining neurons and should be preferred for
NMT networks as shown in https://arxiv.org/abs/1704.03279. The
parameter dictionary ``params_dict`` is updated accordingly.
    
Args:
    to_delete (list): List of i,j neuron pairs which were selected
                        for deletion
    layer (PrunableLayer): The layer which we are currently pruning
    params_dict (dict): Dictionary of numpy arrays (weight matrices)
"""

def add_in_mat(mat, dim, f_idx, t_idx):
    """Helper method to add a row or column to another one in a matrix.
    
    Args:
        mat (array): two dimensional numpy array.
        dim (int): 0 for rows, 1 for columns
        f_idx (int): Index of the first row or column
        t_idx (int): Index of the second row or column
    
    Returns:
        array. Matrix in which the first row or column is added to the
        second one.
    """
    if len(mat.shape) == 1:
        mat[t_idx] += mat[f_idx]
    elif dim == 0:
        mat[t_idx,:] += mat[f_idx,:]
    elif dim == 1:
        mat[:,t_idx] += mat[:,f_idx]
    return mat


def set_zero_in_mat(mat, dim, idx):
    """Helper method to set a row or column to zero.
    
    Args:
        mat (array): two dimensional numpy array.
        dim (int): 0 for rows, 1 for columns
        idx (int): Index of the row or column
    
    Returns:
        array. Matrix in which the ``idx``-the row or column is
        set to zero.
    """
    if len(mat.shape) == 1:
        mat[idx] = 0.0
    elif dim == 0:
        mat[idx,:] = 0.0
    elif dim == 1:
        mat[:,idx] = 0.0
    return mat


class Connection(object):
    """A connection between layer which is represented by a weight 
    matrix.
    """

    def __init__(self, direction, mat_name, dim, start_idx = 0.0):
        """Creates a new connection.
        
        Args:
            direction (string): 'in' if incoming, 'out' if outgoing
            mat_name (string): Name of the corresponding weight matrix
            dim (int): Dimension along which this layer is adjacent
            start_idx (float): Blocks uses single weight matrices for
                               both forget and reset gates. Set this to
                               0.5 if the layer is just connected to 
                               the lower half of the matrix.
        """
        self.mat_name = mat_name
        self.direction = direction
        self.dim = dim
        self.start_idx = start_idx


class PrunableInitializableFeedforwardSequence(FeedforwardSequence, 
                                               Initializable):
    """Version of ``InitializableFeedforwardSequence`` which allows
    keeping track of internal neuron activities.
    """
    
    def __init__(self, application_methods, **kwargs):
        self.pruning_variables_initialized = False
        self.layer_activities = []
        super(PrunableInitializableFeedforwardSequence, self).__init__(
                                       application_methods, 
                                       name='initializablefeedforwardsequence',
                                       **kwargs)

    @application
    def apply(self, *args):
        child_input = args
        for application_method in self.application_methods:
            output = application_method(*pack(child_input))
            if not self.pruning_variables_initialized:
                self.layer_activities.append(output)
            child_input = output
        self.pruning_variables_initialized = True
        return output


class PrunableSequenceGenerator(SequenceGenerator):
    """A sequence generator which keeps prunable layers as class 
    variables s.t. they can be accessed later.
    """
    
    def __init__(self, readout, transition, **kwargs):
        self.pruning_variables_initialized = False
        super(PrunableSequenceGenerator, self).__init__(
                                                    readout, 
                                                    transition, 
                                                    name='sequencegenerator', 
                                                    **kwargs)

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Adapted from ``BaseSequenceGenerator.cost_matrix``
        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        if not self.pruning_variables_initialized:
            self.results = results
            self.pruning_variables_initialized = True
        return costs


class PruningGradientDescent(GradientDescent):
    """This is a drop-in replacement for Blocks GradientDescent 
    optimizer. We intersperse the normal SGD updates with pruning
    operations and record the neuron activities after each training
    batch.
    """

    def __init__(self, 
                 prune_layer_configs, 
                 prune_layout_path, 
                 prune_every, 
                 prune_reset_every, 
                 prune_n_steps, 
                 nmt_model, 
                 **kwargs):
        """Constructor for the pruning SGD optimizer.
        
        Args:
            prune_layer_configs (dict): Layer configurations.
            prune_layout_path (string): Path to the network layout
                                        file.
            prune_every (int): Prune every n training iterations
            prune_reset_every (int): Reset neuron activity records
                                     every n training iterations.
            prune_n_steps (int): Number of pruning operations
            nmt_model (NMTModel): The initialized NMT model.
        """
        self.prune_every = prune_every
        self.prune_reset_every = prune_reset_every if prune_reset_every > 0 \
                                                   else prune_every
        self.n_batches = 0
        self.nmt_model = nmt_model
        self.prune_n_steps = prune_n_steps
        self.initialize_layers(prune_layer_configs, prune_layout_path)
        self.params_dict = nmt_model.training_model.get_parameter_dict()
        self.next_layer_to_prune = -len(self.prunable_layers)
        self.next_layer_to_reset = -len(self.prunable_layers)
        super(PruningGradientDescent, self).__init__(**kwargs)

    def initialize_layers(self, layer_configs, layout_path):
        """Initialize all layers which should be pruned in the course 
        of this training.
        
        Args:
            layer_configs (dict): Layer configurations.
            layout_path (string): Path to the network layout file.
        """
        conns = {}
        with open(layout_path) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if not len(parts) in [4, 5]:
                    logging.warn("Syntax error in prune layout file")
                    continue
                conn = Connection(parts[1], 
                                  parts[2], 
                                  int(parts[3]), 
                                  float(parts[4]) if len(parts) == 5 else 0.0)
                if parts[0] in conns:
                    conns[parts[0]].append(conn)
                else:
                    conns[parts[0]] = [conn]
        seq_gen = self.nmt_model.decoder.sequence_generator
        self.prunable_layers = []
        for conf in layer_configs:
            n,s = conf.split(":")
            maxout = False
            if n == 'encfwdgru':
                theano_var = self.nmt_model.encoder.bidir.forward
            elif n == 'encbwdgru':
                theano_var = self.nmt_model.encoder.bidir.backward
            elif n == 'decgru':
                theano_var = seq_gen.results['states']
            elif n == 'decmaxout':
                theano_var = seq_gen.readout.post_merge.layer_activities[1]
                maxout = True
            else:
                logging.warn("Unknown prunable layer name %s" % n)
                continue
            l = PrunableLayer(n, 
                              theano_var, 
                              int(s), 
                              self.prune_n_steps,
                              maxout=maxout)
            l.connections = conns.get(n, [])
            self.prunable_layers.append(l)
    
    def initialize(self):
        """Initialize the training algorithm.
        """
        logger.info("Initializing the training algorithm")
        update_values = [new_value for _, new_value in self.updates]
        activity_variables = [l.theano_variable for l in self.prunable_layers]
        logger.debug("Inferring graph inputs...")
        self.inputs = ComputationGraph(update_values).inputs
        logger.debug("Compiling training function...")
        self._function = theano.function(self.inputs, 
                                         activity_variables, 
                                         updates=self.updates, 
                                         **self.theano_func_kwargs)
        logger.info("The training algorithm is initialized")

    def process_batch(self, batch):
        """Overrides ``GradientDescent.process_batch`` and adds 
        recording neuron activities and pruning neurons to the
        pipeline.
        
        Args:
            batch (array): Current training batch
        """
        self.n_batches += 1
        self._validate_source_names(batch)
        ordered_batch = [batch[v.name] for v in self.inputs]
        activities = self._function(*ordered_batch)
        for activity, layer in zip(activities, self.prunable_layers):
            layer.register_activities(activity)
        if self.n_batches % self.prune_every == 0:
            self.next_layer_to_prune += 1
            if self.next_layer_to_prune >= 0:
                self.next_layer_to_prune %= len(self.prunable_layers)
                layer = self.prunable_layers[self.next_layer_to_prune]
                layer.prune(self.params_dict)
                layer.sanity_check(self.params_dict)
        if self.n_batches % self.prune_reset_every == 0:
            self.next_layer_to_reset += 1
            if self.next_layer_to_reset >= 0:
                self.next_layer_to_reset %= len(self.prunable_layers)
                self.prunable_layers[self.next_layer_to_reset].reset()

