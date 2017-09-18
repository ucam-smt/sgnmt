"""This module contains implementations of the attention model in NMT.
In addition to the vanilla content based attention from Bahdanau we
support thresholded attention for resharpening the focus and 
parameterized for the Neural Alignment Model (NAM) in which the
attention is defined by a trainable alignment matrix. We also consider
using an external memory like a neural stack as part of the attention.
"""

from blocks.bricks import Initializable, \
                          Linear, \
                          MLP, \
                          Logistic, \
                          Tanh, \
                          Sequence, \
                          Feedforward
from blocks.bricks.attention import SequenceContentAttention, \
                                    GenericSequenceAttention, \
                                    ShallowEnergyComputer
from blocks.bricks.base import application, lazy
from blocks.bricks.parallel import Parallel

from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans
from theano import tensor


class AlignmentAttention(GenericSequenceAttention, Initializable):
    """This is the parameterized attention for the neural alignment
    model. Instead of using a feedforward network as in Bahdanau to
    decide on the energies, we use a trainable alignment matrix. This
    attention is intended to be used for alignment. During training,
    the sentence pair should not change, otherwise the alignment matrix
    will be a mixture of multiple alignments. Therefore, the alignment
    matrix parameter should be reset to the initial value after each
    optimization. In the neural alignment model, this is done by the
    ``NextSentenceExtension`` extension.
    """
    
    def __init__(self, seq_len, **kwargs):
        """Sole constructor.
        
        Args:
            seq_len: Maximum length of any source or target sentence.
                     This defines the size of the default alignment
                     matrix.
        """
        self.matrix_size = seq_len + 1
        super(AlignmentAttention, self).__init__(**kwargs)

    def _allocate(self):
        """Allocates the alignment matrix parameter """
        align_matrix = shared_floatx_nans((self.matrix_size, self.matrix_size),
                                          name='alignment_matrix')
        add_role(align_matrix, WEIGHT)
        self.parameters.append(align_matrix)

    def _initialize(self):
        """Initializes the alignment matrix parameter """
        align_matrix = self.parameters[0]
        self.weights_init.initialize(align_matrix, self.rng)

    @application(outputs=['weighted_averages', 'weights', 'matrix_col'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, matrix_col=None, **states):
        """In addition to the glimpses ``weighted_averages`` and 
        ``weights`` as in ``SequenceContentAttention``, we use the
        variable ``matrix_col`` to refer to the current column in the
        alignment matrix.
        
        Args:
            attended (Variable): Source annotations
            preprocessed_attended (Variable): Not used, but required by
                                              certain bricks in Blocks.
            attended_mask (Variable): Source mask
            matrix_col (Variable): Current column in the alignment 
                                   matrix
            \*\*states (Variable): Decoder state
        
        Returns:
            Tuple. The glimpses ``weighted_averages``, ``weights``, and
            ``matrix_col``. The first one is used as context vector. 
        """
        align_matrix = self.parameters[0]
        energies = align_matrix[:attended.shape[0], matrix_col]
    
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        next_matrix_col = matrix_col + 1
        return weighted_averages, weights.T, next_matrix_col

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        """Defines the ``inputs`` decoration for ``take_glimpses``. """
        return (['attended',
                 'preprocessed_attended', 
                 'attended_mask', 
                 'matrix_col'] +
                self.state_names)

    @application(outputs=['weighted_averages', 'weights', 'matrix_col'])
    def initial_glimpses(self, batch_size, attended):
        """Initial glimpses are set to zero. """
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0])),
                tensor.zeros((batch_size,), dtype='int32')]

    def get_dim(self, name):
        """Get dimensions of variables. Delegates to super class if
        ``name`` is not used in this class.
        """
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        if name in ['matrix_col']:
            return 0
        return super(AlignmentAttention, self).get_dim(name)


class ThresholdedSequenceContentAttention(SequenceContentAttention):
    """This class can be used to resharpen the normal content based
    attention. Instead of using the normal softmax over the energies,
    it keeps only the n-best energies and set all the others to zero.
    This results in a sharper focus and avoids blurry alignment weights
    on long sequences.
    
    Note that this attention can be used during decoding in place of
    the normal content-based attention without retraining. 
    """
    
    def __init__(self, nbest=1, **kwargs):
        """Sole constructor.
        
        Args:
            nbest (int): Number of energies to keep (all others are set
                         to zero.
        """
        self.nbest = nbest
        super(ThresholdedSequenceContentAttention, self).__init__(**kwargs) 

    @application
    def compute_weights(self, energies, attended_mask):
        """Overrides ``SequenceContentAttention.compute_weights()``.
        Instead of a normal softmax, it sets most of the energies to
        zero, resulting in sharp attention. If ``self.nbest`` equals 1,
        the thresholded attention always sets its full attention to one
        single source annotation.
        
        Args:
            energies (Variable): Energies computed by the 
                                 energy_computer
            attended_mask (Variable): Source sentence mask
        
        Returns:
            Variable. Thresholded alignment weights
        """
        # Stabilize energies first and then exponentiate
        energies = energies - energies.max(axis=0)
        unnormalized_weights = tensor.exp(energies)
        if attended_mask:
            unnormalized_weights *= attended_mask
        
        # Set everything to zero except the ``nbest`` best entries
        best_energies = unnormalized_weights.sort(axis=0)[-self.nbest:]
        min_energy = best_energies[0]
        thresholded_weights = tensor.switch(unnormalized_weights >= min_energy,
                                            unnormalized_weights,
                                            0.0)

        # If mask consists of all zeros use 1 as the normalization coefficient
        normalization = (thresholded_weights.sum(axis=0) +
                         tensor.all(1 - attended_mask, axis=0))
        return thresholded_weights / normalization


class PushDownSequenceContentAttention(SequenceContentAttention, Initializable):
    """Adds an external memory structure in form of a neural stack to
    the decoder. The neural stack is operated through a pop operation,
    a push operation, and an input variable, which all are computed
    from the decoder state. This neural stack implementation is similar
    to Mikolovs model:
    
    - Apply the (continuous) pop operation if the pop gate is on
    - Read the top element on the stack
    - Push the stack input vector if the push gate is on
    - Concatenate the read element from the stack to the weighted
      averages of source annotations to obtain the final context
      vector
      
    Note that this implementation realizes a stack with limited depth
    because Blocks didn't allow to have glimpses of varying size. In
    practice, however, we think that a limited size is appropriate for
    machine translation.
    """
    
    def __init__(self, stack_dim=500, **kwargs):
        """Sole constructor.
        
        Args:
            stack_dim (int): Size of vectors on the stack.
        """
        super(PushDownSequenceContentAttention, self).__init__(**kwargs)
        self.stack_dim = stack_dim
        self.max_stack_depth = 25
        
        self.stack_op_names = self.state_names + ['weighted_averages']
        
        self.stack_pop_transformer = MLP(activations=[Logistic()], dims=None)
        self.stack_pop_transformers = Parallel(
                                        input_names=self.stack_op_names,
                                        prototype=self.stack_pop_transformer,
                                        name="stack_pop")
        
        self.stack_push_transformer = MLP(activations=[Logistic()], dims=None)
        self.stack_push_transformers = Parallel(
                                        input_names=self.stack_op_names,
                                        prototype=self.stack_push_transformer,
                                        name="stack_push")
        
        self.stack_input_transformer = Linear()
        self.stack_input_transformers = Parallel(
                                        input_names=self.stack_op_names,
                                        prototype=self.stack_input_transformer,
                                        name="stack_input")
        self.children.append(self.stack_pop_transformers)
        self.children.append(self.stack_push_transformers)
        self.children.append(self.stack_input_transformers)
        
    def _push_allocation_config(self):
        """Sets the dimensions of the stack operation networks """
        super(PushDownSequenceContentAttention, self)._push_allocation_config()
        self.stack_op_dims = self.state_dims + [self.attended_dim]
        n_states = len(self.stack_op_dims)
        self.stack_pop_transformers.input_dims = self.stack_op_dims
        self.stack_pop_transformers.output_dims = [1] * n_states
        
        self.stack_push_transformers.input_dims = self.stack_op_dims
        self.stack_push_transformers.output_dims = [1] * n_states
        
        self.stack_input_transformers.input_dims = self.stack_op_dims
        self.stack_input_transformers.output_dims = [self.stack_dim] * n_states
    
    def _allocate(self):
        """Allocates the single parameter of this brick: the initial
        element on the stack.
        """
        self.parameters.append(shared_floatx_nans((1, self.stack_dim),
                               name='init_stack'))
        add_role(self.parameters[-1], INITIAL_STATE)
    
    def _initialize(self):
        """Initializes the initial element on the stack with zero. """
        self.biases_init.initialize(self.parameters[-1], self.rng)

    @application(outputs=['context_vector', 'weights', 'stack'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, stack=None, **states):
        """This method is an extension to ``take_glimpses`` in
        ``SequenceContentAttention``. After computing the weighted
        averages of source annotations, it operates the stack, i.e.
        pops the top element, reads out the top of the stack, and 
        pushes a new element. The first glimpse ``context_vector`` is
        the concatenation of weighted source annotations and stack
        output.
        
        Args:
            attended (Variable): Source annotations
            preprocessed_attended (Variable): Transformed source 
                                              annotations used to
                                              compute energies
            attended_mask (Variable): Source mask
            stack (Variable): Current state of the stack
            \*\*states (Variable): Decoder state 
        
        Returns:
            Tuple. The first element is used as context vector for the
            decoder state update. ``stack`` is a recurrent glimpse which
            is used in the next ``take_glimpse`` iteration.
        """
        energies = self.compute_energies(attended, preprocessed_attended,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        
        stack_op_input = states
        stack_op_input['weighted_averages'] = weighted_averages
        
        stack_pop = sum(self.stack_pop_transformers.apply(
            as_dict=True, **stack_op_input).values())
        stack_push = sum(self.stack_push_transformers.apply(
            as_dict=True, **stack_op_input).values())
        stack_input = sum(self.stack_input_transformers.apply(
            as_dict=True, **stack_op_input).values())
        
        # the stack has shape (batch_size, stack_depth, stack_dim)
        batch_size = stack.shape[0]
        stack_dim = stack_input.shape[1]
        default_stack_entry = tensor.repeat(self.parameters[-1][None, :, :], 
                                            batch_size, 
                                            0)
    
        pushed_stack = tensor.concatenate([stack_input.reshape((batch_size, 
                                                                1, 
                                                                stack_dim)),
                                           stack[:,1:,:]], 
                                          axis=1)
        popped_stack = tensor.concatenate([stack[:,:-1,:], 
                                           default_stack_entry],
                                          axis=1)
        pop_gate = stack_pop.reshape((batch_size, 1, 1))
        push_gate = stack_push.reshape((batch_size, 1, 1))
        read_stack = pop_gate * popped_stack + (1.0 - pop_gate) * stack
        stack_output = read_stack[:,0,:]
        new_stack = push_gate * pushed_stack + (1.0 - push_gate) * read_stack

        context_vector = tensor.concatenate([weighted_averages, stack_output],
                                            axis=1)
        return context_vector, weights.T, new_stack

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        """Defines the ``inputs`` decoration for ``take_glimpses``. """
        return (['attended', 'preprocessed_attended', 'attended_mask', 'stack'] +
                self.state_names)

    @application(outputs=['context_vector', 'weights', 'stack'])
    def initial_glimpses(self, batch_size, attended):
        """The stack is initialized with the default entry. All other
        glimpses are initialized with zero.
        """
        default_stack_entry = tensor.repeat(self.parameters[-1][None, :, :], 
                                            batch_size, 
                                            0)
        return [tensor.zeros((batch_size, self.attended_dim + self.stack_dim)),
                tensor.zeros((batch_size, attended.shape[0])),
                tensor.repeat(default_stack_entry, 
                              self.max_stack_depth, 
                              1)]

    def get_dim(self, name):
        """Get dimensions of variables. Delegates to super class if
        ``name`` is not used in this class.
        """
        if name in ['context_vector']:
            return self.attended_dim + self.stack_dim
        if name in ['weights']:
            return 0
        if name in ['stack']:
            return self.max_stack_depth,self.stack_dim
        return super(PushDownSequenceContentAttention, self).get_dim(name) 


class PushDownThresholdedAttention(PushDownSequenceContentAttention,
                                   ThresholdedSequenceContentAttention):
    """This class allows to use thresholded attention in combination 
    with an external neural stack memory.  
    """
    
    def __init__(self, **kwargs):
        super(PushDownThresholdedAttention, self).__init__(**kwargs)


class SelfAttendableContentAttention(SequenceContentAttention):
    """This is a variation of ``SequenceContentAttention`` which can
    also attend to own previous states. This might be useful for
    creating higher lever representations in the encoder network.
    
    Note that because of restrictions in Blocks, the number of previous
    states to which we can attend to must be fixed. Therefore, the
    constructor expects to specify the number of recurrent
    steps.
    """
    
    def __init__(self, match_dim, num_steps, **kwargs):
        """Constructor which delegates most of the work to
        ``SequenceContentAttention.__init__``.
        
        Args:
            match_dim (int): Dimension of the match vectors
            num_steps (int): Maximum number of steps. Used to allocate
                             the Theano variable storing the previous
                             preprocessed states
        """
        self.num_steps = num_steps 
        super(SelfAttendableContentAttention, self).__init__(match_dim, 
                                                             **kwargs)

    @application(outputs=['weighted_averages', 
                          'weights', 
                          'step', 
                          'attended_own', 
                          'attended_own_mask'])
    def take_glimpses(self, 
                      attended, 
                      preprocessed_attended=None,
                      attended_mask=None, 
                      step=None, 
                      attended_own=None, 
                      attended_own_mask=None, 
                      **states):
        """Similar to ``SequenceContentAttention.take_glimpses()``, 
        but keeps the previous states in ``attended_own`` in order to
        be able to attend to them.
        """
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        all_attended = tensor.concatenate([preprocessed_attended,
                                           attended_own], axis=0)
        all_mask = tensor.concatenate([attended_mask, attended_own_mask],
                                      axis=0)
        energies = self.compute_energies(all_attended, states)
        weights = self.compute_weights(energies, all_mask)
        weighted_averages = self.compute_weighted_averages(weights,
                                                           all_attended)
        # Update attended_own
        attended_own_mask = tensor.set_subtensor(attended_own_mask[step,:], 
                                                 attended_mask[step])
        attended_own = tensor.set_subtensor(attended_own[step], 
                                            self.preprocess(states['states']))
        step = (step + 1) % self.num_steps
        return weighted_averages, weights.T, step, attended_own, attended_own_mask

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        """Inputs for ``take_climpses``. """
        return (['attended', 
                 'preprocessed_attended', 
                 'attended_mask', 
                 'step', 
                 'attended_own', 
                 'attended_own_mask'] + self.state_names)

    @application(outputs=['weighted_averages', 
                          'weights', 
                          'step', 
                          'attended_own', 
                          'attended_own_mask'])
    def initial_glimpses(self, batch_size, attended):
        """Sets all inital glimpses to zero. """
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0]+self.num_steps)),
                tensor.zeros((batch_size,), dtype='int32'),
                tensor.zeros((self.num_steps, batch_size, self.attended_dim)),
                tensor.zeros((self.num_steps,batch_size))]

    def get_dim(self, name):
        """Get dimensions of variables. Delegates to super class if
        ``name`` is not used in this class.
        """
        if name in ['attended_own']:
            return self.num_steps, self.attended_dim
        if name in ['attended_own_mask']:
            return self.num_steps
        return super(SelfAttendableContentAttention, self).get_dim(name)


class CoverageContentAttention(GenericSequenceAttention, Initializable):
    """This is the 'linguistic' coverage model from Tu et al., 2016. 
    The fertility of each source annotation is estimated with a linear
    transform followed by a sigmoid times N (N is the maximum fertility)
    The coverage model keeps track of the attention record for each
    annotation and feeds the cumulative record divided by the fertility
    to the match vector which eventually determines the attention 
    weight.
    
    This code base of this implementation is close to 
    ``SequenceContentAttention``.
    """
    @lazy(allocation=['match_dim'])
    def __init__(self,
                 match_dim,
                 max_fertility, 
                 state_transformer=None,
                 attended_transformer=None, 
                 fertility_transformer=None,
                 att_record_transformer=None,
                 energy_computer=None, 
                 **kwargs):
        """Creates an attention brick with 'linguistic' coverage.
        Compare with ``SequenceContentAttention``.
        
        Args:
            match_dim (int): Dimensionality of the match vector
            max_fertility (float): Maximum fertility of a source
                                   annotation (N in Tu et al.). If
                                   this is set to 0 or smaller, we fix
                                   fertilities to 1 and do not estimate
                                   them.
            state_transformer (Brick): Transformation for the decoder
                                       state
            attended_transformer (Brick): Transformation for the source
                                          annotations
            fertility_transformer (Brick): Transformation which 
                                           calculates fertilities
            att_record_transformer (Brick): Transformation for the 
                                            attentional records
            energy_computer (Brick): Sub network for calculating the 
                                     energies from the match vector 
        """ 
        
        super(CoverageContentAttention, self).__init__(**kwargs)
        self.use_fertility = (max_fertility > 0.0001)
        self.max_fertility = max_fertility
        if not state_transformer:
            state_transformer = Linear(use_bias=False)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        
        if not att_record_transformer:
            att_record_transformer = Linear(name="att_record_trans")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.attended_transformer = attended_transformer
        self.att_record_transformer = att_record_transformer
        self.energy_computer = energy_computer
        
        self.children = [self.state_transformers, 
                         attended_transformer,
                         att_record_transformer, 
                         energy_computer]
        
        if self.use_fertility:
            if not fertility_transformer:
                fertility_transformer = MLP(activations=[Logistic()],
                                            name='fertility_trans')
            self.fertility_transformer = fertility_transformer
            self.children.append(fertility_transformer)

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.att_record_transformer.input_dim = 1
        self.att_record_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1
        if self.use_fertility:
            self.fertility_transformer.dims = [self.attended_dim, 1]

    @application
    def compute_energies(self, 
                         attended, 
                         preprocessed_attended, 
                         att_records,
                         states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        transformed_att_records = self.att_record_transformer.apply(att_records.dimshuffle((1, 0, 2)))
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                                preprocessed_attended)
        match_vectors = match_vectors + transformed_att_records
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights', 'att_records'])
    def take_glimpses(self, 
                      attended, 
                      preprocessed_attended=None,
                      attended_mask=None, 
                      att_records=None, 
                      **states):
        energies = self.compute_energies(attended, 
                                         preprocessed_attended,
                                         att_records,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        if self.use_fertility:
            fertilities = self.max_fertility * self.fertility_transformer.apply(attended)
            # Theanos optimizer ensures that fertilities are computed only once
            att_records = att_records + weights.dimshuffle((1, 0, 'x')) / \
                                        fertilities.dimshuffle((1, 0, 2))
        else:
            att_records = att_records + weights.dimshuffle((1, 0, 'x'))
        weighted_averages = self.compute_weighted_averages(weights, attended)
        return weighted_averages, weights.T, att_records

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 
                 'preprocessed_attended', 
                 'attended_mask', 
                 'att_records'] + self.state_names)

    @application(outputs=['weighted_averages', 'weights', 'att_records'])
    def initial_glimpses(self, batch_size, attended):
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0])),
                tensor.zeros((batch_size, attended.shape[0], 1))]

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Preprocess the sequence for computing attention weights.

        Args:
            attended (TensorVariable): The attended sequence, time is 
                                       the 1-st dimension.
        """
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights', 'att_records']:
            return 0
        return super(CoverageContentAttention, self).get_dim(name)


class TreeAttention(GenericSequenceAttention, Initializable):
    """This attention replaces the weighted average in the vanilla 
    attention mechanism with a recursive network which is similar to
    the recurrent autoencoder. The source annotations are merged
    recursively until a single representation is obtained. The merge
    network takes two representations and the last decoder state as 
    input and outputs another representation which serves as input to
    the next merge operation.
    """
    
    def __init__(self, **kwargs):
        super(TreeAttention, self).__init__(**kwargs)
        state_transformer = Linear()
        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        self.parent1_transformer = Linear(name="parent1_trans")
        self.parent2_transformer = Linear(name="parent2_trans")

        self.children = [self.state_transformers,
                         self.parent1_transformer,
                         self.parent2_transformer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.attended_dim
                                               for name in self.state_names]
        self.parent1_transformer.input_dim = self.attended_dim
        self.parent1_transformer.output_dim = self.attended_dim
        self.parent2_transformer.input_dim = self.attended_dim
        self.parent2_transformer.output_dim = self.attended_dim

    @application
    def compute_energies(self, attended, preprocessed_attended, states):
        # TODO implement this
        pass
        #if not preprocessed_attended:
        #    preprocessed_attended = self.preprocess(attended)
        #transformed_states = self.state_transformers.apply(as_dict=True,
        #                                                   **states)
        ## Broadcasting of transformed states should be done automatically
        #match_vectors = sum(transformed_states.values(),
        #                    preprocessed_attended)
        #energies = self.energy_computer.apply(match_vectors).reshape(
        #    match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        #return energies

    @application(outputs=['context_vector'])
    def take_glimpses(self, 
                      attended,
                      attended_mask=None, 
                      **states):
        pass
        # TODO implement this
        #energies = self.compute_energies(attended, preprocessed_attended,
        #                                 states)
        #weights = self.compute_weights(energies, attended_mask)
        #weighted_averages = self.compute_weighted_averages(weights, attended)
        #return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'attended_mask'] + self.state_names)

    @application(outputs=['context_vector'])
    def initial_glimpses(self, batch_size, attended):
        return [tensor.zeros((batch_size, self.attended_dim))]

    def get_dim(self, name):
        if name in ['context_vector']:
            return self.attended_dim
        return super(TreeAttention, self).get_dim(name)


class SequenceMultiContentAttention(GenericSequenceAttention, Initializable):
    
    @lazy(allocation=['match_dim'])
    def __init__(self, n_att_weights, match_dim, state_transformer=None,
                 attended_transformer=None, energy_computer=None, **kwargs):
        super(SequenceContentAttention, self).__init__(**kwargs)
        self.n_att_weights = n_att_weights
        if not state_transformer:
            state_transformer = Linear(use_bias=False)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        if not energy_computer:
            energy_computer = MultiShallowEnergyComputer(n_att_weights,
                                                         name="energy_comp")
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, attended_transformer,
                         energy_computer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application
    def compute_energies(self, attended, preprocessed_attended, states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_attended)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, **states):
        r"""Compute attention weights and produce glimpses.

        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.
        preprocessed_attended : :class:`~tensor.TensorVariable`
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        attended_mask : :class:`~tensor.TensorVariable`
            A 0/1 mask specifying available data. 0 means that the
            corresponding sequence element is fake.
        \*\*states
            The states of the network.

        Returns
        -------
        weighted_averages : :class:`~theano.Variable`
            Linear combinations of sequence elements with the attention
            weights.
        weights : :class:`~theano.Variable`
            The attention weights. The first dimension is batch, the second
            is time.

        """
        energies = self.compute_energies(attended, preprocessed_attended,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended', 'attended_mask'] +
                self.state_names)

    @application(outputs=['weighted_averages', 'weights'])
    def initial_glimpses(self, batch_size, attended):
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0]))]

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Preprocess the sequence for computing attention weights.

        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The attended sequence, time is the 1-st dimension.

        """
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(SequenceContentAttention, self).get_dim(name)


class MultiShallowEnergyComputer(Sequence, Initializable, Feedforward):
    """A simple energy computer: first tanh, then weighted sum."""
    @lazy()
    def __init__(self, n_att_weights, **kwargs):
        self.n_att_weights = n_att_weights
        super(ShallowEnergyComputer, self).__init__(
            [Tanh().apply, Linear(use_bias=False).apply], **kwargs)

    @property
    def input_dim(self):
        return self.children[1].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[1].input_dim = value

    @property
    def output_dim(self):
        return self.children[1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[1].output_dim = value
        