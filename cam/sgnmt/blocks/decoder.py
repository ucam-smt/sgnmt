"""This module contains classes related to the decoder network in
encoder-decoder architectures. Note that this has nothing to do with
the search strategies which are also named "decoder" in SGNMT. Instead
of search algorithms, this module defines bricks which can be used to
combine the final encoder-decoder network.
"""

from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator, AbstractEmitter, TrivialFeedback)
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
import logging
from theano import tensor

from blocks.bricks import (Tanh, Logistic, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application, lazy
from blocks.bricks.cost import SquaredError
from blocks.bricks.parallel import Fork

from cam.sgnmt.blocks.pruning import PrunableSequenceGenerator, \
                                     PrunableInitializableFeedforwardSequence
from cam.sgnmt.blocks.attention import AlignmentAttention, \
    ThresholdedSequenceContentAttention, PushDownSequenceContentAttention, \
    PushDownThresholdedAttention, CoverageContentAttention, \
    SequenceMultiContentAttention


class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    """Empty helper class """
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        shp = [outputs.shape[i] for i in range(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)
        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class GRUInitialState(GatedRecurrent):
    """Gated Recurrent with special initial state.
    """
    
    def __init__(self, attended_dim, init_strategy="last", **kwargs):
        """Creates a GRU cell with special custom initial state.
        
        Args:
            attended_dim (int): Dimension of the annotations
            init_strategy (string): This string defines how to set the
                                    initial hidden layer state. "last"
                                    uses an affine transform plus tanh
                                    conditioned on the last annotation.
                                    "average" is similar, but uses the
                                    average of annotations. "constant"
                                    is independent of the source 
                                    annotations and always initializes
                                    the state with the same parameter
        """
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.init_strategy = init_strategy
        if self.init_strategy != 'constant':
            self.initial_transformer = MLP(activations=[Tanh()],
                                           dims=[attended_dim, self.dim],
                                           name='state_initializer')
            self.children.append(self.initial_transformer)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        """Returns the initial state depending on ``init_strategy``."""
        attended = kwargs['attended']
        if self.init_strategy == 'constant':
            initial_state = [tensor.repeat(self.parameters[2][None, :],
                                           batch_size,
                                           0)]
        elif self.init_strategy == 'last':
            initial_state = self.initial_transformer.apply(
                attended[0, :, -self.attended_dim:])
        elif self.init_strategy == 'average':
            initial_state = self.initial_transformer.apply(
                attended[:, :, -self.attended_dim:].mean(0))  
        else:
            logging.fatal("dec_init parameter %s invalid" % self.init_strategy)
        return initial_state

    def _allocate(self):
        """In addition to the GRU parameters ``state_to_state`` and 
        ``state_to_gates``, add the initial state if the search
        strategy is "constant".
        """
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                               name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                               name='state_to_gates'))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        if self.init_strategy == 'constant':
            self.parameters.append(shared_floatx_zeros((self.dim,),
                                                       name="initial_state"))
            add_role(self.parameters[2], INITIAL_STATE)


def _initialize_attention(attention_strategy,
                          seq_len, 
                          transition, 
                          representation_dim, 
                          att_dim,
                          attention_sources='s',
                          readout_sources='sfa',
                          memory="none",
                          memory_size=500):
    """Initializes the attention model according the configuration.
    
    Args:
        attention_strategy (string): "none" disables attention
                                     "content" is vanilla content-based
                                        attention (cf. Bahdanau, 2015)
                                     "nbest-N" is content-based 
                                               attention in which all
                                               alignment weights except
                                               the N best are set to
                                               zero.
                                     "stack" adds a neural stack memory
                                             structure.
                                     "parameterized" uses an trainable
                                                     alignment matrix
                                                     (cf. Neural 
                                                     Alignment model) 
        seq_len (int): Maximum sentence length
        transition (Recurrent): Recurrent transition brick of the
                                decoder network which is to be equipped
                                with an attention mechanism
        representation_dim (int): Dimension of source annotations
        att_dim (int): Number of hidden units in match vector
        attention_sources (string): Defines the sources used by the 
                                    attention model 's' for decoder
                                    states, 'f' for feedback
        readout_sources (string): Defines the sources used in the 
                                  readout network. 's' for decoder
                                  states, 'f' for feedback, 'a' for
                                  attention (context vector)
        memory (string): Defines the external memory structure which is
                         available to the decoder network. "none" does
                         not use any memory, "stack" enables a neural
                         stack.
        memory_size (int): Size of the memory structure. For example,
                           dimension of the vectors on the neural stack
    
    Returns:
        Tuple. First element is the attention, the second element is a
        list of source names for the readout network
    """
    attention = None
    att_src_names = []
    readout_src_names = []
    if 's' in readout_sources:
        readout_src_names.append('states')
    if 'f' in readout_sources:
        readout_src_names.append('feedback')
    if 's' in attention_sources:
        att_src_names.extend(transition.apply.states)
    if 'f' in attention_sources:
        att_src_names.append('feedback')
    if attention_strategy != 'none':
        if attention_strategy == 'parameterized':
            attention = AlignmentAttention(
                seq_len=seq_len,
                state_names=transition.apply.states,
                attended_dim=representation_dim, name="attention")
        elif attention_strategy == 'content':
            if memory == 'stack':
                attention = PushDownSequenceContentAttention(
                    stack_dim=memory_size,
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
            else:
                attention = SequenceContentAttention(
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
        elif 'content-' in attention_strategy:
            if memory == 'stack':
                logging.error("Memory 'stack' cannot used in combination "
                              "with multi content attention strategy (not "
                              "implemented yet)")
            else:
                _,n = attention_strategy.split('-')
                attention = SequenceMultiContentAttention(
                    n_att_weights=int(n),
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
        elif 'nbest-' in attention_strategy:
            _,n = attention_strategy.split('-')
            if memory == 'stack':
                attention = PushDownThresholdedAttention(
                    stack_dim=memory_size,
                    nbest=int(n),
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
            else:
                attention = ThresholdedSequenceContentAttention(
                    nbest=int(n),
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
        elif 'coverage-' in attention_strategy:
            _,n = attention_strategy.split('-')
            if memory == 'stack':
                logging.error("Memory 'stack' cannot used in combination "
                              "with coverage attention strategy (not "
                              "implemented yet)")
            else:
                attention = CoverageContentAttention(
                    max_fertility=int(n),
                    state_names=att_src_names,
                    attended_dim=representation_dim,
                    match_dim=att_dim, name="attention")
        else:
            logging.fatal("Unknown attention strategy '%s'"
                              % attention_strategy)
        if 'a' in readout_sources:
            readout_src_names.append(attention.take_glimpses.outputs[0])
    return attention,readout_src_names


class Decoder(Initializable):
    """Decoder of RNNsearch model which uses a full softmax output
    layer and embedding matrices for feedback.
    """

    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 state_dim,
                 att_dim,
                 maxout_dim,
                 representation_dim, 
                 attention_strategy='content',
                 attention_sources='s',
                 readout_sources='sfa',
                 memory='none',
                 memory_size=500,
                 seq_len=50,
                 init_strategy='last', 
                 make_prunable=False,
                 theano_seed=None, 
                 **kwargs):
        """Creates a new decoder brick.
        
        Args:
            vocab_size (int): Target language vocabulary size
            embedding_dim (int): Size of feedback embedding layer
            state_dim (int): Number of hidden units
            att_dim (int): Size of attention match vector
            maxout_dim (int): Size of maxout layer
            representation_dim (int): Dimension of source annotations
            attention_strategy (string): Which attention should be used
                                         cf.  ``_initialize_attention``
            attention_sources (string): Defines the sources used by the 
                                        attention model 's' for decoder
                                        states, 'f' for feedback
            readout_sources (string): Defines the sources used in the 
                                      readout network. 's' for decoder
                                      states, 'f' for feedback, 'a' for
                                      attention (context vector)
            memory (string): Which external memory should be used
                             (cf.  ``_initialize_attention``)
            memory_size (int): Size of the external memory structure
            seq_len (int): Maximum sentence length
            init_strategy (string): How to initialize the RNN state
                                    (cf.  ``GRUInitialState``)
            theano_seed: Random seed
        """
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=representation_dim / 2,
            init_strategy=init_strategy, 
            dim=state_dim,
            activation=Tanh(), 
            name='decoder')

        # Initialize the attention mechanism
        att_dim = att_dim if att_dim > 0 else state_dim
        self.attention,src_names = _initialize_attention(attention_strategy,
                                                         seq_len, 
                                                         self.transition, 
                                                         representation_dim, 
                                                         att_dim,
                                                         attention_sources,
                                                         readout_sources,
                                                         memory,
                                                         memory_size)

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        maxout_dim = maxout_dim if maxout_dim > 0 else state_dim
        post_layers = [Bias(dim=maxout_dim, name='maxout_bias').apply,
                       Maxout(num_pieces=2, name='maxout').apply,
                       Linear(input_dim=maxout_dim / 2, output_dim=embedding_dim,
                           use_bias=False, name='softmax0').apply,
                       Linear(input_dim=embedding_dim, name='softmax1').apply]
        if make_prunable:
            post_merge = PrunableInitializableFeedforwardSequence(post_layers)
        else:
            post_merge = InitializableFeedforwardSequence(post_layers)
        readout = Readout(
            source_names=src_names,
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=post_merge,
            merged_dim=maxout_dim)

        # Build sequence generator accordingly
        if make_prunable:
            self.sequence_generator = PrunableSequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )
        else:
            self.sequence_generator = SequenceGenerator(
                readout=readout,
                transition=self.transition,
                attention=self.attention,
                fork=Fork([name for name in self.transition.apply.sequences
                           if name != 'mask'], prototype=Linear())
            )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]

    @application
    def generate(self, source_shape, representation, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_shape[1],
            batch_size=source_shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_shape).T,
            **kwargs)


class NoLookupEmitter(AbstractEmitter):
    """Emitter brick in the readout network without embedding (i.e.
    with target sparse feature maps). Directly emits the readouts
    """
    
    @lazy(allocation=['readout_dim'])
    def __init__(self, initial_output, readout_dim, cost_brick, **kwargs):
        super(NoLookupEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim
        self.initial_output = initial_output
        self.cost_brick = cost_brick
        self.children = [cost_brick]

    @application
    def probs(self, readouts):
        return readouts

    @application
    def emit(self, readouts):
        return readouts

    @application
    def cost(self, readouts, outputs):
        return self.cost_brick.apply(readouts, outputs)

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size, self.readout_dim))
    
    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(NoLookupEmitter, self).get_dim(name)


class NoLookupDecoder(Initializable):
    """This is the decoder implementation without embedding layer or
    softmax. The target sentence is represented as a sequence of #
    vectors as defined by the sparse feature map.
    """

    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 state_dim,
                 att_dim,
                 maxout_dim,
                 representation_dim,
                 attention_strategy='content',
                 attention_sources='s',
                 readout_sources='sfa',
                 memory='none',
                 memory_size=500,
                 seq_len=50,
                 init_strategy='last', 
                 theano_seed=None, 
                 **kwargs):
        """Creates a new decoder brick without embedding.
        
        Args:
            vocab_size (int): Target language vocabulary size
            embedding_dim (int): Size of feedback embedding layer
            state_dim (int): Number of hidden units
            att_dim (int): Size of attention match vector
            maxout_dim (int): Size of maxout layer
            representation_dim (int): Dimension of source annotations
            attention_strategy (string): Which attention should be used
                                         cf.  ``_initialize_attention``
            attention_sources (string): Defines the sources used by the 
                                        attention model 's' for decoder
                                        states, 'f' for feedback
            readout_sources (string): Defines the sources used in the 
                                      readout network. 's' for decoder
                                      states, 'f' for feedback, 'a' for
                                      attention (context vector)
            memory (string): Which external memory should be used
                             (cf.  ``_initialize_attention``)
            memory_size (int): Size of the external memory structure
            seq_len (int): Maximum sentence length
            init_strategy (string): How to initialize the RNN state
                                    (cf.  ``GRUInitialState``)
            theano_seed: Random seed
        """
        super(NoLookupDecoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRUInitialState(
            attended_dim=state_dim,
            init_strategy=init_strategy,
            dim=state_dim,
            activation=Tanh(),
            name='decoder')

        # Initialize the attention mechanism
        att_dim = att_dim if att_dim > 0 else state_dim
        self.attention,src_names = _initialize_attention(attention_strategy,
                                                         seq_len, 
                                                         self.transition, 
                                                         representation_dim, 
                                                         att_dim,
                                                         attention_sources,
                                                         readout_sources,
                                                         memory,
                                                         memory_size)

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        maxout_dim = maxout_dim if maxout_dim > 0 else state_dim
        readout = Readout(
            source_names=src_names,
            readout_dim=embedding_dim,
            emitter=NoLookupEmitter(initial_output=-1,
                                    readout_dim=embedding_dim,
                                    cost_brick=SquaredError()),
            #                        cost_brick=CategoricalCrossEntropy()),
            feedback_brick=TrivialFeedback(output_dim=embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=maxout_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=maxout_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Logistic(name='softmax1').apply]),
            merged_dim=maxout_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'representation_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, representation_mask,
             target_sentence, target_sentence_mask):

        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended': representation,
            'attended_mask': representation_mask}
        )

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]

    @application
    def generate(self, source_shape, representation, **kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * source_shape[1],
            batch_size=source_shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_shape).T,
            **kwargs)        
        
