"""Contains the different implementations of the encoder network in a
NMT encoder-decoder model. This module is accessible via the ``model`` 
module and should not be used directly.
"""

from blocks.bricks import Tanh, Linear, Initializable
from blocks.bricks.attention import SequenceContentAttention, AttentionRecurrent
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
import logging
from picklable_itertools.extras import equizip

from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.utils import shared_floatx_zeros
from theano import tensor
from toolz import merge

from cam.sgnmt.blocks.attention import SelfAttendableContentAttention


class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""
    def __init__(self, prototype, **kwargs):
        self.pruning_variables_initialized = False
        super(BidirectionalWMT15, self).__init__(prototype, **kwargs)

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates 
        outputs.
        """
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        if not self.pruning_variables_initialized:
            self.forward = forward[0]
            self.backward = backward[0]
            self.pruning_variables_initialized = True
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """A generalized version of the vanilla encoder of the RNNsearch 
    model which supports different numbers of layers. Zero layers 
    represent non-recurrent encoders.
    """

    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 n_layers, 
                 skip_connections, 
                 state_dim, 
                 **kwargs):
        """Sole constructor.
        
        Args:
            vocab_size (int): Source vocabulary size
            embedding_dim (int): Dimension of the embedding layer
            n_layers (int): Number of layers. Layers share the same
                            weight matrices.
            skip_connections (bool): Skip connections connect the
                                     source word embeddings directly 
                                     with deeper layers to propagate 
                                     the gradient more efficiently
            state_dim (int): Number of hidden units in the recurrent
                             layers.
        """
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.skip_connections = skip_connections

        self.lookup = LookupTable(name='embeddings')
        if self.n_layers >= 1:
            self.bidir = BidirectionalWMT15(
                GatedRecurrent(activation=Tanh(), dim=state_dim))
            self.fwd_fork = Fork(
                [name for name in self.bidir.prototype.apply.sequences
                 if name != 'mask'], prototype=Linear(), name='fwd_fork')
            self.back_fork = Fork(
                [name for name in self.bidir.prototype.apply.sequences
                 if name != 'mask'], prototype=Linear(), name='back_fork')
            self.children = [self.lookup, self.bidir,
                             self.fwd_fork, self.back_fork]
            if self.n_layers > 1: # Deep encoder
                self.mid_fwd_fork = Fork(
                    [name for name in self.bidir.prototype.apply.sequences
                     if name != 'mask'], prototype=Linear(), name='mid_fwd_fork')
                self.mid_back_fork = Fork(
                    [name for name in self.bidir.prototype.apply.sequences
                     if name != 'mask'], prototype=Linear(), name='mid_back_fork')
                self.children.append(self.mid_fwd_fork)
                self.children.append(self.mid_back_fork)
        elif self.n_layers == 0:
            self.embedding_dim = state_dim*2
            self.children = [self.lookup]
        else:
            logging.fatal("Number of encoder layers must be non-negative")

    def _push_allocation_config(self):
        """Sets the parameters of sub bricks """
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        if self.n_layers >= 1:
            self.fwd_fork.input_dim = self.embedding_dim
            self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
            self.back_fork.input_dim = self.embedding_dim
            self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]
            if self.n_layers > 1: # Deep encoder
                inp_dim = self.state_dim * 2
                if self.skip_connections:
                    inp_dim += self.embedding_dim
                self.mid_fwd_fork.input_dim = inp_dim
                self.mid_fwd_fork.output_dims = [
                                        self.bidir.children[0].get_dim(name)
                                        for name in self.fwd_fork.output_names]
                self.mid_back_fork.input_dim = inp_dim
                self.mid_back_fork.output_dims = [
                                        self.bidir.children[1].get_dim(name)
                                        for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation', 'representation_mask'])
    def apply(self, source_sentence, source_sentence_mask):
        """Produces source annotations, either non-recurrently or with
        a bidirectional RNN architecture.
        """
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        if self.n_layers >= 1:
            representation = self.bidir.apply(
                merge(self.fwd_fork.apply(embeddings, as_dict=True),
                      {'mask': source_sentence_mask}),
                merge(self.back_fork.apply(embeddings, as_dict=True),
                      {'mask': source_sentence_mask})
            )
            for _ in xrange(self.n_layers-1):
                if self.skip_connections:
                    inp = tensor.concatenate([representation, embeddings],
                                             axis=2)
                else:
                    inp = representation
                representation = self.bidir.apply(
                    merge(self.mid_fwd_fork.apply(inp, as_dict=True),
                          {'mask': source_sentence_mask}),
                    merge(self.mid_back_fork.apply(inp, as_dict=True),
                          {'mask': source_sentence_mask})
                )
        else:
            representation = embeddings
        return representation, source_sentence_mask


class DeepBidirectionalEncoder(Initializable):
    """This encoder is a multi-layered version of 
    ``BidirectionalEncoder`` where parameters between layers are not
    shared. 
    """

    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 n_layers, 
                 skip_connections, 
                 state_dim, 
                 **kwargs):
        """Sole constructor.
        
        Args:
            vocab_size (int): Source vocabulary size
            embedding_dim (int): Dimension of the embedding layer
            n_layers (int): Number of layers. Layers share the same
                            weight matrices.
            skip_connections (bool): Skip connections connect the
                                     source word embeddings directly 
                                     with deeper layers to propagate 
                                     the gradient more efficiently
            state_dim (int): Number of hidden units in the recurrent
                             layers.
        """
        super(DeepBidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.skip_connections = skip_connections
        self.lookup = LookupTable(name='embeddings')
        self.bidirs = []
        self.fwd_forks =[]
        self.back_forks = []        
        for i in xrange(self.n_layers):
            bidir = BidirectionalWMT15(
                GatedRecurrent(activation=Tanh(), dim=state_dim),
                name='bidir%d' % i)
            self.bidirs.append(bidir)
            self.fwd_forks.append(Fork(
                [name for name in bidir.prototype.apply.sequences 
                 if name != 'mask'], 
                prototype=Linear(), name='fwd_fork%d' % i))
            self.back_forks.append(Fork(
                [name for name in bidir.prototype.apply.sequences
                 if name != 'mask'], 
                prototype=Linear(), name='back_fork%d' % i))
        self.children = [self.lookup] \
                        + self.bidirs \
                        + self.fwd_forks \
                        + self.back_forks

    def _push_allocation_config(self):
        """Sets the parameters of sub bricks """
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim
        
        self.fwd_forks[0].input_dim = self.embedding_dim
        self.fwd_forks[0].output_dims = [
                                self.bidirs[0].children[0].get_dim(name)
                                    for name in self.fwd_forks[0].output_names]
        self.back_forks[0].input_dim = self.embedding_dim
        self.back_forks[0].output_dims = [
                                self.bidirs[0].children[1].get_dim(name)
                                    for name in self.back_forks[0].output_names]
        for i in xrange(1, self.n_layers):
            inp_dim = self.state_dim * 2
            if self.skip_connections:
                inp_dim += self.embedding_dim
            self.fwd_forks[i].input_dim = inp_dim
            self.fwd_forks[i].output_dims = [
                                    self.bidirs[i].children[0].get_dim(name)
                                    for name in self.fwd_forks[i].output_names]
            self.back_forks[i].input_dim = inp_dim
            self.back_forks[i].output_dims = [
                                    self.bidirs[i].children[1].get_dim(name)
                                    for name in self.back_forks[i].output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation', 'representation_mask'])
    def apply(self, source_sentence, source_sentence_mask):
        """Produces source annotations, either non-recurrently or with
        a bidirectional RNN architecture.
        """
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T
        embeddings = self.lookup.apply(source_sentence)
        representation = self.bidirs[0].apply(
                merge(self.fwd_forks[0].apply(embeddings, as_dict=True),
                      {'mask': source_sentence_mask}),
                merge(self.back_forks[0].apply(embeddings, as_dict=True),
                      {'mask': source_sentence_mask}))
        for i in xrange(1, self.n_layers):
            if self.skip_connections:
                inp = tensor.concatenate([representation, embeddings],
                                         axis=2)
            else:
                inp = representation
            representation = self.bidirs[i].apply(
                merge(self.fwd_forks[i].apply(inp, as_dict=True),
                      {'mask': source_sentence_mask}),
                merge(self.back_forks[i].apply(inp, as_dict=True),
                      {'mask': source_sentence_mask})
            )
        return representation, source_sentence_mask


class NoLookupEncoder(Initializable):
    """This is a variation of ``BidirectionalEncoder`` which works with
    sparse feature maps. It does not use a lookup table but directly 
    feeds the predefined distributed representations into the encoder
    network."""

    def __init__(self, embedding_dim, state_dim, **kwargs):
        """Constructor. Note that this implementation only supports
        single layer architectures.
        
        Args:
            embedding_dim (int): Dimensionality of the word vectors
                                 defined by the sparse feature map.
            state_dim (int): Size of the recurrent layer.
        """
        super(NoLookupEncoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')
        self.children = [self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        """Sets the dimensions of the forward and backward forks. """
        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation', 'representation_mask'])
    def apply(self, source_sentence, source_sentence_mask):
        """Creates bidirectional RNN source annotations.
        
        Args:
            source_sentence (Variable): Source sentence with words in
                                        vector representation.
            source_sentence_mask (Variable): Source mask
        
        Returns:
            Variable. source annotations
        """
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(source_sentence, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(source_sentence, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation, source_sentence_mask


class HierarchicalAnnotator(Initializable):
    """This annotator creates higher level annotations by using a 
    network which is similar to the attentional decoder network to
    produce a sequence of new annotations.
    """
    
    def __init__(self,
                 base_encoder, 
                 state_dim=1000, 
                 self_attendable=False, 
                 **kwargs):
        """Constructor.
        
        Args:
            base_encoder (Brick): Low level encoder network which
                                  produces annotations to attend to
            state_dim (int): Size of the recurrent layer.
            self_attendable (bool): If true, the annotator can attend
                                    to its own previous states. If 
                                    false it can only attend to base
                                    annotations
        """
        super(HierarchicalAnnotator, self).__init__(**kwargs)
        self.state_dim = state_dim*2
        self.base_encoder = base_encoder
        self.self_attendable = self_attendable
        trans_core = GatedRecurrent(activation=Tanh(), dim=self.state_dim)
        if self_attendable:
            self.attention = SelfAttendableContentAttention(
                    state_names=trans_core.apply.states,
                    attended_dim=self.state_dim,
                    match_dim=self.state_dim,
                    num_steps=10,
                    name="hier_attention")
        else:
            self.attention = SequenceContentAttention(
                    state_names=trans_core.apply.states,
                    attended_dim=self.state_dim,
                    match_dim=self.state_dim,
                    name="hier_attention")
        self.transition = AttentionRecurrent(trans_core, 
                                             self.attention, 
                                             name="hier_att_trans")
        self.children = [self.transition]

    def _push_allocation_config(self):
        """Sets the dimensions of rnn inputs. """
        self.rnn_inputs = {name: shared_floatx_zeros(
                                            self.transition.get_dim(name))
                              for name in self.transition.apply.sequences 
                              if name != 'mask'}
    
    @application(inputs=['base_annotations', 'base_mask'],
                 outputs=['annotations', 'annotations_mask'])
    def apply(self, base_annotations, base_mask):
        ann_representation = self.transition.apply(
            **merge(self.rnn_inputs, {
                'mask': base_mask,
                'attended': base_annotations,
                'attended_mask': base_mask}))[0]
        return ann_representation, base_mask


class EncoderWithAnnotators(Initializable):
    """This encoder extends the annotations of the standard encoder by
    additional ones. These additional annotations are derived from the
    flat encoder annotations by ``Annotator`` instances.
    """

    def __init__(self, base_encoder, annotators, add_direct, **kwargs):
        """Constructor.
        
        Args:
            base_encoder (Brick): The base encoder which produces the
                                  annotations used by the annotators
            annotators (list): List of annotators
            add_direct (bool): If false, only the annotations from
                               the annotators are used. If true, the
                               original annotations of the base encoder
                               are also passed through
        """
        super(EncoderWithAnnotators, self).__init__(**kwargs)
        self.base_encoder = base_encoder
        self.annotators = annotators
        self.add_direct = add_direct
        try:
            self.bidir = base_encoder
        except AttributeError:
            pass # Its fine, no bidirectional encoder
        self.children = annotators + [base_encoder]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation', 'representation_mask'])
    def apply(self, source_sentence, source_sentence_mask):
        """Creates the final list of annotations.
        
        Args:
            source_sentence (Variable): Source sentence with words in
                                        vector representation.
            source_sentence_mask (Variable): Source mask
        
        Returns:
            Variable. source annotations
        """
        # Time as first dimension
        base_representations,base_mask = self.base_encoder.apply(
                                                          source_sentence,
                                                          source_sentence_mask)
        annotations = []
        masks = []
        if self.add_direct:
            annotations.append(base_representations)
            masks.append(base_mask)
        for annotator in self.annotators:
            ann,mask = annotator.apply(base_representations,
                                       base_mask)
            annotations.append(ann)
            masks.append(mask)
        return tensor.concatenate(annotations), tensor.concatenate(masks)
