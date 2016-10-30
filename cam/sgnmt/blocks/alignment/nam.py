"""This module implements the Neural Alignment Model (NAM). The NAM
uses the same architecture as normal NMT, but replaces the attention
model with a trainable alignment matrix. The alignment model reloads
NMT weight matrices trained with ``nmt.train`` except the ones in the
attention sub-network. To align a sentence pair, we initialize the
alignment matrix randomly, and run several iterations of SGD in which
we fix all parameters (i.e. weight matrices loaded from the NMT model)
except the newly added alignment matrix.

We implemented this workflow using the MainLoop framework in blocks.
The data stream emits the same sentence pair until it is explicitly
requested by ``NextSentenceExtension`` to switch to the next pair.
"""

from blocks.extensions import SimpleExtension, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from fuel.schemes import ConstantScheme
from fuel.transformers import Merge, Batch, Filter, Transformer
import logging

from blocks.algorithms import GradientDescent
from blocks.main_loop import MainLoop
from fuel.datasets import TextFile


from cam.sgnmt import utils
from cam.sgnmt.blocks import stream
from cam.sgnmt.blocks.model import NMTModel, LoadNMTUtils
from cam.sgnmt.blocks.nmt import get_nmt_model_path

import numpy as np

alignments = []
"""Global variable for collecting the alignments. """


class NextSentenceExtension(SimpleExtension):
    """This Blocks extension is invoked each ``--iterations`` batches.
    It saves the previous alignment matrix, resets the matrix in the
    computation graph, and iterates to the next sentence pair.
    """

    def __init__(self, align_stream, **kwargs):
        """Constructor of ``NextSentenceExtension``.
        
        Args:
            align_stream (ExplicitNext): DataStream which supports
                                         explicitly switching to the
                                         next sentence pair
        """
        super(NextSentenceExtension, self).__init__(**kwargs)
        self.init_matrix = None
        self.align_stream = align_stream

    def _initialize(self):
        """Fetches the alignment matrix parameter and stores the
        initial matrix
        """
        params = self.main_loop.model.get_parameter_dict()
        for name in params:
            if 'alignment_matrix' in name:
                self.align_matrix = params[name]
                self.init_matrix = self.align_matrix.get_value()
                return
        logging.fatal("Could not find alignment matrix parameter!")
        
    def do(self, which_callback, *args):
        """This method is called from the MainLoop in blocks and 
        handles all the work which has to be done after aligning a
        sentence pair. This involves:
        
          - Fetching the alignment matrix from the computation graph
            and append it to the ``alignments`` global
          - Call the data stream s.t. it iterates to the next sentence
            pair
          - Override the current value of the alignment matrix variable
            with its initial value to get a fresh start for the next
            optimization
        """
        global alignments
        if which_callback == "before_training":
            self._initialize()
            return
        data = self.align_stream.get_data()
        src_sen = data[0][0]
        trg_sen = data[2][0]
        # -1 to drop EOS
        al = np.exp(self.align_matrix.get_value()[:len(src_sen)-1,
                                                  :len(trg_sen)-1])
        al_norm = al / al.sum(axis=0)
        alignments.append(al_norm)
        self.align_stream.switch_to_next_data()
        self.align_matrix.set_value(self.init_matrix)


class PrintCurrentLogRow(SimpleExtension):
    """This extension prints the current log row to the screen."""

    def __init__(self, **kwargs):
        super(PrintCurrentLogRow, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in attribute_tuples.iteritems():
            if not attr.startswith("_"):
                print("%s: %s" % (attr, value))

    def do(self, which_callback, *args):
        """Prints the current log status on the screen """
        log = self.main_loop.log
        print("Log records from the iteration {}:".format(
                log.status['iterations_done']))
        self._print_attributes(log.current_row)


class ExplicitNext(Transformer):
    """This data transformer outputs the same data until it is
    explicitly iterated by calling ``switch_to_next_data()``
    """
    def __init__(self, data_stream, **kwargs):
        super(ExplicitNext, self).__init__(data_stream, **kwargs)
        self.current_data = None

    def get_data(self, request=None):
        """Returns ``self.current_data``. """
        if not self.current_data:
            self.switch_to_next_data()
        return self.current_data
    
    def switch_to_next_data(self):
        """Switch to the next data given by the child data stream """
        self.current_data = next(self.child_epoch_iterator)
        logging.info("Align sentence pair %s <-> %s" % (
            ' '.join([str(w) for w in self.current_data[0][0]]),
            ' '.join([str(w) for w in self.current_data[2][0]])))
    

def _add_special_ids(vocab):
    """Add fuel/blocks style entries to vocabulary. """
    vocab['<S>'] = utils.GO_ID
    vocab['</S>'] = utils.EOS_ID
    vocab['<UNK>'] = utils.UNK_ID
    return vocab


def _get_align_stream(src_data, 
                      trg_data, 
                      src_vocab_size, 
                      trg_vocab_size, 
                      seq_len, 
                      **kwargs):
    """Creates the stream which is used for the main loop.
    
    Args:
        src_data (string): Path to the source sentences
        trg_data (string): Path to the target sentences
        src_vocab_size (int): Size of the source vocabulary in the NMT
                              model
        trg_vocab_size (int): Size of the target vocabulary in the NMT
                              model
        seq_len (int): Maximum length of any source or target sentence
    
    Returns:
        ExplicitNext. Alignment data stream which can be iterated
        explicitly
    """
    # Build dummy vocabulary to make TextFile happy
    src_vocab = _add_special_ids({str(i) : i for i in xrange(src_vocab_size)})
    trg_vocab = _add_special_ids({str(i) : i for i in xrange(trg_vocab_size)})
    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)
    # Merge them to get a source, target pair
    s = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))
    s = Filter(s, predicate=stream._too_long(seq_len=seq_len))
    s = Batch(s, iteration_scheme=ConstantScheme(1))
    masked_stream = stream.PaddingWithEOS(s, [utils.EOS_ID, utils.EOS_ID])
    return ExplicitNext(masked_stream)


def align_with_nam(config, args):
    """Main method for using the Neural Alignment Model.
    
    Args:
        config (dict): NMT configuration
        args (object): ArgumentParser object containing the command
                       line arguments
    
    Returns:
        list. List of alignments, where alignments are represented as
        numpy matrices containing confidences between 0 and 1.
    """
    global alignments
    config['attention'] = 'parameterized'
    alignments = []
    nmt_model = NMTModel(config)
    nmt_model.set_up()
    align_stream = _get_align_stream(**config)
    extensions = [
        FinishAfter(after_epoch=True),
        TrainingDataMonitoring([nmt_model.cost], after_batch=True),
        PrintCurrentLogRow(after_batch=True),
        NextSentenceExtension(align_stream=align_stream,
                              every_n_batches=args.iterations,
                              before_training=True)
    ]
    train_params = []
    for p in nmt_model.cg.parameters:
        if p.name in 'alignment_matrix':
            train_params.append(p)
            break
    algorithm = GradientDescent(
        cost=nmt_model.cost,
        parameters=train_params
    )
    main_loop = MainLoop(
        model=nmt_model.training_model,
        algorithm=algorithm,
        data_stream=align_stream,
        extensions=extensions
    )
    nmt_model_path = get_nmt_model_path(args.nmt_model_selector, config)
    loader = LoadNMTUtils(nmt_model_path,
                          config['saveto'],
                          nmt_model.training_model)
    loader.load_weights()
    try:
        main_loop.run()
    except StopIteration:
        logging.info("Alignment finished")
    return alignments
