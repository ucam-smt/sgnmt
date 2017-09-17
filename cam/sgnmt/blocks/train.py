""""This script starts training an NMT system with blocks. This
follows the NMT blocks 0.1 example except the following points:

- This implementation supports reshuffling between training epochs
- We introduce the --fix_embeddings parameter for fixing word 
  embeddings in later training stages.
- The BleuValidator in the standard blocks implementation evaluates 
  with the </S> symbol. We remove </S> before passing through to the 
  BLEU evaluation script.
- We use reserved indices which are more compatible to the syntactical
  MT system HiFST: 0: UNK/eps, 1: <S>, 2: </S>
- The --bleu_script parameter supports the %s placeholder. This makes
  it possible to use alternative BLEU scripts for model selection, e.g.
  Moses' mteval_v13a.pl.
- Blocks changed the BRICK_DELIMITER variable at some point from '-' to
  '/'. This causes problems when trying to load old model files.
  Therefore, we keep using the '-' character in our model files.
- The NMT implementation in blocks had a bug in creating checkpoint
  files
    https://github.com/mila-udem/blocks-examples/issues/97
    https://github.com/mila-udem/blocks-examples/issues/72
  Therefore, we modified the code similarly to #72 to fix this
- Dropout fix https://github.com/mila-udem/blocks-examples/issues/46

This module contains modified code directly taken from 
blocks-examples/machine_translation.
"""


from blocks.extensions import FinishAfter, Printing, SimpleExtension
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.search import BeamSearch
from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from fuel.transformers import Merge, Batch, Filter, SortMapping, Unpack, Mapping
import logging
import pprint
import signal

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta, Adam,
                               AdaGrad, Scale, CompositeRule)
from blocks.main_loop import MainLoop
from fuel.datasets import TextFile
from fuel.streams import DataStream

from cam.sgnmt import utils
from cam.sgnmt.blocks import stream
from cam.sgnmt.blocks.pruning import PruningGradientDescent
from cam.sgnmt.blocks.checkpoint import CheckpointNMT, LoadNMT
from cam.sgnmt.blocks.model import NMTModel
from cam.sgnmt.blocks.nmt import blocks_get_default_nmt_config
from cam.sgnmt.blocks.sampling import BleuValidator
from cam.sgnmt.blocks.stream import ParallelSourceSwitchDataset, \
                                    ShuffledParallelSource, \
                                    ParallelTextFile, DummyParallelSource, \
                                    BacktranslatedParallelSource, \
                                    MergedParallelSource, \
                                    OldBacktranslatedParallelSource
from cam.sgnmt.misc.sparse import FileBasedFeatMap
from cam.sgnmt.ui import get_blocks_train_parser


try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


def _add_special_ids(vocab):
    """Add fuel/blocks style entries to vocabulary. """
    vocab['<S>'] = utils.GO_ID
    vocab['</S>'] = utils.EOS_ID
    vocab['<UNK>'] = utils.UNK_ID
    return vocab


def _get_sgnmt_tr_stream(data_stream,
                       src_vocab_size=30000, 
                       trg_vocab_size=30000,
                       seq_len=50, 
                       batch_size=80, 
                       sort_k_batches=12, 
                       src_sparse_feat_map='',
                       trg_sparse_feat_map='',
                       **kwargs):
    """Prepares the raw text file stream ``data_stream`` for the Blocks
    main loop. This includes handling UNKs, splitting ino batches, sort
    locally by sequence length, and masking. This roughly corresponds 
    to ``get_sgnmt_tr_stream`` in ``machine_translation/stream`` in the
    blocks examples.
    
    The arguments to this method are given by the configuration dict.
    """

    # Filter sequences that are too long
    s = Filter(data_stream, predicate=stream._too_long(seq_len=seq_len))

    # Replacing out of vocabulary tokens with unk token already
    # handled in the `DataSet`s

    # Build a batched version of stream to read k batches ahead
    s = Batch(s, iteration_scheme=ConstantScheme(batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    s = Mapping(s, SortMapping(stream._length))

    # Convert it into a stream again
    s = Unpack(s)

    # Construct batches from the stream with specified batch size
    s = Batch(s, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = stream.PaddingWithEOS(s, [utils.EOS_ID, utils.EOS_ID])

    return masked_stream


def _get_dataset_with_mono(mono_data_integration='exp3s',
                           backtrans_nmt_config='',
                           backtrans_store=True,
                           add_mono_dummy_data=True,
                           min_parallel_data=0.2,
                           backtrans_reload_frequency=0,
                           backtrans_max_same_word=0.3,
                           src_data='',
                           trg_data='',
                           src_mono_data='',
                           trg_mono_data='',
                           src_vocab_size=30000,
                           trg_vocab_size=30000,
                           src_sparse_feat_map='',
                           trg_sparse_feat_map='',
                           saveto='',
                           **kwargs):
    """Creates a parallel data stream with monolingual data. This is
    based on the ``ParallelSource`` framework in ``stream``.
    
    The arguments to this method are given by the configuration dict.
    """
    src_sens = stream.load_sentences_from_file(src_data, src_vocab_size)
    trg_sens = stream.load_sentences_from_file(trg_data, trg_vocab_size)
    trg_mono_sens = stream.load_sentences_from_file(trg_mono_data,
                                                    trg_vocab_size)
    
    backtrans_config = blocks_get_default_nmt_config()
    if backtrans_nmt_config:
        for pair in backtrans_nmt_config.split(","):
            (k,v) = pair.split("=", 1)
            backtrans_config[k] = type(backtrans_config[k])(v)
    
    parallel_src = ShuffledParallelSource(src_sens, trg_sens)
    dummy_src = None
    if add_mono_dummy_data:
        dummy_src = DummyParallelSource(utils.GO_ID, trg_mono_sens)
    if backtrans_store:
        backtrans_file = "%s/backtrans.txt" % saveto
        old_backtrans_src = OldBacktranslatedParallelSource(backtrans_file)
        backtrans_src = BacktranslatedParallelSource(trg_mono_sens,
                                                     backtrans_config,
                                                     backtrans_file,
                                                     backtrans_max_same_word,
                                                     backtrans_reload_frequency,
                                                     old_backtrans_src)
    else:
        backtrans_src = BacktranslatedParallelSource(trg_mono_sens,
                                                     backtrans_config,
                                                     None,
                                                     backtrans_max_same_word,
                                                     backtrans_reload_frequency)

    if min_parallel_data > 0.0:
        if add_mono_dummy_data:
            dummy_src = MergedParallelSource(parallel_src,
                                             dummy_src,
                                             min_parallel_data)
        backtrans_src = MergedParallelSource(parallel_src,
                                             backtrans_src,
                                             min_parallel_data)
        old_backtrans_src = MergedParallelSource(parallel_src,
                                                 old_backtrans_src,
                                                 min_parallel_data)
    sources = []
    sources.append(parallel_src)
    if add_mono_dummy_data:
        sources.append(dummy_src)
    sources.append(backtrans_src)
    if backtrans_store:
        sources.append(old_backtrans_src)
                        
    return ParallelSourceSwitchDataset(sources,
                                       src_vocab_size,
                                       trg_vocab_size,
                                       src_sparse_feat_map=src_sparse_feat_map, 
                                       trg_sparse_feat_map=trg_sparse_feat_map)


def _get_shuffled_text_stream(src_data,
                              trg_data,
                              src_vocab_size=30000,
                              trg_vocab_size=30000,
                              src_sparse_feat_map='',
                              trg_sparse_feat_map='',
                              **kwargs):
    """Creates a parallel data stream using ``ParallelTextFile``. This
    data set implementation allows random access, so we return a 
    shuffled data stream using the ``ShuffledExampleScheme`` iteration 
    scheme.
    
    The arguments to this method are given by the configuration dict.
    """

    parallel_dataset = ParallelTextFile(src_data,
                                        trg_data,
                                        src_vocab_size,
                                        trg_vocab_size,
                                        src_sparse_feat_map=src_sparse_feat_map, 
                                        trg_sparse_feat_map=trg_sparse_feat_map)
    #iter_scheme = SequentialExampleScheme(parallel_dataset.num_examples)
    iter_scheme = ShuffledExampleScheme(parallel_dataset.num_examples)
    return DataStream(parallel_dataset, iteration_scheme=iter_scheme)


def _get_text_stream(src_data,
                     trg_data,
                     src_vocab_size=30000,
                     trg_vocab_size=30000,
                     **kwargs):
    """Creates a parallel data stream from two text files without 
    random access. This stream cannot be used with reshuffling.
    
    The arguments to this method are given by the configuration dict.
    """

    # Build dummy vocabulary to make TextFile happy
    src_vocab = _add_special_ids({str(i) : i for i in xrange(src_vocab_size)})
    trg_vocab = _add_special_ids({str(i) : i for i in xrange(trg_vocab_size)})

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    return Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))


def _get_sgnmt_dev_stream(val_set=None,
                          src_vocab=None,
                          src_vocab_size=30000,
                          **kwargs):
    """Setup development set stream if necessary.
    
    The arguments to this method are given by the configuration dict.
    """
    dev_stream = None
    if val_set is not None:
        src_vocab = _add_special_ids({str(i) : i 
                                        for i in xrange(src_vocab_size)})
        dev_dataset = TextFile([val_set], src_vocab, None)
        dev_stream = DataStream(dev_dataset)
    return dev_stream


class AlwaysEpochInterrupt(SimpleExtension):
    """Extension which overrides the handle_batch_interrupt routine 
    with the handle_epoch_interrupt."""

    def __init__(self, **kwargs):
        super(AlwaysEpochInterrupt, self).__init__(**kwargs)

    def _handle_interrupt(self, signal_number, frame):
        self.main_loop.log.current_row['epoch_interrupt_received'] = True
        self.main_loop.status['epoch_interrupt_received'] = True
        self.main_loop.log.current_row['batch_interrupt_received'] = True
        self.main_loop.status['batch_interrupt_received'] = True

    def do(self, which_callback, *args):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self.main_loop.original_sigint_handler = self._handle_interrupt
        self.main_loop.original_sigterm_handler = self._handle_interrupt


def main(config,
         tr_stream,
         dev_stream,
         use_bokeh=False, 
         slim_iteration_state=False, 
         switch_controller=None,
         reset_epoch=False):
    """This method largely corresponds to the ``main`` method in the
    original Blocks implementation in blocks-examples and most of the
    code is copied from there. Following modifications have been made:
    
    - Support fixing word embedding during training
    - Dropout fix https://github.com/mila-udem/blocks-examples/issues/46
    - If necessary, add the exp3s extension
    
    Args:
        config (dict): NMT config
        tr_stream (DataStream): Training data stream
        dev_stream (DataStream): Validation data stream
        use_bokeh (bool): Whether to use bokeh for plotting
        slim_iteration_state (bool): Whether to store the full iteration
                                     state or only the epoch iterator
                                     without data stream state
        switch_controller (SourceSwitchController): Controlling strategy
                                                    if monolingual data
                                                    is used as well
        reset_epoch (bool): Set epoch_started in main loop status to
                            false. Sometimes required if you change
                            training parameters such as 
                            mono_data_integration
    """
    
    nmt_model = NMTModel(config)
    nmt_model.set_up(make_prunable = (args.prune_every > 0))

    # Set extensions
    logging.info("Initializing extensions")
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoring([nmt_model.cost], after_batch=True),
        Printing(after_batch=True),
        CheckpointNMT(config['saveto'], 
                      slim_iteration_state, 
                      every_n_batches=config['save_freq'])
    ]

    # Add early stopping based on bleu
    if config['bleu_script'] is not None:
        logging.info("Building bleu validator")
        extensions.append(
            BleuValidator(nmt_model.sampling_input, 
                          samples=nmt_model.samples, 
                          config=config,
                          model=nmt_model.search_model, data_stream=dev_stream,
                          normalize=config['normalized_bleu'],
                          store_full_main_loop=config['store_full_main_loop'],
                          every_n_batches=config['bleu_val_freq']))

    if switch_controller:
        switch_controller.beam_search = BeamSearch(samples=nmt_model.samples)
        switch_controller.src_sentence = nmt_model.sampling_input
        extensions.append(switch_controller)

    # Reload model if necessary
    if config['reload']:
        extensions.append(LoadNMT(config['saveto'], 
                          slim_iteration_state, 
                          reset_epoch))

    # Plot cost in bokeh if necessary
    if use_bokeh and BOKEH_AVAILABLE:
        extensions.append(
            Plot('Decoding cost', channels=[['decoder_cost_cost']],
                 after_batch=True))
    
    # Add an extension for correct handling of SIGTERM and SIGINT
    extensions.append(AlwaysEpochInterrupt(every_n_batches=1))

    # Set up training algorithm
    logging.info("Initializing training algorithm")
    # https://github.com/mila-udem/blocks-examples/issues/46
    train_params = nmt_model.cg.parameters
    # fs439: fix embeddings?
    if config['fix_embeddings']:
        train_params = []
        embedding_params = ['softmax1', 
                            'softmax0', 
                            'maxout_bias', 
                            'embeddings', 
                            'lookuptable', 
                            'transform_feedback']
        for p in nmt_model.cg.parameters:
            add_param = True
            for ann in p.tag.annotations:
                if ann.name in embedding_params:
                    logging.info("Do not train %s: %s" % (p, ann))
                    add_param = False
                    break
            if add_param:
                train_params.append(p)
    # Change cost=cost to cg.outputs[0] ?
    cost_func = nmt_model.cg.outputs[0] if config['dropout'] < 1.0 \
                                        else nmt_model.cost
    if config['step_rule'] in ['AdaGrad', 'Adam']:
        step_rule = eval(config['step_rule'])(learning_rate=args.learning_rate)
    else:
        step_rule = eval(config['step_rule'])()
    step_rule = CompositeRule([StepClipping(config['step_clipping']),
                               step_rule])
    if args.prune_every < 1:
        algorithm = GradientDescent(
            cost=cost_func,
            parameters=train_params,
            step_rule=step_rule)
    else:
        algorithm = PruningGradientDescent(
            prune_layer_configs=args.prune_layers.split(','),
            prune_layout_path=args.prune_layout_path,
            prune_n_steps=args.prune_n_steps,
            prune_every=args.prune_every,
            prune_reset_every=args.prune_reset_every,
            nmt_model=nmt_model,
            cost=cost_func,
            parameters=train_params,
            step_rule=step_rule)

    # Initialize main loop
    logging.info("Initializing main loop")
    main_loop = MainLoop(
        model=nmt_model.training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Reset epoch
    if reset_epoch:
        main_loop.status['epoch_started'] = False

    # Train!
    main_loop.run()


# MAIN CODE STARTS HERE

parser = get_blocks_train_parser()
args = parser.parse_args()

# Get configuration
configuration = blocks_get_default_nmt_config()
for k in dir(args):
    if k in configuration:
        configuration[k] = getattr(args, k)
logging.info("Model options:\n{}".format(pprint.pformat(configuration)))
if configuration['src_sparse_feat_map']:
    configuration['src_sparse_feat_map'] = FileBasedFeatMap(
                                        configuration['enc_embed'],
                                        configuration['src_sparse_feat_map'])
if configuration['trg_sparse_feat_map']:
    configuration['trg_sparse_feat_map'] = FileBasedFeatMap(
                                        configuration['dec_embed'],
                                        configuration['trg_sparse_feat_map'])

# Get data streams and start building the blocks main loop
switch_controller = None
if args.mono_data_integration != 'none':
    logging.fatal("Could not find policy %s" % args.mono_data_integration)
elif args.reshuffle:
    configuration['data_stream'] = _get_shuffled_text_stream(**configuration)
else:
    configuration['data_stream'] = _get_text_stream(**configuration)
    
main(configuration,
     _get_sgnmt_tr_stream(**configuration),
     _get_sgnmt_dev_stream(**configuration),
     args.bokeh,
     args.slim_iteration_state,
     switch_controller,
     args.reset_epoch)
