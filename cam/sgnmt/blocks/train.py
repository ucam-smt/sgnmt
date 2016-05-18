""""This script starts training an NMT system with blocks. This largely
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

import sys

import argparse
import logging
import pprint

from machine_translation import configurations,stream
from fuel.datasets import TextFile,Dataset
from fuel.transformers import Merge, Batch, Filter, SortMapping, Unpack, Mapping
from fuel.schemes import ConstantScheme,ShuffledExampleScheme
from fuel.streams import DataStream
from machine_translation.org__init__ import main
from cam.sgnmt import utils
from cam.sgnmt.blocks.ui import get_train_parser

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


class ParallelTextFile(Dataset):
    """This is like TextFile in fuel supports random access. This makes
    it possible to use it in combination with ``ShuffledExampleScheme``
    in fuel.
    """
    provides_sources = ('source','target')
    example_iteration_scheme = None

    def __init__(self, src_file, trgt_file, src_dictionary, trgt_dictionary,
                 bos_token='<S>', eos_token='</S>', unk_token='<UNK>',
                 level='word', preprocess=None):
        """Constructor like for ``TextFile``"""
        self.src_file = src_file
        self.trgt_file = trgt_file
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        for dictionary in [src_dictionary, trgt_dictionary]:
            if bos_token is not None and bos_token not in dictionary:
                raise ValueError
            if eos_token is not None and eos_token not in dictionary:
                raise ValueError
            if unk_token not in dictionary:
                raise ValueError
        if level not in ('word', 'character'):
            raise ValueError
        self.src_dictionary = src_dictionary
        self.trgt_dictionary = trgt_dictionary
        self.level = level
        self.preprocess = preprocess
        with open(self.src_file) as f:
            self.src_sentences = f.readlines()
        with open(self.trgt_file) as f:
            self.trgt_sentences = f.readlines()
        self.num_examples = len(self.src_sentences)
        if self.num_examples != len(self.trgt_sentences):
            raise ValueError
        super(ParallelTextFile, self).__init__()

    def open(self):
        """Dummy implementation as opening is done in constructor """
        return None

    def get_data(self, state=None, request=None):
        """Returns a data entry, which is a pair (tuple) of source and
        target sentence. Similarly to ``TextFile`` the sentences are
        lists of word ids.
        
        Args:
            state (None):  not used
            requres (int): Index of the entry to load
        
        Returns:
            2-tuple of source and target sentence at given position
        """
        return (self._process_sentence(self.src_sentences[request],
                                       self.src_dictionary), 
                self._process_sentence(self.trgt_sentences[request],
                                       self.trgt_dictionary))

    def _process_sentence(self, sentence, dictionary):
        """Prepares string representation of sentence for passing 
        down the data stream"""
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data.extend(dictionary.get(word,
                                            dictionary[self.unk_token])
                        for word in sentence.split())
        else:
            data.extend(dictionary.get(char,
                                            dictionary[self.unk_token])
                        for char in sentence.strip())
        if self.eos_token:
            data.append(dictionary[self.eos_token])
        #return (data,)
        return data


def add_special_ids(vocab):
    """Add fuel/blocks style entries to vocabulary. """
    vocab['<S>'] = utils.GO_ID
    vocab['</S>'] = utils.EOS_ID
    vocab['<UNK>'] = utils.UNK_ID
    return vocab


def get_sgnmt_tr_stream(src_data, trg_data,
                       src_vocab_size=30000, trg_vocab_size=30000,
                       unk_id=1, seq_len=50, batch_size=80, 
                       sort_k_batches=12, **kwargs):
    """Prepares the unshuffled training data stream. This corresponds 
    to ``get_sgnmt_tr_stream`` in ``machine_translation/stream`` in the
    blocks examples."""

    # Build dummy vocabulary to make TextFile happy
    src_vocab = add_special_ids({str(i) : i for i in xrange(src_vocab_size)})
    trg_vocab = add_special_ids({str(i) : i for i in xrange(trg_vocab_size)})

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    s = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    s = Filter(s, predicate=stream._too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    s = Mapping(s, stream._oov_to_unk(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               unk_id=utils.UNK_ID))

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


def get_sgnmt_shuffled_tr_stream(src_data, trg_data,
                                src_vocab_size=30000, trg_vocab_size=30000,
                                unk_id=1, seq_len=50, batch_size=80, 
                                sort_k_batches=12, **kwargs):
    """Prepares the shuffled training data stream. This is similar to 
    ``get_sgnmt_tr_stream`` but uses ``ParallelTextFile`` in combination
    with ``ShuffledExampleScheme`` to support reshuffling."""

    # Build dummy vocabulary to make TextFile happy
    src_vocab = add_special_ids({str(i) : i for i in xrange(src_vocab_size)})
    trg_vocab = add_special_ids({str(i) : i for i in xrange(trg_vocab_size)})

    parallel_dataset = ParallelTextFile(src_data, trg_data,
                                        src_vocab, trg_vocab, None)
    #iter_scheme = SequentialExampleScheme(parallel_dataset.num_examples)
    iter_scheme = ShuffledExampleScheme(parallel_dataset.num_examples)
    s = DataStream(parallel_dataset, iteration_scheme=iter_scheme)

    # Filter sequences that are too long
    s = Filter(s, predicate=stream._too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    s = Mapping(s, stream._oov_to_unk(src_vocab_size=src_vocab_size,
                               trg_vocab_size=trg_vocab_size,
                               unk_id=utils.UNK_ID))

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


def get_sgnmt_dev_stream(val_set=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set is not None:
        src_vocab = add_special_ids({str(i) : i 
                                        for i in xrange(src_vocab_size)})
        dev_dataset = TextFile([val_set], src_vocab, None)
        dev_stream = DataStream(dev_dataset)
    return dev_stream


parser = get_train_parser()
args = parser.parse_args()

# Get configuration
configuration = getattr(configurations, args.proto)()
for k in dir(args):
    if k in configuration:
        configuration[k] = getattr(args, k)
configuration['unk_id'] = utils.UNK_ID
logger.info("Model options:\n{}".format(pprint.pformat(configuration)))

# Get data streams and start building the blocks main loop 
if args.reshuffle:
    tr_stream = get_sgnmt_shuffled_tr_stream(**configuration)
else:
    tr_stream = get_sgnmt_tr_stream(**configuration)
main(configuration,
     tr_stream,
     get_sgnmt_dev_stream(**configuration),
     args.bokeh)
