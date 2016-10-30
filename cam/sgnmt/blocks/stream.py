"""This module acts as an replacement of the original ``stream`` module
in the Blocks NMT implementation. Some methods are copied over from the 
Blocks code.

Additionally, this module contains more advanced data sources such as
``ParallelTextFile`` for reading a parallel corpus with random access, 
and the ``ParallelSourceSwitchDataset`` for integrating reinforcement 
learning methods into the training process.
"""

from __future__ import print_function

from abc import abstractmethod
from blocks.extensions import SimpleExtension
import datetime
from fuel.transformers import Padding
import logging
import math
import numpy
import os
import random
import re
from subprocess import Popen, PIPE
import sys
import time

from fuel.datasets import Dataset

from cam.sgnmt import utils
from cam.sgnmt.blocks.nmt import get_nmt_model_path_params
from cam.sgnmt.blocks.vanilla_decoder import BlocksNMTVanillaDecoder
from cam.sgnmt.misc.sparse import FlatSparseFeatMap


class ParallelSource(object):
    """Interface for sources which can be fed into 
    ``ParallelSourceSwitchDataset``. These sources always represent an
    indexed data set of sentence pairs. This can be either real 
    parallel training data or synthesized data were only one side is
    original text and the other side is artificial (e.g. a dummy token
    or back-translated sentences).
    """
    
    def next(self):
        """Get the next sentence pair.
        
        Returns:
            tuple. First element is the source sentence, the second is
                   the target sentence (list of integers) without <S>
                   but with </S>
        """
        raise NotImplementedError


class ShuffledParallelSource(ParallelSource):
    """Simplest ``ParallelSource`` implementation which allows access
    to parallel data in a random order.
    """
    
    def __init__(self, src_sentences, trg_sentences):
        """Creates a new data source which represents the parallel
        text ``src_sentences`` - ``trg_sentences``.
        
        Args:
            src_sentences (list): list of source language sentences
            trg_sentences (list): list of target language sentences
        
        Raises:
            ValueError. If both arguments do not have the same length.
        """
        if len(src_sentences) != len(trg_sentences):
            raise ValueError
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.shuffler = Reshuffler(0, len(trg_sentences))
    
    def next(self):
        """Emits the sentence pairs in random order.
        
        Returns:
            tuple. Next sentence pair
        """
        idx = self.shuffler.next()
        return (self.src_sentences[idx], self.trg_sentences[idx])


class DummyParallelSource(ParallelSource):
    """Represents monolingual data and uses a dummy token on the
    source side to fill in the missing gap.
    """
    
    def __init__(self, src_token, trg_sentences):
        """Creates a new data source which is based on monolingual
        target data and uses a dummy token on the source side.
        
        Args:
            trg_sentences (list): list of target language sentences
        """
        self.src_sentence = [src_token] + [utils.EOS_ID]
        self.trg_sentences = trg_sentences
        self.shuffler = Reshuffler(0, len(trg_sentences))
    
    def next(self):
        """Emits the target sentences in random order with a dummy
        token plus </S> on the source sentence
        
        Returns:
            tuple. Next dummy token - target sentence pair
        """
        return (self.src_sentence, self.trg_sentences[self.shuffler.next()])


_backtrans_decoder = None
"""Decoder for backtranslation (needs to be global, otherwise
the main loop iteration state cannot be pickled
"""

class BacktranslatedParallelSource(ParallelSource):
    """This data source is based on monolingual target data. The source
    sentences are translated from the target sentence like described by
    Senrich et al., 2015.
    """
    
    def __init__(self, 
                 trg_sentences, 
                 nmt_config, 
                 store_trans=None, 
                 max_same_word=0.3,
                 reload_frequency=0,
                 old_backtrans_src=None):
        """Creates a new back translating data source.
        
        Args:
            trg_sentences (list): list of target language sentences
            nmt_config (dict): NMT configuration of the back-
                               translating NMT system
            store_trans (string): Write the back-translated sentences
                                  to the file system (append to that 
                                  file)
            max_same_word (float): Used for sanity check of the back
                                   translation. If the most frequent
                                   word in the backtranslated sentence
                                   has relative frequency higher than
                                   this, discard this sentence pair
            reload_frequency (int): The back-translating NMT model is
                                    reloaded every n updates. This is 
                                    useful if the back-translating NMT
                                    system is currently trained by 
                                    itself with the same policy. This
                                    enables us to train two NMT systems
                                    in opposite translation directions
                                    and benefit from gains in the other
                                    system immediately. Set to 0 to
                                    disable reloading
            old_backtrans_src (OldBacktranslatedParallelSource):
                        Instance of ``OldBacktranslatedParallelSource``
                        to send the backtranslated sentences to
        """
        self.trg_sentences = trg_sentences
        self.nmt_config = nmt_config
        self.seq_len = nmt_config['seq_len']
        self.log_file = store_trans
        self.max_same_word = max_same_word
        self.reload_frequency = reload_frequency
        self.old_backtrans_src = old_backtrans_src
        self.shuffler = Reshuffler(0, len(trg_sentences))
        self.get_count = 0
        self._load_nmt()
    
    def next(self):
        """Emits the target sentences in random order with the
        backtranslated source sentence.
        
        Returns:
            tuple. synthetic source - target sentence pair
        """
        if self.reload_frequency > 0:
            self.get_count += 1
            if self.get_count % self.reload_frequency == 0:
                self._load_nmt()
        idx = self.shuffler.next()
        trg_sen = self.trg_sentences[idx]
        # TODO: Should be in conf
        if len(trg_sen) > self.seq_len:
            return self.next()
        src_sen = self.backtranslate(trg_sen)
        # Sanity check
        counts = {}
        for w in src_sen:
            counts[w] = counts.get(w,0) + 1
        max_count = max(counts.itervalues())
        if max_count > self.max_same_word * len(src_sen):
            logging.info("Discard back translation %s (max count: %d)" % (
                      src_sen,
                      max_count))
            return self.next() 
        # Write to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("%d ||| %s ||| %s ||| %s\n" % (
                                     idx,
                                     datetime.datetime.now(),
                                     ' '.join([str(w) for w in src_sen]),
                                     ' '.join([str(w) for w in trg_sen])))
        # Send to old backtranslated data source
        if self.old_backtrans_src:
            self.old_backtrans_src.add(idx, src_sen, trg_sen)
        return (src_sen, trg_sen)
    
    def backtranslate(self, trg_sentence):
        """Translates a sentence from the target language back into the
        source language.
        
        Args:
            trg_sentence (list): Target sentence ending with </S>
        
        Returns:
            list. Source sentence ending with </S>
        """
        global _backtrans_decoder
        hypos = _backtrans_decoder.decode(trg_sentence[0:-1]) # Without EOS
        if not hypos: # No translation found, return dummy token
            logging.warn("No back-translation found for %s" % trg_sentence) 
            return [utils.GO_ID, utils.EOS_ID]
        if self.nmt_config['normalized_bleu']: # Length normalization
            for hypo in hypos:
                hypo.total_score /= float(len(hypo.trgt_sentence))
            hypos.sort(key=lambda hypo: hypo.total_score, reverse=True)
        s = hypos[0].trgt_sentence
        return s if s and s[-1] == utils.EOS_ID else (s + [utils.EOS_ID])        
        
    def _load_nmt(self):
        """Loads the back-translating NMT model. """
        global _backtrans_decoder
        _backtrans_decoder = BlocksNMTVanillaDecoder(
                                get_nmt_model_path_params(self.nmt_config),
                                self.nmt_config)


class OldBacktranslatedParallelSource(ParallelSource):
    """This ``ParallelSource`` implementation allows access to 
    sentences which have been translated before. It reads a log file
    created by ``BacktranslatedParallelSource`` and yields sentence
    pairs from that file. The backtranslating parallel source can add 
    new sentence pairs with ``add`` which have not been logged in the
    file yet.
    """
    
    def __init__(self, log_file):
        """Creates a new data source which reads out sentence pairs 
        from a backtranslation log file
        
        Args:
            log_file (string): Path to the log file
        """
        self.src_sentences = {}
        self.trg_sentences = {}
        if os.path.isfile(log_file):
            try:
                with open(log_file) as f:
                    for line in f:
                        parts = line.split("|||")
                        if len(parts) == 4:
                            idx = int(parts[0])
                            self.src_sentences[idx] = [int(w) 
                                            for w in parts[2].strip().split()]
                            self.trg_sentences[idx] = [int(w) 
                                            for w in parts[3].strip().split()]
            except Exception as e:
                logging.warn("An %s error has occurred while reading %s: %s" 
                             % (sys.exc_info()[0], log_file, e))
        self.shuffler = Reshuffler([k for k in self.src_sentences.iterkeys()])
    
    def add(self, idx, src_sen, trg_sen):
        """Adds a new backtranslated sentence to this source. If a
        sentence pair with the same index already exists, override it.
        
        Args:
            idx (int): Index of the sentence pair
            src_sen (list): Source sentence with EOS
            trg_sen (list): Target sentence with EOS
        """
        if not idx in self.src_sentences:
            self.shuffler.l.append(idx)
        self.src_sentences[idx] = src_sen
        self.trg_sentences[idx] = trg_sen
    
    def next(self):
        """Emits the sentence pairs in random order.
        
        Returns:
            tuple. Next sentence pair
        """
        idx = self.shuffler.next()
        return (self.src_sentences[idx], self.trg_sentences[idx])


class MergedParallelSource(ParallelSource):
    """Each time ``next`` is called, we randomly select one of two 
    parallel sources and pass through the ``next`` request to the 
    selected one.
    """
    
    def __init__(self, src1, src2, src1prob):
        """Creates a new data source which merges two other sources
        by randomly selecting one of them each time ``next`` is called.
        
        Args:
            src1 (ParallelSource): First parallel source
            src2 (ParallelSource): Second parallel source
            src1prob (float): Probability of source 1
        """
        self.src1 = src1
        self.src2 = src2
        self.src1prob = src1prob
    
    def next(self):
        """Emits the next sentence pair of one of the sources.
        
        Returns:
            tuple. Next sentence pair
        """
        src = self.src1 if random.random() < self.src1prob else self.src2
        return src.next() 


_current_src_sparse_feat_map = None
_current_trg_sparse_feat_map = None
_current_parallel_sources = []
_current_controller = None
"""Pickling makes a lot of things ugly. This is a great example. The
sources in ``ParallelSourceSwitchDataset`` store the training data,
and adding them to iteration_state leads to very large files and long
time to store them during training. Its much better to reload them from
the file system each time. ``current_parallel_sources`` stores the
reloaded sources. We hack ``__setstate__`` in 
``ParallelSourceSwitchDataset`` to use this global variable.
""" 


class ParallelSourceSwitchDataset(Dataset):
    """This bridges the gap between Fuel and the ``ParallelSource``
    implementations in this module. This is a Fuel ``Dataset`` which
    uses a set of ``ParallelSource`` instances to produce a stream of
    sentence pairs. Only one of the parallel sources is active at a
    time. The active source can be changed with the 
    ``set_active_source`` method. This is particular useful in
    combination with reinforcement learning policies where switching
    the source corresponds to the action the agent takes.
    """
    
    provides_sources = ('source','target')
    example_iteration_scheme = None
    
    def __init__(self, parallel_sources,
                 src_vocab_size=30000,
                 trg_vocab_size=30000,
                 src_sparse_feat_map = None, 
                 trg_sparse_feat_map = None):
        """Creates a new ``Dataset`` which allows switching between 
        multiple parallel sources. The first source is activated per
        default
        
        Args:
            parallel_sources (list): List of ``ParallelSource`` objects
            src_vocab_size (int): Size of source vocabulary
            trg_vocab_size (int): Size of target vocabulary
            src_sparse_feat_map (SparseFeatMap): Map between words and
                                                 their features if you 
                                                 use sparse vectors.
            trg_sparse_feat_map (SparseFeatMap): Map between words and
                                                 their features if you 
                                                 use sparse vectors.
        
        Raises:
            IndexError. If the list is empty
        """
        super(ParallelSourceSwitchDataset, self).__init__()
        global _current_parallel_sources
        global _current_src_sparse_feat_map, _current_trg_sparse_feat_map
        if not parallel_sources:
            raise ValueError
        self.parallel_sources = parallel_sources
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.active_idx = 0
        self.src_sparse_feat_map = src_sparse_feat_map if src_sparse_feat_map \
                                                       else FlatSparseFeatMap()
        self.trg_sparse_feat_map = trg_sparse_feat_map if trg_sparse_feat_map \
                                                       else FlatSparseFeatMap()
        # pickle hacks
        _current_parallel_sources = parallel_sources
        _current_src_sparse_feat_map = self.src_sparse_feat_map
        _current_trg_sparse_feat_map = self.trg_sparse_feat_map 
    
    def set_active_idx(self, idx):
        """Activates a new parallel source.
        
        Args:
            idx (int): index of the new data source
        
        Raises:
            IndexError: if ``idx`` is too large
        """
        self.active_idx = idx
    
    def open(self):
        """Dummy implementation """
        return None

    def get_data(self, state=None, request=None):
        """Get next data entry from ``active_source``, ignores args."""
        if request is not None:
            raise ValueError
        (s,t) = self.parallel_sources[self.active_idx].next()
        return (self.src_sparse_feat_map.words2dense(
                                    utils.oov_to_unk(s, self.src_vocab_size)),
                self.trg_sparse_feat_map.words2dense(
                                    utils.oov_to_unk(t, self.trg_vocab_size)))
    
    def __getstate__(self):
        """Return state values to be pickled."""
        d = dict(self.__dict__)
        del d['parallel_sources']
        del d['src_sparse_feat_map']
        del d['trg_sparse_feat_map']
        d['controller'] = _current_controller.__getstate__()
        return d

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        global _current_controller
        if 'controller' in state:
            _current_controller.__setstate__(state['controller'])
            del state['controller']
        self.__dict__.update(state)
        self.parallel_sources = _current_parallel_sources
        self.src_sparse_feat_map = _current_src_sparse_feat_map
        self.trg_sparse_feat_map = _current_trg_sparse_feat_map
        _current_controller.switch_dataset = self


SWITCH_CONTROLLER_VAL_FILE = "switch_controller.val.txt"
"""File name for the reference sentences for switch controllers """ 


class SourceSwitchController(SimpleExtension):
    """This is a super class for controlling algorithms for 
    ``ParallelSourceSwitchDataset``. This is hooked into Blocks main
    loop as extension. An example of a controller is the EXP3S
    algorithm. The attributes are similar to the ones used in the
    original ``BleuValidator`` implementation in machine_translation.
    sampling. 
    
    Attributes:
        src_sentence (theano.variable): The Theano variable representing
                                        the source sentence in the
                                        computational graph
        beam_search (BeamSearch): Blocks beam search instance
    """
    
    def __init__(self, nmt_config, switch_dataset, part_val_set_size, **kwargs):
        """Connects a new instance with the switch dataset. Note that
        the attributes ``src_sentence`` and ``beam_search`` need to be
        set separately.
        
        Args:
            nmt_config (dict): NMT configuration array
            switch_dataset (ParallelSourceSwitchDataset): The dataset
                                                          to control
            part_val_set_size (int): Number of sentences to decode for
                                     reward calculation
            every_n_batches (int): Make a new decision every n batches
        """
        super(SourceSwitchController, self).__init__(**kwargs)
        global _current_controller
        self.switch_dataset = switch_dataset
        _current_controller = self # Pickle hack
        self.switch_dataset.controller = self
        self.n_actions = len(switch_dataset.parallel_sources)
        self.src_sentence = None
        self.beam_search = None
        self.cur_sen_idxs = []
        self.normalize = nmt_config['normalized_bleu']
        self.beam_size = nmt_config['beam_size']
        self.burn_in = nmt_config['val_burn_in']
        self.val_src = load_sentences_from_file(nmt_config['val_set'], 
                                                nmt_config['src_vocab_size'])
        self.shuffler = Reshuffler(0, len(self.val_src))
        self.part_val_set_size = part_val_set_size
        with open(nmt_config['val_set_grndtruth'], 'r') as f:
            # Do not use load_sentences_from_file for ground truth because we
            # need strings and do not want to replace UNK tokens
            self.val_trg = f.readlines()
        self.tmp_val_file = "%s/%s" % (nmt_config['saveto'],
                                       SWITCH_CONTROLLER_VAL_FILE)
        self.multibleu_cmd = (nmt_config['bleu_script'] % 
                                        self.tmp_val_file).split()

    def do(self, which_callback, *args):
        """This method is called every n batches. If we have already
        seen ``burn_in`` batches, decode part of the dev set and change
        the data source switch if necessary.
        """
        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.burn_in:
            return
        
        if self.cur_sen_idxs: # Calculate reward and process it
            # +0.01 avoids division by zero
            sec = (datetime.datetime.now()-self.prev_time).seconds + 0.01
            bleu = self._calculate_bleu()
            logging.info("Absolute BLEU gain since last action: %f" % (
                                                        bleu - self.prev_bleu))
            self.process_reward(self.prev_bleu, bleu, sec)
        
        # Select new sentences for next comparison
        self.cur_sen_idxs = [self.shuffler.next() 
                                    for _ in xrange(self.part_val_set_size)]
        self._create_tmp_val_file()
        self.prev_bleu = self._calculate_bleu()
        self.prev_time = datetime.datetime.now()  
        
    @abstractmethod
    def process_reward(self, old_bleu, new_bleu, sec):
        """This is a GoF template method. Subclasses implement their
        controlling strategy in this method. Actions can be realized
        by using ``self.switch_dataset.set_active_idx``.
        
        Args:
            old_bleu (float): Achieved bleu before the previous action
            new_bleu (float): Achieved bleu after the previous action
            sec (float): Time in seconds required for the last action
        """
        raise NotImplementedError
    
    def _create_tmp_val_file(self):
        """Creates the ground truth file from the current sentence
        indices ``self.cur_sen_idxs``. After that, ``_calculate_bleu``
        can be called.
        """
        with open(self.tmp_val_file, "w") as f:
            for idx in self.cur_sen_idxs:
                f.write(self.val_trg[idx])
    
    def _calculate_bleu(self):
        """Decodes the current sentences and calculates the BLEU.
        Assumes that the file ``tmp_val_file`` is pointing at has been
        created already with ``_create_tmp_val_file``.
        
        Returns:
            float. Current BLEU score
        """
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        for idx in self.cur_sen_idxs:
            seq = self.val_src[idx]
            input_ = numpy.tile(seq, (self.beam_size, 1))
            trans, costs = self.beam_search.search(
                    input_values={self.src_sentence: input_},
                    max_length=3*len(seq), eol_symbol=utils.EOS_ID,
                    ignore_first_eol=True)
            if self.normalize: # normalize costs according to lengths
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths
            best_idx = numpy.argmax(costs)
            trans = trans[best_idx]
            if trans and trans[-1] == utils.EOS_ID:
                trans = trans[:-1]
            trans_out = ' '.join([str(w) for w in trans])
            print(trans_out, file=mb_subprocess.stdin)
            mb_subprocess.stdin.flush()
        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        mb_subprocess.terminate()
        try:
            out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
            # extract the score
            bleu_score = float(out_parse.group()[6:])
            logging.info("Decoded partial dev set: bleu=%f time=%f"
                         % (bleu_score, time.time() - val_start_time))
        except:
            logging.info("Partial BLEU evaluation failed")
        return bleu_score




def load_sentences_from_file(path, vocab_size):
    """Loads sentences from a plain text file. For each sentence we add
    </S> (but not <S>) as expected by the data stream pipeline. Tokens 
    larger than ``vocab_size`` are replaced by the UNK id.
     
    Args:
        path(string): Path to the text file
        vocab_size(int): Vocabulary size (all tokens larger than
                         this are replaced by ``utils.UNK_ID``
    
    Returns:
        list. List of list of integers representing the indexed
              sentences in the input file.
        
    Raises:
        IOError. If the file could not be read
        ValueError. If the text file contains non-integer tokens
    """
    sens = []
    with open(path) as f:
        for line in f:
            sens.append([int(w) if int(w) < vocab_size else utils.UNK_ID 
                            for w in line.strip().split()] + [utils.EOS_ID])
    return sens
        
    
class Reshuffler(object):
    """This class can be used for reshuffling. Given a range of 
    integers, it shuffles them and yields the shuffled numbers one by
    one. After having shown all numbers once, shuffle again and iterate
    through the new list. This assures complete coverage of the data.
    """
    
    def __init__(self, from_idx, to_idx=None):
        """Create a new instance. Range boundaries are similar to
        ``xrange``. If the first and single argument is a list, it
        provides the allowed indices directly
        
        Args:
            from_idx (int or list): Lower bound of integer range 
                                    (inclusive). If this is a list
                                    and the second argument is not
                                    provided, this lists all allowed
                                    indices
            to_idx (int): Upper bound of integer rang (exclusive)
        
        Raises:
            ValueError. If ``to_idx`` is smaller or equal ``from_idx``.
        """
        if to_idx:
            self.l = [i for i in xrange(from_idx, to_idx)]
            if not self.l:
                raise ValueError
        else:
            self.l = from_idx # from_idx is list
        self.pos = len(self.l)
    
    def next(self):
        """Get an integer within the range specified in the constructor.
        Numbers will be generated in a random order, but this method
        guarantees that all numbers will be visited by this method.
        
        Returns:
            int. A number in the given range.
        """
        self.pos += 1
        if self.pos >= len(self.l):
            random.shuffle(self.l)
            self.pos = 0
        return self.l[self.pos]


class ParallelTextFile(Dataset):
    """This ``Dataset`` implementation is similar to ``TextFile`` in 
    Fuel but supports random access. This makes it possible to use it
    in combination with ``ShuffledExampleScheme`` in fuel. Another
    difference is that it directly represents a database of two 
    text files, i.e. the resulting string does not need to be merged
    with another one.
    """
    
    provides_sources = ('source','target')
    example_iteration_scheme = None

    def __init__(self, 
                 src_file, 
                 trgt_file, 
                 src_vocab_size, 
                 trgt_vocab_size,
                 preprocess=None, 
                 src_sparse_feat_map = None, 
                 trg_sparse_feat_map = None):
        """Constructor like for ``TextFile``"""
        global _current_src_sparse_feat_map, _current_trg_sparse_feat_map
        super(ParallelTextFile, self).__init__()
        self.src_file = src_file
        self.trgt_file = trgt_file
        self.src_vocab_size = src_vocab_size
        self.trgt_vocab_size = trgt_vocab_size
        self.preprocess = preprocess
        with open(self.src_file) as f:
            self.src_sentences = f.readlines()
        with open(self.trgt_file) as f:
            self.trgt_sentences = f.readlines()
        self.num_examples = len(self.src_sentences)
        if self.num_examples != len(self.trgt_sentences):
            raise ValueError
        self.src_sparse_feat_map = src_sparse_feat_map if src_sparse_feat_map \
                                                       else FlatSparseFeatMap()
        self.trg_sparse_feat_map = trg_sparse_feat_map if trg_sparse_feat_map \
                                                       else FlatSparseFeatMap()
        # Pickle hacks
        _current_src_sparse_feat_map = self.src_sparse_feat_map
        _current_trg_sparse_feat_map = self.trg_sparse_feat_map 

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
        return (self.src_sparse_feat_map.words2dense(
                        self._process_sentence(self.src_sentences[request],
                                               self.src_vocab_size)), 
                self.trg_sparse_feat_map.words2dense(
                        self._process_sentence(self.trgt_sentences[request],
                                               self.trgt_vocab_size)))

    def _process_sentence(self, sentence, vocab):
        """Prepares string representation of sentence for passing 
        down the data stream"""
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        ws = [int(w) for w in sentence.strip().split()]
        return [w if w < vocab else utils.UNK_ID for w in ws] + [utils.EOS_ID]
    
    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.src_file,
                self.trgt_file,
                self.src_vocab_size,
                self.trgt_vocab_size,
                self.preprocess)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        (src_file,
         trgt_file,
         src_vocab_size,
         trgt_vocab_size,
         preprocess) = state
        self.__init__(src_file,
                      trgt_file,
                      src_vocab_size,
                      trgt_vocab_size,
                      preprocess,
                      _current_src_sparse_feat_map,
                      _current_trg_sparse_feat_map)
             

# Beyond this point is code copied from the machine_translation.stream
# module in blocks-examples.

def _length(sentence_pair):
    """Assumes target is the last element in the tuple.
    
    This method is copied from machine_translation.stream in blocks-examples.
    """
    return len(sentence_pair[-1])


class PaddingWithEOS(Padding):
    """Pads a stream with given end of sequence idx.
    
    This class is copied from machine_translation.stream in blocks-examples.
    """
    
    def __init__(self, data_stream, eos_idx, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_batch[0]).dtype

            padded_batch = numpy.zeros(
                (len(source_batch), max_sequence_length) + rest_shape,
                dtype=dtype)
            for i, sample in enumerate(source_batch):
                padded_batch[i, :len(sample)] = sample
            # fs439: Nasty bug in blocks: If rest_shape is not empty 
            # padded_batch shape is batch_size x sen_len x rest. The sequence 
            # generator expects sen_len x batch_size x ... Somewhere between 
            # sequenceGenerator.cost_matrix and here, the last and first dims 
            # are shuffled, i.e. if rest_shape is empty we are fine but 
            # otherwise we need to shuffle here to batch_size x rest x sen_len
            # to get the expected behavior
            # fs439: TODO works only if len(rest_shape) == 1
            if len(rest_shape) == 1:
                padded_batch = numpy.transpose(padded_batch, (2, 0, 1))
            batch_with_masks.append(padded_batch)

            mask = numpy.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class _too_long(object):
    """Filters sequences longer than given sequence length.
    
    This class is copied from `machine_translation.stream` in 
    blocks-examples.
    """
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])
