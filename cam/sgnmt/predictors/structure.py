# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module implements constraints which assure that highly structured
output is well-formatted. For example, the bracket predictor checks for
balanced bracket expressions, and the OSM predictor prevents any sequence
of operations which cannot be compiled to a string.
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor

# Default operation IDs
OSM_EOP_ID = 4

OSM_SRC_POP_ID = 4
OSM_SET_MARKER_ID = 5
OSM_JUMP_FWD_ID = 6
OSM_JUMP_BWD_ID = 7
OSM_SRC_POP2_ID = 8
OSM_COPY_ID = 8
OSM_SRC_UNPOP_ID = 9


def load_external_lengths(path):
    """Loads a length distribution from a plain text file. The file
    must contain blank separated <length>:<score> pairs in each line.
    
    Args:
        path (string): Path to the length file.
    
    Returns:
        list of dicts mapping a length to its scores, one dict for each
        sentence.
    """
    lengths = []
    with open(path) as f:
        for line in f:
            scores = {}
            for pair in line.strip().split():
                if ':' in pair:
                    length, score = pair.split(':')
                    scores[int(length)] = float(score)
                else:
                    scores[int(pair)] = 0.0
            lengths.append(scores)
    return lengths


def update_trg_osm_ids(wmap_path):
    """Update the OSM_*_ID variables using a target word map.

    Args:
        wmap_path (string): Path to the wmap file.
    """
    global OSM_SRC_POP_ID, OSM_SET_MARKER_ID, OSM_JUMP_FWD_ID, \
           OSM_JUMP_BWD_ID, OSM_SRC_POP2_ID, OSM_COPY_ID, \
           OSM_SRC_UNPOP_ID
    if not wmap_path:
        return
    with open(wmap_path) as f:
        for line in f:
            word, word_id = line.strip().split()
            if word == "<SRC_POP>":
                OSM_SRC_POP_ID = int(word_id)
                logging.debug("OSM SRC_POP = %d" % OSM_SRC_POP_ID)
            elif word == "<SET_MARKER>":
                OSM_SET_MARKER_ID = int(word_id)
                logging.debug("OSM SET_MARKER = %d" % OSM_SET_MARKER_ID)
            elif word == "<JUMP_FWD>":
                OSM_JUMP_FWD_ID = int(word_id)
                logging.debug("OSM JUMP_FWD = %d" % OSM_JUMP_FWD_ID)
            elif word == "<JUMP_BWD>":
                OSM_JUMP_BWD_ID = int(word_id)
                logging.debug("OSM JUMP_BWD = %d" % OSM_JUMP_BWD_ID)
            elif word == "<SRC_POP2>":
                OSM_SRC_POP2_ID = int(word_id)
                logging.debug("OSM SRC_POP2 = %d" % OSM_SRC_POP2_ID)
            elif word == "<COPY>":
                OSM_COPY_ID = int(word_id)
                logging.debug("OSM COPY = %d" % OSM_COPY_ID)
            elif word == "<SRC_UNPOP>":
                OSM_SRC_UNPOP_ID = int(word_id)
                logging.debug("SRC_UNPOP = %d" % OSM_SRC_UNPOP_ID)


def update_src_osm_ids(wmap_path):
    """Update the OSM_*_ID variables using a source word map.

    Args:
        wmap_path (string): Path to the wmap file.
    """
    global OSM_EOP_ID
    if not wmap_path:
        return
    with open(wmap_path) as f:
        for line in f:
            word, word_id = line.strip().split()
            if word == "<EOP>":
                OSM_EOP_ID = int(word_id)
                logging.debug("OSM EOP = %d" % OSM_EOP_ID)


class OSMPredictor(Predictor):
    """This predictor applies the following constraints to an OSM output:

      - The number of POP tokens must be equal to the number of source
        tokens
      - JUMP_FWD and JUMP_BWD tokens are constraint to avoid jumping out 
        of bounds.

    The predictor supports the original OSNMT operation set (default)
    plus a number of variations that are set by the use_* arguments in
    the constructor.
    """
    
    def __init__(self, src_wmap, trg_wmap, use_jumps=True, use_auto_pop=False, 
                 use_unpop=False, use_pop2=False, use_src_eop=False,
                 use_copy=False):
        """Creates a new osm predictor.

        Args:
            src_wmap (string): Path to the source wmap. Used to grap
                               EOP id.
            trg_wmap (string): Path to the target wmap. Used to update
                               IDs of operations.
            use_jumps (bool): If true, use SET_MARKER, JUMP_FWD and
                              JUMP_BWD operations
            use_auto_pop (bool): If true, each word insertion
                                 automatically moves read head
            use_unpop (bool): If true, use SRC_UNPOP to move read head
                              to the left.
            use_pop2 (bool): If true, use two read heads to align
                             phrases
            use_src_eop (bool): If true, expect EOP tokens in the src
                                sentence
            use_copy (bool): If true, move read head at COPY operations
        """
        super(OSMPredictor, self).__init__()
        update_trg_osm_ids(trg_wmap)
        self.use_jumps = use_jumps
        self.use_auto_pop = use_auto_pop
        self.use_unpop = use_unpop
        self.use_src_eop = use_src_eop
        if use_src_eop:
            update_src_osm_ids(src_wmap)
        self.pop_ids = set([OSM_SRC_POP_ID])
        if use_pop2:
            self.pop_ids.add(OSM_SRC_POP2_ID)
        if use_copy:
            self.pop_ids.add(OSM_COPY_ID)
        self.illegal_sequences = []
        if use_jumps:
            self.illegal_sequences.extend([
                #[OSM_JUMP_FWD_ID, OSM_JUMP_BWD_ID],
                #[OSM_JUMP_BWD_ID, OSM_JUMP_FWD_ID],
                #[OSM_JUMP_FWD_ID, OSM_SET_MARKER_ID, OSM_JUMP_FWD_ID],
                #[OSM_JUMP_FWD_ID, OSM_SET_MARKER_ID, OSM_JUMP_BWD_ID],
                #[OSM_JUMP_BWD_ID, OSM_SET_MARKER_ID, OSM_JUMP_FWD_ID],
                #[OSM_JUMP_BWD_ID, OSM_SET_MARKER_ID, OSM_JUMP_BWD_ID],
                [OSM_SET_MARKER_ID, OSM_SET_MARKER_ID]
            ])
        if use_auto_pop:
            self.no_auto_pop = set()
            if use_jumps:
                self.no_auto_pop.add(OSM_JUMP_FWD_ID)
                self.no_auto_pop.add(OSM_JUMP_BWD_ID)
                self.no_auto_pop.add(OSM_SET_MARKER_ID)
            if use_unpop:
                self.no_auto_pop.add(OSM_SRC_UNPOP_ID)

    def _is_pop(self, token):
        if token in self.pop_ids:
            return True
        return self.use_auto_pop and token not in self.no_auto_pop
    
    def initialize(self, src_sentence):
        """Sets the number of source tokens.
        
        Args:
            src_sentence (list): Not used
        """
        if self.use_src_eop:
            self.src_len = src_sentence.count(OSM_EOP_ID) + 1
        else:
            self.src_len = len(src_sentence)
        self.n_holes = 0
        self.head = 0
        self.n_pop = 0
        self.history = []

    def predict_next(self):
        """Apply OSM constraints.
        
        Returns:
            dict.
        """
        ret = {}
        if self.n_pop >= self.src_len:
            return {utils.EOS_ID: 0.0} # Force EOS
        else:
            ret[utils.EOS_ID] = utils.NEG_INF
        if self.use_unpop and self.n_pop <= 0:
            ret[OSM_SRC_UNPOP_ID] = utils.NEG_INF
        if self.use_jumps:
            if self.head <= 0:
                ret[OSM_JUMP_BWD_ID] = utils.NEG_INF
            if self.head >= self.n_holes:
                ret[OSM_JUMP_FWD_ID] = utils.NEG_INF
        for seq in self.illegal_sequences:
            hist = seq[:-1]
            if self.history[-len(hist):] == hist:
                ret[seq[-1]] = utils.NEG_INF
        return ret
        
    def get_unk_probability(self, posterior):
        if self.n_pop >= self.src_len: # Force EOS
            return utils.NEG_INF
        return 0.0
    
    def consume(self, word):
        """Updates the number of holes, EOPs, and the head position."""
        if not self._is_pop(word):
            if self.use_unpop and word == OSM_SRC_UNPOP_ID:
                self.n_pop -= 1
            else:
                self.history.append(word)
        else:
            self.n_pop += 1
        if self.use_jumps:
            if word == OSM_SET_MARKER_ID:
                self.n_holes += 1
                self.head += 1
            elif word == OSM_JUMP_FWD_ID:
                self.head += 1
            elif word == OSM_JUMP_BWD_ID:
                self.head -= 1
    
    def get_state(self):
        return self.n_holes, self.head, self.n_pop
    
    def set_state(self, state):
        self.n_holes, self.head, self.n_pop = state

    def is_equal(self, state1, state2):
        """Trivial implementation"""
        return state1 == state2


class ForcedOSMPredictor(Predictor):
    """This predictor allows forced decoding with an OSM output, which
    essentially means running the OSM in alignment mode. This predictor
    assumes well-formed operation sequences. Please combine this
    predictor with the osm constraint predictor to satisfy this
    requirement. The state of this predictor is the compiled version of
    the current history. It allows terminal symbols which are 
    consistent with the reference. The end-of-sentence symbol is
    supressed until all words in the reference have been consumed.
    """
    
    def __init__(self, trg_wmap, trg_test_file):
        """Creates a new forcedosm predictor.

        Args:
            trg_wmap (string): Path to the target wmap file. Used to
                               grap OSM operation IDs.
            trg_test_file (string): Path to the plain text file with 
                                    the target sentences. Must have the
                                    same number of lines as the number
                                    of source sentences to decode
        """
        super(ForcedOSMPredictor, self).__init__()
        update_trg_osm_ids(trg_wmap)
        self.trg_sentences = []
        with open(trg_test_file) as f:
            for line in f:
                self.trg_sentences.append([int(w) 
                            for w in line.strip().split()])
    
    def initialize(self, src_sentence):
        """Resets compiled and head.
        
        Args:
            src_sentence (list): Not used
        """
        self.compiled = ["X"]
        self.head = 0
        self.cur_trg_sentence = self.trg_sentences[self.current_sen_id] 

    def _is_complete(self):
        """Returns true if the compiled sentence contains the right
        number of terminals.
        """
        n_terminals = len([s for s in self.compiled if s != "X"])
        return n_terminals == len(self.cur_trg_sentence)

    def _generate_alignments(self, align_stub=[], compiled_start_pos=0, 
                             sentence_start_pos=0):
        for pos in range(compiled_start_pos, len(self.compiled)):
            if self.compiled[pos] != 'X':
                word = int(self.compiled[pos])
                for sen_pos in range(sentence_start_pos, 
                                     len(self.cur_trg_sentence)):
                    if self.cur_trg_sentence[sen_pos] == word:
                        self._generate_alignments(
                            align_stub + [(pos, sen_pos)],
                            pos+1,
                            sen_pos+1)
                return
        self.alignments.append(align_stub)

    def _align(self):
        possible_words = [set() for _ in range(len(self.compiled))]
        self.alignments = []
        self._generate_alignments(align_stub=[])
        for alignment in self.alignments:
            alignment.append((len(self.compiled), len(self.cur_trg_sentence)))
            prev_compiled_pos = -1
            prev_sentence_pos = -1
            for compiled_pos, sentence_pos in alignment:
               section_words = set(
                   self.cur_trg_sentence[prev_sentence_pos+1:sentence_pos])
               if section_words:
                   seen_gap = False
                   for section_pos in range(prev_compiled_pos+1, 
                                            compiled_pos):
                       if self.compiled[section_pos] == "X":
                           if seen_gap:
                               possible_words[section_pos] |= section_words
                           else:
                               possible_words[section_pos].add(
                                   self.cur_trg_sentence[prev_sentence_pos 
                                                         + section_pos 
                                                         - prev_compiled_pos])
                               seen_gap = True
               prev_compiled_pos = compiled_pos
               prev_sentence_pos = sentence_pos
        return possible_words

    def predict_next(self):
        """Apply word reference constraints.
        
        Returns:
            dict.
        """
        ret = {OSM_SRC_POP_ID: 0.0}
        possible_words = self._align()
        if possible_words[self.head]:
            ret[OSM_SET_MARKER_ID] = 0.0
        if any(possible_words[:self.head]):
            ret[OSM_JUMP_BWD_ID] = 0.0
        if any(possible_words[self.head+1:]):
            ret[OSM_JUMP_FWD_ID] = 0.0
        if self._is_complete():
            ret[utils.EOS_ID] = 0.0
        for word in possible_words[self.head]:
            ret[word] = 0.0
        return ret
        
    def get_unk_probability(self, posterior):
        """Always returns -inf."""
        return utils.NEG_INF

    def _jump_op(self, step):
        self.head += step
        while self.compiled[self.head] != "X":
            self.head += step

    def _insert_op(self, op):
        self.compiled = self.compiled[:self.head] + [op] + \
                        self.compiled[self.head:]
        self.head += 1
    
    def consume(self, word):
        """Updates the compiled string and the head position."""
        if word == OSM_SET_MARKER_ID:
            self._insert_op("X")
        elif word == OSM_JUMP_FWD_ID:
            self._jump_op(1)
        elif word == OSM_JUMP_BWD_ID:
            self._jump_op(-1)
        elif word != OSM_SRC_POP_ID:
            self._insert_op(str(word))
    
    def get_state(self):
        return self.compiled, self.head
    
    def set_state(self, state):
        self.compiled, self.head = state

    def is_equal(self, state1, state2):
        """Trivial implementation"""
        return state1 == state2


class BracketPredictor(UnboundedVocabularyPredictor):
    """This predictor constrains the output to well-formed bracket
    expressions. It also allows to specify the number of terminals with
    an external length distribution file.
    """
    
    def __init__(self, max_terminal_id, closing_bracket_id, max_depth=-1, 
                 extlength_path=""):
        """Creates a new bracket predictor.
        
        Args:
            max_terminal_id (int): All IDs greater than this are 
                brackets
            closing_bracket_id (string): All brackets except these ones are 
                opening. Comma-separated list of integers.
            max_depth (int): If positive, restrict the maximum depth
            extlength_path (string): If this is set, restrict the 
                number of terminals to the distribution specified in
                the referenced file. Terminals can be implicit: We
                count a single terminal between each adjacent opening
                and closing bracket.
        """
        super(BracketPredictor, self).__init__()
        self.max_terminal_id = max_terminal_id
        try:
            self.closing_bracket_ids = utils.split_comma(closing_bracket_id, int)
        except:
            self.closing_bracket_ids = [int(closing_bracket_id)]
        self.max_depth = max_depth if max_depth >= 0 else 1000000
        if extlength_path:
            self.length_scores = load_external_lengths(extlength_path)
        else:
            self.length_scores = None
            self.max_length = 1000000
    
    def initialize(self, src_sentence):
        """Sets the current depth to 0.
        
        Args:
            src_sentence (list): Not used
        """
        self.cur_depth = 0
        self.ends_with_opening = True
        self.n_terminals = 0
        if self.length_scores:
            self.cur_length_scores = self.length_scores[self.current_sen_id]
            self.max_length = max(self.cur_length_scores)

    def _no_closing_bracket(self):
        return {i: utils.NEG_INF for i in self.closing_bracket_ids}
    
    def predict_next(self, words):
        """If the maximum depth is reached, exclude all opening
        brackets. If history is not balanced, exclude EOS. If the
        current depth is zero, exclude closing brackets.
        
        Args:
            words (list): Set of words to score
        Returns:
            dict.
        """
        if self.cur_depth == 0:
            # Balanced: Score EOS with extlengths, supress closing bracket
            if self.ends_with_opening:  # Initial predict next call
                ret = self._no_closing_bracket()
                ret[utils.EOS_ID] = utils.NEG_INF
                return ret
            return {utils.EOS_ID: self.cur_length_scores.get(
                        self.n_terminals, utils.NEG_INF) 
                       if self.length_scores else 0.0}
        # Unbalanced: do not allow EOS
        ret = {utils.EOS_ID: utils.NEG_INF}
        if (self.cur_depth >= self.max_depth 
                or self.n_terminals >= self.max_length):
            # Do not allow opening brackets
            ret.update({w: utils.NEG_INF for w in words 
                if (w > self.max_terminal_id 
                    and not w in self.closing_bracket_ids)})
        if (self.length_scores 
                and self.cur_depth == 1 
                and self.n_terminals > 0 
                and not self.n_terminals in self.cur_length_scores):
            # Do not allow to go back to depth 0 with wrong number of terminals
            ret.update(self._no_closing_bracket())
        return ret
        
    def get_unk_probability(self, posterior):
        """Always returns 0.0"""
        if self.cur_depth == 0 and not self.ends_with_opening:
            return utils.NEG_INF 
        return 0.0
    
    def consume(self, word):
        """Updates current depth and the number of consumed terminals."""
        if word in self.closing_bracket_ids:
            if self.ends_with_opening:
                self.n_terminals += 1
            self.cur_depth -= 1
            self.ends_with_opening = False
        elif word > self.max_terminal_id:
            self.cur_depth += 1
            self.ends_with_opening = True
    
    def get_state(self):
        """Returns the current depth and number of consumed terminals"""
        return self.cur_depth, self.n_terminals, self.ends_with_opening
    
    def set_state(self, state):
        """Sets the current depth and number of consumed terminals"""
        self.cur_depth, self.n_terminals, self.ends_with_opening = state

    def is_equal(self, state1, state2):
        """Trivial implementation"""
        return state1 == state2

    
