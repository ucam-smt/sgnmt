"""This module implements constraints which assure that highly structured
output is well-formatted. For example, the bracket predictor checks for
balanced bracket expressions, and the OSM predictor prevents any sequence
of operations which cannot be compiled to a string.
"""

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor, UnboundedVocabularyPredictor

OSM_EOP_ID = 4
OSM_GAP_ID = 5
OSM_JUMP_FWD_ID = 6
OSM_JUMP_BWD_ID = 7


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


class OSMPredictor(Predictor):
    """This predictor applies the following constraints to an OSM output:

      - The number of EOP (end-of-phrase) tokens must not exceed the number
        of source tokens.
      - JUMP_FWD and JUMP_BWD tokens are constraint to avoid jumping out of
        bounds.
    """
    
    def __init__(self):
        """Creates a new osm predictor."""
        super(OSMPredictor, self).__init__()
        self.illegal_sequences = [
            [OSM_JUMP_FWD_ID, OSM_JUMP_BWD_ID],
            [OSM_JUMP_BWD_ID, OSM_JUMP_FWD_ID],
            [OSM_JUMP_FWD_ID, OSM_GAP_ID, OSM_JUMP_FWD_ID],
            [OSM_JUMP_FWD_ID, OSM_GAP_ID, OSM_JUMP_BWD_ID],
            [OSM_JUMP_BWD_ID, OSM_GAP_ID, OSM_JUMP_FWD_ID],
            [OSM_JUMP_BWD_ID, OSM_GAP_ID, OSM_JUMP_BWD_ID],
            [OSM_GAP_ID, OSM_GAP_ID]
        ]
    
    def initialize(self, src_sentence):
        """Sets the number of source tokens.
        
        Args:
            src_sentence (list): Not used
        """
        self.src_len = len(src_sentence)
        self.n_holes = 0
        self.head = 0
        self.n_eop = 0
        self.history = []

    def predict_next(self):
        """Apply OSM constraints.
        
        Returns:
            dict.
        """
        ret = {}
        if self.n_eop >= self.src_len:
            ret[OSM_EOP_ID] = utils.NEG_INF
        else:
            ret[utils.EOS_ID] = utils.NEG_INF
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
        """Always returns 0.0"""
        return 0.0
    
    def consume(self, word):
        """Updates the number of holes, EOPs, and the head position."""
        if word != OSM_EOP_ID:
            self.history.append(word)
        if word == OSM_EOP_ID:
            self.n_eop += 1
        elif word == OSM_GAP_ID:
            self.n_holes += 1
            self.head += 1
        elif word == OSM_JUMP_FWD_ID:
            self.head += 1
        elif word == OSM_JUMP_BWD_ID:
            self.head -= 1
    
    def get_state(self):
        return self.n_holes, self.head, self.n_eop
    
    def set_state(self, state):
        self.n_holes, self.head, self.n_eop = state

    def reset(self):
        """Empty."""
        pass
    
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
    
    def __init__(self, trg_test_file):
        """Creates a new forcedosm predictor.

        Args:
            trg_test_file (string): Path to the plain text file with 
                                    the target sentences. Must have the
                                    same number of lines as the number
                                    of source sentences to decode
        """
        super(ForcedOSMPredictor, self).__init__()
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
        for pos in xrange(compiled_start_pos, len(self.compiled)):
            if self.compiled[pos] != 'X':
                word = int(self.compiled[pos])
                for sen_pos in xrange(sentence_start_pos, 
                                      len(self.cur_trg_sentence)):
                    if self.cur_trg_sentence[sen_pos] == word:
                        self._generate_alignments(
                            align_stub + [(pos, sen_pos)],
                            pos+1,
                            sen_pos+1)
                return
        self.alignments.append(align_stub)

    def _align(self):
        possible_words = [set() for _ in xrange(len(self.compiled))]
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
                   for section_pos in xrange(prev_compiled_pos+1, 
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
            

    def _align_old(self):
        """Aligns the compiled string to the reference and returns the
        possible words at each position."""
        possible_words = [set(self.cur_trg_sentence) 
            for _ in xrange(len(self.compiled))] # Initially no constraints
        # Constrain words before first gap
        for pos, symbol in enumerate(self.compiled):
            possible_words[pos] = set([self.cur_trg_sentence[pos]])
            if symbol == "X":
                break
        # Constrain using terminalls in self.compiled
        for pos, symbol in enumerate(self.compiled):
            if symbol != "X": # Propagate this constraint
                isymb = int(symbol)
                possible_words[pos] = set([])
                first_idx = self.cur_trg_sentence.index(isymb)
                last_idx = len(self.cur_trg_sentence) - \
                        self.cur_trg_sentence[::-1].index(isymb) - 1
                # First, a coarse constraint which disallows changing order
                rm_after = set(self.cur_trg_sentence[:first_idx] + [isymb]) \
                    - set(self.cur_trg_sentence[first_idx+1:])
                rm_before = set(self.cur_trg_sentence[last_idx:] + [isymb]) \
                    - set(self.cur_trg_sentence[:last_idx])
                for before_pos in xrange(pos):
                    possible_words[before_pos] -= rm_before
                for after_pos in xrange(pos+1, len(self.compiled)):
                    possible_words[after_pos] -= rm_after
                # Then, we refine constraints until the nearest gap
                if first_idx == last_idx:
                    cur_idx = first_idx - 1
                    for before_pos in xrange(pos-1, -1, -1):
                        if self.compiled[before_pos] == "X":
                            break
                        possible_words[before_pos] &= set(
                                [self.cur_trg_sentence[cur_idx]])
                        cur_idx -= 1
                    cur_idx = first_idx + 1
                    for after_pos in xrange(pos+1, len(self.compiled)):
                        if cur_idx >= len(self.cur_trg_sentence):
                            constraint = set([])
                        else:
                            constraint = set([self.cur_trg_sentence[cur_idx]])
                        possible_words[after_pos] &= constraint
                        if self.compiled[after_pos] == "X":
                            break
                        cur_idx += 1
        return possible_words
                
        

    def predict_next(self):
        """Apply word reference constraints.
        
        Returns:
            dict.
        """
        ret = {OSM_EOP_ID: 0.0}
        possible_words = self._align()
        if possible_words[self.head]:
            ret[OSM_GAP_ID] = 0.0
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
        if word == OSM_GAP_ID:
            self._insert_op("X")
        elif word == OSM_JUMP_FWD_ID:
            self._jump_op(1)
        elif word == OSM_JUMP_BWD_ID:
            self._jump_op(-1)
        elif word != OSM_EOP_ID:
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
            self.closing_bracket_ids = map(int, closing_bracket_id.split(","))
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

    def reset(self):
        """Empty."""
        pass
    
    def is_equal(self, state1, state2):
        """Trivial implementation"""
        return state1 == state2

    
