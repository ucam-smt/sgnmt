"""This module implements constraints which assure that highly structured
output is well-formatted. For example, the bracket predictor checks for
balanced bracket expressions, and the OSM predictor prevents any sequence
of operations which cannot be compiled to a string.
"""

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor

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


class OSMPredictor(UnboundedVocabularyPredictor):
    """This predictor applies the following constraints to an OSM output:

      - The number of EOP (end-of-phrase) tokens must not exceed the number
        of source tokens.
      - JUMP_FWD and JUMP_BWD tokens are constraint to avoid jumping out of
        bounds.
    """
    
    def __init__(self):
        """Creates a new osm predictor."""
        super(OSMPredictor, self).__init__()
    
    def initialize(self, src_sentence):
        """Sets the number of source tokens.
        
        Args:
            src_sentence (list): Not used
        """
        self.src_len = len(src_sentence)
        self.n_holes = 0
        self.head = 0
        self.n_eop = 0

    def predict_next(self, words):
        """Apply OSM constraints.
        
        Args:
            words (list): Set of words to score
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
        return ret
        
    def get_unk_probability(self, posterior):
        """Always returns 0.0"""
        return 0.0
    
    def consume(self, word):
        """Updates the number of holes, EOPs, and the head position."""
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

    
