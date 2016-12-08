"""This module contains predictors for forced decoding. This can be
done either with one reference (forced ``ForcedPredictor``), or with
multiple references in form of a n-best list (forcedlst 
``ForcedLstPredictor``). 
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt.utils import NEG_INF


class ForcedPredictor(Predictor):
    """This predictor realizes forced decoding. It stores one target
    sentence for each source sentence and outputs predictive probability
    1 along this path, and 0 otherwise.
    """
    
    def __init__(self, trg_test_file):
        """Creates a new forced decoding predictor.
        
        Args:
            trg_test_file (string): Path to the plain text file with 
                                    the target sentences. Must have the
                                    same number of lines as the number
                                    of source sentences to decode
        """
        super(ForcedPredictor, self).__init__()
        self.trg_sentences = []
        with open(trg_test_file) as f:
            for line in f:
                self.trg_sentences.append([int(w) 
                            for w in line.strip().split()] + [utils.EOS_ID])
        self.n_consumed = 0 
        
    def get_unk_probability(self, posterior):
        """Returns negative infinity unconditionally: Words which are
        not in the target sentence have assigned probability 0 by
        this predictor.
        """
        return posterior.get(utils.UNK_ID, NEG_INF)
    
    def predict_next(self):
        """Returns a dictionary with one entry and value 0 (=log 1). The
        key is either the next word in the target sentence or (if the
        target sentence has no more words) the end-of-sentence symbol.
        """
        if self.n_consumed < len(self.cur_trg_sentence):
            return {utils.EOS_ID : NEG_INF,
                    self.cur_trg_sentence[self.n_consumed] : 0.0}
        else:
            return {utils.EOS_ID : 0.0}
    
    def initialize(self, src_sentence):
        """Fetches the corresponding target sentence and resets the
        current history.
        
        Args:
            src_sentence (list):  Not used
        """
        self.cur_trg_sentence = self.trg_sentences[self.current_sen_id] 
        self.n_consumed = 0
    
    def consume(self, word):
        """If ``word`` matches the target sentence, we increase the
        current history by one. Otherwise, we set this predictor in
        an invalid state, in which it always predicts </S>
        
        Args:
            word (int): Next word to consume
        """
        if self.n_consumed < len(self.cur_trg_sentence):
            trg_word = self.cur_trg_sentence[self.n_consumed]
            if trg_word != utils.UNK_ID and trg_word != word:
                self.cur_trg_sentence = [] # Mismatch with our target sentence
            else:
                self.n_consumed = self.n_consumed + 1
    
    def get_state(self):
        """``cur_trg_sentence`` can be changed so its part of the 
        predictor state
        """
        return self.n_consumed,self.cur_trg_sentence
    
    def set_state(self, state):
        """Set the predictor state. """
        self.n_consumed,self.cur_trg_sentence = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the state is the same """
        n1,s1 = state1
        n2,s2 = state2
        return n1 == n2 and s1 == s2


class ForcedLstPredictor(Predictor):
    """This predictor can be used for direct n-best list rescoring. In
    contrast to the ``ForcedPredictor``, it reads an n-best list in 
    Moses format and uses its scores as predictive probabilities of the
    </S> symbol. Everywhere else it gives the predictive probability 1 
    if the history corresponds to at least one n-best list entry, 0 
    otherwise. From the n-best list we use
    First column: Sentence id
    Second column: Hypothesis in integer format
    Last column: score
    
    Note: Behavior is undefined if you have duplicates in the n-best
    list
    
    TODO: Would be much more efficient to use Tries for 
    cur_trgt_sentences instead of a flat list.
    """
    
    def __init__(self, trg_test_file, use_scores = True, feat_name = None):
        """Creates a new n-best rescoring predictor instance.
        
        Args:
            trg_test_file (string):  Path to the n-best list
            use_scores (bool): Whether to use the scores from the
                               n-best list. If false, use uniform
                               scores of 0 (=log 1).
            feat_name (string): Instead of the combined score in the
                                last column of the Moses n-best list,
                                we can use one of the sparse features.
                                Set this to the name of the feature
                                (denoted as <name>= in the n-best list)
                                if you wish to do that.
        """
        super(ForcedLstPredictor, self).__init__()
        self.trg_sentences = []
        score = 0.0
        with open(trg_test_file) as f:
            for line in f:
                parts = line.split("|||")
                if len(parts) < 2:
                    logging.warn("Malformed line %s in n-best list %s" % (
                                        line.strip(),
                                        trg_test_file))
                else:
                    if use_scores:
                        score = self._get_score(parts, feat_name)
                    sen_id = int(parts[0].strip())
                    while len(self.trg_sentences) <= sen_id:
                        self.trg_sentences.append([])
                    sen = [int(w) for w in parts[1].strip().split()]
                    if sen and sen[0] == utils.GO_ID:
                        sen  = sen[1:]
                    if sen and sen[-1] == utils.EOS_ID:
                        sen = sen[:-1]
                    self.trg_sentences[sen_id].append((score, sen))
        
    def _get_score(self, parts, feat_name):
        """Get the score for a hypothesis.
        
        Args:
            parts (list): Parts of the n-best entry (separated by |||
                          in the Moses n-best format)
            feat_name (string): Name of the sparse feature which should
                                be used as score (or None to use the
                                combined score)
        """
        feat_str = "%s=" % feat_name
        if not feat_name:
            return float(parts[-1].strip()) if len(parts) > 2 else 0.0
        feat_parts = parts[-2].strip().split()
        for idx in xrange(len(feat_parts)-1):
            if feat_parts[idx] == feat_str:
                return float(feat_parts[idx+1])
        return 0.0

    def get_unk_probability(self, posterior):
        """Return negative infinity unconditionally - words outside the
        n-best list are not possible according to this predictor.
        """
        return posterior.get(utils.UNK_ID, NEG_INF)
    
    def predict_next(self):
        """Outputs 0.0 (i.e. prob=1) for all words for which there is 
        an entry ``in cur_trg_sentences``, and the score in 
        ``cur_trg_sentences`` if the current history is by itself equal
        to an entry in ``cur_trg_sentences``.
        
        TODO: The implementation here is fairly inefficient as it scans 
        through all target sentences linearly. Would be better to 
        organize the target sentences in a Trie
        """
        scores = {}
        hist_len = len(self.history)
        for sen_score,trg_sentence in self.cur_trg_sentences:
            sen_len = len(trg_sentence)
            if sen_len < hist_len:
                continue
            hist = [self.history[i] if trg_sentence[i] != utils.UNK_ID else utils.UNK_ID
                      for i in xrange(hist_len)] 
            if trg_sentence[:hist_len] == hist:
                if sen_len == hist_len:
                    scores[utils.EOS_ID] = sen_score
                else:
                    scores[trg_sentence[hist_len]] = 0.0
        if not utils.EOS_ID in scores:
            scores[utils.EOS_ID] = NEG_INF
        return scores
    
    def initialize(self, src_sentence):
        """Resets the history and loads the n-best list entries for the
        next source sentence
        
        Args:
            src_sentence (list): Not used
        """
        self.cur_trg_sentences = self.trg_sentences[self.current_sen_id] 
        self.history = []
    
    def consume(self, word):
        """Extends the current history by ``word``. """
        self.history.append(word)
    
    def get_state(self):
        """Returns the current history. """
        return self.history
    
    def set_state(self, state):
        """Sets the current history. """
        self.history = state

    def reset(self):
        """Empty method. """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1 == state2


