"""This module contains wrapper predictors which support decoding with
diverse tokenization. The ``Word2charPredictor`` can be used if the 
decoder operates on fine-grained tokens such as characters, but the
tokenization of a predictor is coarse-grained (e.g. words or subwords).

The ``word2char`` predictor maintains an explicit list of word boundary
characters and applies consume and predict_next whenever a word boundary
character is consumed.

The ``fsttok`` predictor also masks coarse grained predictors when SGNMT
uses fine-grained tokens such as characters. This wrapper loads an FST
which transduces character to predictor-unit sequences.
"""

import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor, Predictor
from cam.sgnmt.utils import NEG_INF, common_get


EPS_ID = 0
"""OpenFST's reserved ID for epsilon arcs. """


class CombinedState(object):
    """Combines an FST state with predictor state. Use by the fsttok
    predictor.
    """
    
    def __init__(self, 
                 fst_node, 
                 pred_state, 
                 posterior, 
                 unconsumed = [], 
                 pending_score = 0.0):
        self.fst_node = fst_node
        self.pred_state = pred_state
        self.posterior = posterior
        self.unconsumed = list(unconsumed)
        self.pending_score = pending_score
    
    def traverse_fst(self, trans_fst, char):
        """Returns a list of ``CombinedState``s with the same predictor
        state and posterior, but an ``fst_node`` which is reachable
        via the input label ``char``. If the output tabe contains
        symbols, add them to ``unconsumed``.
        
        Args:
            trans_fst (Fst): FST to traverse
            char (int): Index of character
        
        Returns:
            list. List of combined states reachable via ``char``
        """
        ret = []
        self._dfs(trans_fst, ret, self.fst_node, char, self.unconsumed)
        return ret
    
    def _dfs(self, trans_fst, acc, root_node, char, cur_unconsumed):
        """Helper method for ``traverse_fst`` for traversing the FST
        along ``char`` and epsilon arcs with DFS.
        
        Args:
            trans_fst (Fst): FST to traverse
            acc (list): Accumulator list
            root_node (int): State in the FST to start
            char (int): Index of character
            cur_unconsumed (list): Unconsumed predictor tokens so far
        """
        for arc in trans_fst.arcs(root_node):
            next_unconsumed = list(cur_unconsumed)
            if arc.olabel != EPS_ID:
                next_unconsumed.append(arc.olabel)
            if arc.ilabel == EPS_ID:
                self._dfs(trans_fst, acc, arc.nextstate, char, next_unconsumed)
            elif arc.ilabel == char:
                acc.append(CombinedState(arc.nextstate,
                                         self.pred_state,
                                         self.posterior,
                                         next_unconsumed,
                                         self.pending_score))
    
    def score(self, token, predictor):
        """Returns a score which can be added if ``token`` is consumed
        next. This is not necessarily the full score but an upper bound
        on it: Continuations will have a score lower or equal than
        this. We only use the current posterior vector and do not
        consume tokens with the wrapped predictor.
        """
        if token and self.unconsumed:
            self.consume_all(predictor)
        s = self.pending_score
        if token:
            s += self._get_token_score(token, predictor)
        return s
    
    def consume_all(self, predictor):
        """Consume all unconsumed tokens and update pred_state, 
        pending_score, and posterior accordingly.
        
        Args:
            predictor (Predictor): Predictor instance
        """
        if not self.unconsumed:
            return
        if self.posterior is None:
            self.update_posterior(predictor)
        predictor.set_state(copy.deepcopy(self.pred_state))
        for token in self.unconsumed:
            self.pending_score += self._get_token_score(token, predictor)
            predictor.consume(token)
            self.posterior = predictor.predict_next()
        self.pred_state = copy.deepcopy(predictor.get_state())
        self.unconsumed = []
    
    def consume_single(self, predictor):
        """Consume a single token in ``self.unconsumed``.
        
        Args:
            predictor (Predictor): Predictor instance
        """
        if not self.unconsumed:
            return
        if not self.posterior is None:
            self.pending_score += self._get_token_score(self.unconsumed[0],
                                                        predictor)
            self.posterior = None
    
    def _get_token_score(self, token, predictor):
        """Look up ``token`` in ``self.posterior``. """
        return utils.common_get(self.posterior,
                                token,
                                predictor.get_unk_probability(self.posterior))
        
    def update_posterior(self, predictor):
        """If ``self.posterior`` is None, call ``predict_next`` to
        be able to score the next tokens.
        """
        if not self.posterior is None:
            return
        predictor.set_state(copy.deepcopy(self.pred_state))
        predictor.consume(self.unconsumed[0])
        self.posterior = predictor.predict_next()
        self.pred_state = copy.deepcopy(predictor.get_state())
        self.unconsumed = self.unconsumed[1:]
        

class FSTTokPredictor(Predictor):
    """This wrapper can be used if the SGNMT decoder operates on the
    character level, but a predictor uses a more coarse grained 
    tokenization. The mapping is defined by an FST which transduces
    character to predictor unit sequences. This wrapper maintains a
    list of ``CombinedState`` objects which are tuples of an FST node
    and a predictor state for which holds:
    
    - The input labels on the path to the node are consistent with the
      consumed characters
    - The output labels on the path to the node are consistent with the
      predictor states
    """
    
    def __init__(self, path, fst_unk_id, max_pending_score, slave_predictor):
        """Constructor for the fsttok wrapper
        
        Args:
            path (string): Path to an FST which transduces characters 
                           to predictor tokens
            fst_unk_id (int): ID used to represent UNK in the FSTs
                              (usually 999999998)
            max_pending_score (float): Maximum pending score in a
                                       ``CombinedState`` instance.
            slave_predictor (Predictor): Wrapped predictor
        """
        super(FSTTokPredictor, self).__init__()
        self.max_pending_score = max_pending_score 
        self.fst_unk_id = fst_unk_id
        self.slave_predictor = slave_predictor
        if isinstance(slave_predictor, UnboundedVocabularyPredictor):
            logging.fatal("fsttok cannot wrap an unbounded "
                          "vocabulary predictor.")
        self.trans_fst = utils.load_fst(path)
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified. ``states`` is updated to the initial FST node and
        predictor posterior and state.
        """
        self.slave_predictor.initialize(src_sentence)
        posterior = self.slave_predictor.predict_next()
        self.states = [CombinedState(self.trans_fst.start(),
                                     self.slave_predictor.get_state(),
                                     posterior)]
        self.last_prediction = {}
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        logging.warning("fsttok does not support predictor heuristics")
        self.slave_predictor.initialize_heuristic(src_sentence)
    
    def predict_next(self):
        self.last_prediction = {}
        for state in self.states:
            self._collect_chars(state, state.fst_node, None)
        return self.last_prediction
    
    def _collect_chars(self, state, root_node, first_olabel):
        """Recursively builds up ``last_prediction`` by traversing
        epsilon arcs in the FST from ``root_node``
        """
        for arc in self.trans_fst.arcs(root_node):
            arc_first_olabel = first_olabel if first_olabel else arc.olabel
            if arc.ilabel == EPS_ID:
                self._collect_chars(state, arc.nextstate, arc_first_olabel)
            else:
                score = state.score(arc_first_olabel, self.slave_predictor)
                if arc.ilabel in self.last_prediction: 
                    self.last_prediction[arc.ilabel] = max(
                                            self.last_prediction[arc.ilabel], 
                                            score)
                else:
                    self.last_prediction[arc.ilabel] = score
        
    def get_unk_probability(self, posterior):
        """Always returns negative infinity. Handling UNKs needs to be 
        realized by the FST.
        """
        return utils.NEG_INF

    def _choose_better(self, s1, s2):
        """``consume`` merges states if they have the same ``fst_node``
        This method defines which one to keep. We prefer states that
        
        1) have less unconsumed UNK tokens
        2) have higher pending_score
        """
        if s1 is None:
            return s2
        n_unk1 = len([1 for t in s1.unconsumed if t == self.fst_unk_id])
        n_unk2 = len([1 for t in s2.unconsumed if t == self.fst_unk_id])
        if n_unk1 > n_unk2:
            return s2
        if n_unk1 < n_unk2:
            return s1
        if s1.pending_score < s2.pending_score:
            return s2
        return s1
    
    def consume(self, word):
        """Update ``self.states`` to be consistent with ``word`` and 
        consumes all the predictor tokens.
        """
        next_states = []
        for state in self.states:
            next_states.extend(state.traverse_fst(self.trans_fst, word))
        consumed_score = self.last_prediction.get(word, 0.0)
        for state in next_states:
            state.pending_score -= consumed_score
            state.consume_single(self.slave_predictor)
        # if two states have the same fst_node, keep only the better one
        # Also: Remove states with too large pending_score
        uniq_states = {}
        for state in next_states:
            if state.pending_score < -self.max_pending_score:
                continue
            n = state.fst_node
            uniq_states[n] = self._choose_better(uniq_states.get(n, None),
                                                 state)
        self.states = list(uniq_states.itervalues())
    
    def get_state(self):
        return self.states, self.last_prediction
    
    def set_state(self, state):
        self.states, self.last_prediction = state

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

    def estimate_future_cost(self, hypo):
        """Not implemented yet"""
        return 0.0

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(FSTTokPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Not implemented yet"""
        return False
    

class Word2charPredictor(UnboundedVocabularyPredictor):
    """This predictor wraps word level predictors when SGNMT is running
    on the character level. The mapping between word ID and character 
    ID sequence is loaded from the file system. All characters which
    do not appear in that mapping are treated as word boundary
    makers. The wrapper blocks consume and predict_next calls until a
    word boundary marker is consumed, and updates the slave predictor
    according the word between the last two word boundaries. The 
    mapping is done only on the target side, and the source sentences
    are passed through as they are. To use alternative tokenization on
    the source side, see the altsrc predictor wrapper. The word2char
    wrapper is always an ``UnboundedVocabularyPredictor``.
    """
    
    def __init__(self, map_path, slave_predictor):
        """Creates a new word2char wrapper predictor. The map_path 
        file has to be plain text files, each line containing the 
        mapping from a word index to the character index sequence
        (format: word char1 char2... charn).
        
        Args:
            map_path (string): Path to the mapping file
            slave_predictor (Predictor): Instance of the predictor with
                                         a different wmap than SGNMT
        """
        super(Word2charPredictor, self).__init__()
        self.slave_predictor = slave_predictor
        self.words = SimpleTrie()
        self.word_chars = {}
        with open(map_path) as f:
            for line in f:
                l = [int(x) for x in line.strip().split()]
                word = l[0]
                chars = l[1:]
                self.words.add(chars, word)
                for c in chars:
                    self.word_chars[c] = True   
        if isinstance(slave_predictor, UnboundedVocabularyPredictor): 
            self._get_stub_prob = self._get_stub_prob_unbounded
            self._start_new_word = self._start_new_word_unbounded
        else:
            self._get_stub_prob = self._get_stub_prob_bounded
            self._start_new_word = self._start_new_word_bounded             
    
    def initialize(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize(src_sentence)
        self._start_new_word()
    
    def initialize_heuristic(self, src_sentence):
        """Pass through to slave predictor. The source sentence is not
        modified 
        """
        self.slave_predictor.initialize_heuristic(src_sentence)
    
    def _update_slave_vars(self, posterior):
        self.slave_unk = self.slave_predictor.get_unk_probability(posterior)
        self.slave_go = common_get(posterior, utils.GO_ID, self.slave_unk)
        self.slave_eos = common_get(posterior, utils.EOS_ID, self.slave_unk)
        
    def _start_new_word_unbounded(self):
        """start_new_word implementation for unbounded vocabulary slave
        predictors. Needs to set slave_go, slave_eos, and slave_unk
        """
        self.word_stub = []
        posterior = self.slave_predictor.predict_next([utils.UNK_ID,
                                                       utils.GO_ID,
                                                       utils.EOS_ID])
        self._update_slave_vars(posterior)
    
    def _start_new_word_bounded(self):
        """start_new_word implementation for bounded vocabulary slave
        predictors. Needs to set slave_go, slave_eos, slave_unk, and
        slave_posterior
        """
        self.word_stub = []
        self.slave_posterior = self.slave_predictor.predict_next()
        self._update_slave_vars(self.slave_posterior)
    
    def _get_stub_prob_unbounded(self):
        """get_stub_prob implementation for unbounded vocabulary slave
        predictors.
        """
        word = self.words.get(self.word_stub)
        if word:
            posterior = self.slave_predictor.predict_next([word])
            return common_get(posterior, word, self.slave_unk)
        return self.slave_unk
    
    def _get_stub_prob_bounded(self):
        """get_stub_prob implementation for bounded vocabulary slave
        predictors.
        """
        word = self.words.get(self.word_stub)
        return common_get(self.slave_posterior,
                          word if word else utils.UNK_ID,
                          self.slave_unk)
    
    def predict_next(self, trgt_words):
        posterior = {}
        stub_prob = False
        for ch in trgt_words:
            if ch in self.word_chars:
                posterior[ch] = 0.0
            else: # Word boundary marker
                if stub_prob is False:
                    stub_prob = self._get_stub_prob() if self.word_stub else 0.0
                posterior[ch] = stub_prob
        if utils.GO_ID in posterior:
            posterior[utils.GO_ID] += self.slave_go
        if utils.EOS_ID in posterior:
            posterior[utils.EOS_ID] += self.slave_eos
        return posterior
        
    def get_unk_probability(self, posterior):
        """This is about the unkown character, not word. Since the word
        level slave predictor has no notion of the unknown character, 
        we return NEG_INF unconditionally.
        """
        return NEG_INF
    
    def consume(self, word):
        """If ``word`` is a word boundary marker, truncate ``word_stub``
        and let the slave predictor consume word_stub. Otherwise, 
        extend ``word_stub`` by the character.
        """
        if word in self.word_chars:
            self.word_stub.append(word)
        elif self.word_stub:
            word = self.words.get(self.word_stub)
            self.slave_predictor.consume(word if word else utils.UNK_ID)
            self._start_new_word()
    
    def get_state(self):
        """Pass through to slave predictor """
        return self.word_stub, self.slave_predictor.get_state()
    
    def set_state(self, state):
        """Pass through to slave predictor """
        self.word_stub, slave_state = state
        self.slave_predictor.set_state(slave_state)

    def reset(self):
        """Pass through to slave predictor """
        self.slave_predictor.reset()

    def estimate_future_cost(self, hypo):
        """Not supported """
        logging.warn("Cannot use future cost estimates of predictors "
                     "wrapped by word2char")
        return 0.0

    def set_current_sen_id(self, cur_sen_id):
        """We need to override this method to propagate current\_
        sentence_id to the slave predictor
        """
        super(Word2charPredictor, self).set_current_sen_id(cur_sen_id)
        self.slave_predictor.set_current_sen_id(cur_sen_id)
    
    def is_equal(self, state1, state2):
        """Pass through to slave predictor """
        stub1, slave_state1 = state1
        stub2, slave_state2 = state2
        return (stub1 == stub2 
                and self.slave_predictor.is_equal(slave_state1, slave_state2))

