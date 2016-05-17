"""This module contains high level decoder algorithms such as greedy,
A*, beam search, or depth first search schemes. Note that all decoders
are implemented in a monotonic left-to-right way which works well in
the predictors paradigm. If we use features which do not have natural
left-to-right semantics, we

1) Restrict it to a accept/not-accept decision or
2) Change it s.t. it does have left-to-right semantics

For example, to use synchronous grammars, we could

1.) Keep track of all parse trees which match the partial prefix 
    sequence
2.) Transform the grammar into Greibach normal form

The reason for this design decision is the emphasis on NMT: The neural 
decoder decodes the sequence from left to right, all other features
(i.e. predictors) are conceptually rather guiding the neural decoding. 
"""

import copy
from heapq import heappush, heappop
import logging
import operator

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder
from cam.sgnmt.decoding import core
import numpy as np


NEG_INF = float("-inf")


class Hypothesis:
    """Complete translation hypotheses are represented by an instance
    of this class. We store the produced sentence, the combined score,
    and a score breakdown to the separate predictor scores.
    """
    
    def __init__(self, trgt_sentence, total_score, score_breakdown = []):
        """Creates a new full hypothesis.
        
        Args:
            trgt_sentence (list): List of target word ids without <S> 
                                  or </S> which make up the target 
                                  sentence
            total_score (float): combined total score of this hypo
            score_breakdown (list): Predictor score breakdown for each
                                    target token in ``trgt_sentence``
        """
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown

    def __repr__(self):
        """Returns a string representation of this hypothesis."""
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)


class PartialHypothesis:
    """Represents a partial hypothesis in various decoders. """
    
    def __init__(self, initial_states = None):
        """Creates a new partial hypothesis with zero score and empty
        translation prefix.
        
        Args:
            initial_states: Initial predictor states
        """
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score = 0.0
        self.score_breakdown = []
        self.word_to_consume = None
    
    def get_last_word(self):
        """Get the last word in the translation prefix. """
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]
    
    def generate_full_hypothesis(self):
        """Create a ``Hypothesis`` instance from this hypothesis. """
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown)
    
    def expand(self, word, new_states, score, score_breakdown):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability and updates the
        stored predictor states.
        
        Args:
            word (int): New word to add to the translation prefix
            new_states (object): Predictor states after consuming
                                 ``word``
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        hypo = PartialHypothesis(new_states)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.add_score_breakdown(score_breakdown)
        return hypo
    
    def cheap_expand(self, word, score, score_breakdown):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability. Does NOT update the
        predictor states but adds a flag which signals that the last 
        word in this hypothesis has not been consumed yet by the 
        predictors. This can save memory because we can reuse the 
        current state for many hypothesis. It also saves computation
        as we do not consume words which are then discarded anyway by
        the search procedure.
        
        Args:
            word (int): New word to add to the translation prefix
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        hypo = PartialHypothesis(self.predictor_states)
        hypo.score = self.score + score
        hypo.score_breakdown = copy.copy(self.score_breakdown)
        hypo.trgt_sentence = self.trgt_sentence + [word]
        hypo.word_to_consume = word
        hypo.add_score_breakdown(score_breakdown)
        return hypo

    def add_score_breakdown(self, added_scores):
        """Helper function for adding the word level score breakdowns
        for the newly added word to the hypothesis score breakdown.
        """
        self.score_breakdown.append(added_scores)
        self.score = core.breakdown2score_partial(self.score,
                                                  self.score_breakdown)


class GreedyDecoder(Decoder):
    """The greedy decoder does not revise decisions and therefore does
    not have to maintain predictor states. Therefore, this 
    implementation is particularly simple and can be used as template
    for more complex decoders. The greedy decoder can be imitated with
    the ``BeamDecoder`` with beam size 1.
    """
    
    def __init__(self, closed_vocab_norm):
        """Initialize the greedy decoder. """
        super(GreedyDecoder, self).__init__(closed_vocab_norm)
    
    def decode(self, src_sentence):
        """Decode a single source sentence in a greedy way: Always take
        the highest scoring word as next word and proceed to the next
        position. This makes it possible to decode without using the 
        predictors ``get_state()`` and ``set_state()`` methods as we
        do not have to keep track of predictor states.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of a single best ``Hypothesis`` instance."""
        self.initialize_predictors(src_sentence)
        trgt_sentence = []
        score_breakdown = []
        trgt_word = None
        score = 0.0
        while trgt_word != utils.EOS_ID:
            posterior,breakdown = self.apply_predictors()
            trgt_word = utils.argmax(posterior)
            score += posterior[trgt_word]
            trgt_sentence.append(trgt_word)
            score_breakdown.append(breakdown[trgt_word])
            self.consume(trgt_word)
        return [Hypothesis(trgt_sentence, score, score_breakdown)]
        

class DFSDecoder(Decoder):
    """This decoder implements depth first search without using
    heuristics. This is the most efficient search algorithm for 
    complete enumeration of the search space as it minimizes the 
    number of ``get_state()`` and ``set_state()`` calls. Note that
    this DFS implementation has no cycle detection, i.e. if the search
    space has cycles this decoder may run into an infinite loop.
    """
    
    def __init__(self,
                 closed_vocab_norm,
                 early_stopping = True,
                 max_expansions = 0):
        """Creates new DFS decoder instance.
        
        Args:
            closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
            early_stopping (bool): Enable safe (admissible) branch
                                   pruning if the accumulated score
                                   is already worse than the currently
                                   best complete score. Do not use if
                                   scores can be positive
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
        """
        super(DFSDecoder, self).__init__(closed_vocab_norm)
        self.early_stopping = early_stopping
        self.max_expansions = 1000000 if max_expansions <= 0 else max_expansions
    
    def _dfs(self, partial_hypo):
        """Recursive function for doing dfs. Note that we do not keep
        track of the predictor states inside ``partial_hypo``, because 
        at each call of ``_dfs`` the current predictor states are equal
        to the hypo predictor states.
        ATTENTION: Early stopping plus DFS produces wrong results if 
        you have positive scores!
        
        Args:
            partial_hypo (PartialHypothesis): Partial hypothesis 
                                              generated so far. 
        """
        logging.debug("Expand: best_score: %f partial_score: %f sentence: %s" %
                      (self.best_score,
                       partial_hypo.score,
                       partial_hypo.trgt_sentence))
        if partial_hypo.get_last_word() == utils.EOS_ID:
            self.hypos.append(partial_hypo.generate_full_hypothesis())
            self.best_score = max(self.best_score, partial_hypo.score)
            return
        if self.apply_predictors_count > self.max_expansions: # pruning
            return
        posterior,score_breakdown = self.apply_predictors()
        if len(posterior) > 1: # deep copy only if necessary
            pred_states = copy.deepcopy(self.get_predictor_states()) 
        reload_states = False
        for trgt_word,score in sorted(posterior.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True):
            if reload_states: # TODO: save one deepcopy (in last iteration)
                self.set_predictor_states(copy.deepcopy(pred_states))
            new_hypo =  partial_hypo.expand(trgt_word,
                                            None, # Do not store states
                                            score,
                                            score_breakdown[trgt_word])
            if not self.early_stopping or new_hypo.score > self.best_score:
                self.consume(trgt_word)
                self._dfs(new_hypo)
            reload_states = True
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using depth first search.
        If ``max_expansions`` equals 0, this corresponds to exhaustive
        search for the globally best scoring hypothesis. Note that with
        ``early_stopping`` enabled, the returned set of hypothesis are
        not necessarily the global n-best hypotheses. To create an 
        exact n-best list, disable both ``max_expansions`` and 
        ``early_stopping`` in the constructor.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score. If ``max_expansions`` equals 0, the first element
            holds the global best scoring hypothesis
        """
        self.initialize_predictors(src_sentence)
        self.hypos = []
        self.best_score = NEG_INF
        self._dfs(PartialHypothesis())
        return sorted(self.hypos,
                      key=lambda hypo: hypo.total_score,
                      reverse=True)


class BeamDecoder(Decoder):
    """This decoder implements beam search without heuristics. """
    
    def __init__(self, closed_vocab_norm, beam_size, early_stopping = True):
        """Creates a new beam decoder instance
        
        Args:
            closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
            beam_size (int): Absolute beam size. A beam of 12 means
                             that we keep track of 12 active hypothesis
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable
        """
        super(BeamDecoder, self).__init__(closed_vocab_norm)
        self.beam_size = beam_size
        self.stop_criterion = self._best_eos if early_stopping else self._all_eos

    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        return hypos[-1].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        for hypo in hypos:
            if hypo.get_last_word() != utils.EOS_ID:
                return True
        return False
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]
        it = 0
        while self.stop_criterion(hypos):
            if it > 2*len(src_sentence): # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(hypo.score)
                    continue 
                self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
                if not hypo.word_to_consume is None: # Consume if cheap expand
                    self.consume(hypo.word_to_consume)
                posterior,score_breakdown = self.apply_predictors()
                hypo.predictor_states = self.get_predictor_states()
                top = utils.argmax_n(posterior, self.beam_size)
                for trgt_word in top:
                    next_hypo = hypo.cheap_expand(trgt_word,
                                                  posterior[trgt_word],
                                                  score_breakdown[trgt_word])
                    next_hypos.append(next_hypo)
                    next_scores.append(next_hypo.score)
            hypos = [next_hypos[idx] 
                        for idx in np.argsort(next_scores)[-self.beam_size:]]
        ret = [hypos[-idx-1].generate_full_hypothesis() 
                for idx in xrange(len(hypos)) 
                     if hypos[-idx-1].get_last_word() == utils.EOS_ID]
        if not ret:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            return [hypos[-idx-1].generate_full_hypothesis() 
                        for idx in xrange(len(hypos))]
        return ret


class AstarDecoder(Decoder):
    """This decoder implements A*. For heuristics, see the the 
    ``decoding.core`` module for interfaces and the general handling of
    heuristics, and the ``decoding.heuristics`` package for heuristic
    implementations. This A* implementation does not have a 'closed
    set', i.e. we do not keep track of already visited states. Make 
    sure that your search space is acyclic (normally it is unless you
    decode on cyclic lattices with the fst predictor.
    """
    
    def __init__(self, closed_vocab_norm, capacity = 0, nbest=1):
        """Creates a new A* decoder instance
        
        Args:
            closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
            capacity (int): If positive, defines the maximum size of
                            the priority queue. This can be used to
                            introduce some pruning. If 0, we use a
                            PQ with unlimited capacity.
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        """
        super(AstarDecoder, self).__init__(closed_vocab_norm)
        self.nbest = nbest
        self.capacity = capacity

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.initialize_predictors(src_sentence)
        open_set = []
        ret = []
        heappush(open_set, (0.0,
                            PartialHypothesis(self.get_predictor_states())))
        while open_set:
            c,hypo = heappop(open_set)
            logging.debug("Expand; estimated cost: %f score: %f sentence: %s" % 
                         (-c,hypo.score,hypo.trgt_sentence))
            if hypo.get_last_word() == utils.EOS_ID: # Found best hypothesis
                ret.append(hypo.generate_full_hypothesis())
                if len(ret) >= self.nbest: # return if we have enough hypos
                    return ret
                continue
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
            if not hypo.word_to_consume is None: # Consume if cheap expand
                self.consume(hypo.word_to_consume)
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()
            for trgt_word in posterior: # Estimate future cost, add to heap
                next_hypo = hypo.cheap_expand(trgt_word, posterior[trgt_word],
                                                  score_breakdown[trgt_word])
                combined_score = next_hypo.score - self.estimate_future_cost(
                                                                    next_hypo)
                heappush(open_set, (-combined_score, next_hypo))
            # Limit heap capacity
            if self.capacity > 0 and len(open_set) > self.capacity:
                new_open_set = []
                for _ in xrange(self.capacity):
                    heappush(new_open_set, heappop(open_set))
                open_set = new_open_set
        return ret


class RestartingChild(object):
    """Helper class for ``RestartingDecoder``` representing a child
    object in the search tree. """
    
    def __init__(self, word, score, score_breakdown):
        """Creates a new child instance """
        self.word = word
        self.score = score
        self.score_breakdown = score_breakdown


class RestartingNode(object):
    """Helper class for ``RestartingDecoder``` representing a node
    in the search tree. """
    
    def __init__(self, hypo, children):
        """Creates a new node instance """
        self.hypo = hypo
        self.children = children
    
    
class RestartingDecoder(Decoder):
    """This decoder first creates a path to the final node greedily.
    Then, it looks for the node on this path with the smallest 
    difference between best and second best child, and restarts greedy 
    decoding from this point. In order to do so, it maintains a 
    priority queue of all visited nodes, which is ordered by the 
    difference between the worst expanded child and the best unexpanded
    one. If this queue is empty, we have visited the best path. This 
    algorithm is similar to DFS but does not backtrace to the last call
    of the recursive function but to the one which is most promising.
    
    Note that this algorithm is exact. It tries to exploit the problem
    characteristics of NMT search: Reloading predictor states can be
    expensive, node expansion is even more expensive but for free from
    visited nodes, and there is no good admissible heuristic.
    
    Note2: Does not work properly if predictor scores can be positive 
    because of admissible pruning
    """
    
    def __init__(self, closed_vocab_norm, max_expansions = 0):
        """Creates new Restarting decoder instance.
        
        Args:
            closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
        """
        super(RestartingDecoder, self).__init__(closed_vocab_norm)
        self.max_expansions = 1000000 if max_expansions <= 0 else max_expansions
    
    def greedy_decode(self, hypo):
        """Helper function for greedy decoding from a certain point in
        the search tree."""
        best_word = hypo.trgt_sentence[-1]
        prev_hypo = hypo
        while best_word != utils.EOS_ID:
            self.consume(best_word)
            posterior,score_breakdown = self.apply_predictors()
            if len(posterior) < 1:
                return
            best_word = utils.argmax(posterior)
            best_word_score = posterior[best_word]
            new_hypo = prev_hypo.expand(best_word,
                                        None,
                                        best_word_score,
                                        score_breakdown[best_word])
            if new_hypo.score < self.best_score: # Admissible pruning
                return
            logging.debug("Expanded hypo: score=%f prefix=%s" % (
                                                    new_hypo.score,
                                                    new_hypo.trgt_sentence))
            if len(posterior) > 1:
                posterior.pop(best_word)
                children = sorted([RestartingChild(w,
                                                   posterior[w],
                                                   score_breakdown[w])
                    for w in posterior], key=lambda c: c.score, reverse=True)
                prev_hypo.predictor_states = copy.deepcopy(
                                                self.get_predictor_states())
                heappush(self.open_nodes,
                         (best_word_score-children[0].score,
                          RestartingNode(prev_hypo, children)))
            prev_hypo = new_hypo
        self.hypos.append(prev_hypo.generate_full_hypothesis())
        self.best_score = max(self.best_score, prev_hypo.score)
    
    def create_initial_node(self):
        """Create the root node for the search tree. """
        init_hypo = PartialHypothesis()
        posterior,score_breakdown = self.apply_predictors()
        children = sorted([RestartingChild(w, posterior[w], score_breakdown[w])
                            for w in posterior],
                          key=lambda c: c.score, reverse=True)
        init_hypo.predictor_states = self.get_predictor_states()
        heappush(self.open_nodes, (0, RestartingNode(init_hypo, children)))

    def decode(self, src_sentence):
        """Decodes a single source sentence using Restarting search. """
        self.initialize_predictors(src_sentence)
        self.hypos = []
        self.open_nodes = []
        self.best_score = NEG_INF
        # First, create a RestartingNode object for the initial state
        self.create_initial_node()
        # Then, restart from open nodes until the heap is empty
        while (self.open_nodes
               and self.apply_predictors_count <= self.max_expansions):
            _,node = heappop(self.open_nodes)
            best_child = node.children.pop(0)
            new_hypo = node.hypo.expand(best_child.word,
                                        None,
                                        best_child.score,
                                        best_child.score_breakdown)
            if new_hypo.score > self.best_score: # Admissible pruning
                logging.debug("Best score: %f. Restart from %s" % (
                                                    self.best_score,
                                                    new_hypo.trgt_sentence))
                if node.children: # Still has children -> back to heap
                    heappush(self.open_nodes,
                            (best_child.score-node.children[0].score, node))
                    self.set_predictor_states(copy.deepcopy(
                                                node.hypo.predictor_states))
                else: # No need to copy, don't put back to heap
                    self.set_predictor_states(node.hypo.predictor_states)
                self.greedy_decode(new_hypo)
        return sorted(self.hypos,
                      key=lambda hypo: hypo.total_score,
                      reverse=True)
