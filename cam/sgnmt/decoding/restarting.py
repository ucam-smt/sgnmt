"""Implementation of the restarting search strategy """

import copy
from heapq import heappop, heappush, heapify
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
from cam.sgnmt.utils import INF
import numpy as np


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
    
    def __init__(self, 
                 decoder_args,
                 hypo_recombination,
                 max_expansions = 0,
                 low_memory_mode = True,
                 node_cost_strategy='difference',
                 stochastic=False,
                 always_single_step=False):
        """Creates new Restarting decoder instance.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination 
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
            low_memory_mode (bool): Switch on low memory mode at cost 
                                    of some computational overhead as
                                    the set of open nodes is reduced
                                    after each decoding pass
            node_cost_strategy (string): How to decide which node to 
                                          restart from next
            stochastic (bool): If true, select the next node to restart
                               from randomly. If false, take the one
                               with the best node score
            always_single_step (bool): If false, do greedy decoding 
                                       when restarting. If true, expand
                                       the hypothesis only by a single
                                       token
        """
        super(RestartingDecoder, self).__init__(decoder_args)
        self.max_expansions_param = max_expansions
        self.always_single_step = always_single_step
        self.low_memory_mode = low_memory_mode
        self.hypo_recombination = hypo_recombination
        if node_cost_strategy == 'difference':
            self.get_node_cost = self._get_node_cost_difference
        elif node_cost_strategy == 'absolute':
            self.get_node_cost = self._get_node_cost_absolute
        elif node_cost_strategy == 'constant':
            self.get_node_cost = self._get_node_cost_constant
        elif node_cost_strategy == 'expansions':
            self.get_node_cost = self._get_node_cost_expansions
        else:
            logging.fatal("restarting node score strategy unknown!")
        if stochastic:
            self.select_node = self._select_node_stochastic
        else:
            self.select_node = self._select_node_max
    
    def _select_node_stochastic(self):
        """Implements stochastic node selection. """
        exps = np.exp([-c for c,_ in self.open_nodes])
        total = sum(exps)
        idx = np.random.choice(range(len(exps)),
                               p=[e/total for e in exps])
        return self.open_nodes.pop(idx)
    
    def _select_node_max(self):
        """Implements deterministic node selection. """
        return heappop(self.open_nodes)
    
    def _get_node_cost_difference(self,
                                   prev_node_cost,
                                   first_word_score, 
                                   sec_word_score):
        """Implements the node scoring function difference. """
        return first_word_score - sec_word_score
    
    def _get_node_cost_absolute(self,
                                   prev_node_cost,
                                   first_word_score, 
                                   sec_word_score):
        """Implements the node scoring function absolute. """
        return -sec_word_score
    
    def _get_node_cost_constant(self,
                                   prev_node_cost,
                                   first_word_score, 
                                   sec_word_score):
        """Implements the node scoring function constant. """
        return 0.0
    
    def _get_node_cost_expansions(self,
                                   prev_node_cost,
                                   first_word_score, 
                                   sec_word_score):
        """Implements the node scoring function constant. """
        return prev_node_cost + 1.0
    
    def greedy_decode(self, hypo):
        """Helper function for greedy decoding from a certain point in
        the search tree."""
        best_word = hypo.trgt_sentence[-1]
        prev_hypo = hypo
        remaining_exps = max(self.max_expansions - self.apply_predictors_count,
                             1)
        while (best_word != utils.EOS_ID 
               and len(prev_hypo.trgt_sentence) <= self.max_len):
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
            logging.debug("Expanded hypo: score=%f prefix= %s" % (
                            new_hypo.score,
                            ' '.join([str(w) for w in new_hypo.trgt_sentence])))
            if len(posterior) > 1:
                if not self.always_single_step:
                    posterior.pop(best_word)
                children = sorted([RestartingChild(w,
                                                   posterior[w],
                                                   score_breakdown[w])
                    for w in posterior], key=lambda c: c.score, reverse=True)
                children = children[:remaining_exps]
                node_cost = self.get_node_cost(0.0, 
                                               best_word_score, 
                                               children[0].score)
                if node_cost <= self.max_heap_node_cost:
                    prev_hypo.predictor_states = copy.deepcopy(
                                                self.get_predictor_states())
                    heappush(self.open_nodes, (node_cost,
                                               RestartingNode(prev_hypo,
                                                              children)))
            prev_hypo = new_hypo
            if self.always_single_step:
                break
        if best_word == utils.EOS_ID:
            self.add_full_hypo(prev_hypo.generate_full_hypothesis())
            if prev_hypo.score > self.best_score: 
                logging.info("New_best (ID: %d): score=%f exp=%d hypo=%s" 
                    % (self.current_sen_id + 1,
                       prev_hypo.score, 
                       self.apply_predictors_count,
                       ' '.join([str(w) for w in prev_hypo.trgt_sentence])))
                self.best_score = prev_hypo.score
    
    def create_initial_node(self):
        """Create the root node for the search tree. """
        init_hypo = PartialHypothesis()
        posterior,score_breakdown = self.apply_predictors()
        children = sorted([RestartingChild(w, posterior[w], score_breakdown[w])
                            for w in posterior],
                          key=lambda c: c.score, reverse=True)
        init_hypo.predictor_states = self.get_predictor_states()
        heappush(self.open_nodes, (0.0, RestartingNode(init_hypo, children)))

    def decode(self, src_sentence):
        """Decodes a single source sentence using Restarting search. """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        self.open_nodes = []
        self.best_score = self.get_lower_score_bound()
        self.max_heap_node_cost = INF
        # First, create a RestartingNode object for the initial state
        self.create_initial_node()
        # Then, restart from open nodes until the heap is empty
        while self.open_nodes:
            prev_node_score,node = self.select_node()
            best_child = node.children.pop(0)
            new_hypo = node.hypo.expand(best_child.word,
                                        None,
                                        best_child.score,
                                        best_child.score_breakdown)
            if new_hypo.score > self.best_score: # Admissible pruning
                logging.debug("Restart from %s" % (
                            ' '.join([str(w) for w in new_hypo.trgt_sentence])))
                if node.children: # Still has children -> back to heap
                    node_cost = self.get_node_cost(prev_node_score, 
                                                   best_child.score, 
                                                   node.children[0].score)
                    heappush(self.open_nodes, (node_cost, node))
                    self.set_predictor_states(copy.deepcopy(
                                                node.hypo.predictor_states))
                else: # No need to copy, don't put back to heap
                    self.set_predictor_states(node.hypo.predictor_states)
                self.greedy_decode(new_hypo)
            # Reduce heap size (we don't need more nodes than remaining exps
            rest = self.max_expansions - self.apply_predictors_count
            if rest <= 0:
                break
            if self.hypo_recombination:
                new_open = []
                while len(new_open) < rest and self.open_nodes:
                    c_cost,candidate = heappop(self.open_nodes)
                    valid = True
                    for _,node in new_open:
                        if self.are_equal_predictor_states(
                                                candidate.hypo.predictor_states,
                                                node.hypo.predictor_states):
                            valid = False
                            break
                    if valid:
                        new_open.append((c_cost, candidate))
                        if len(new_open) > rest:
                            break
                self.open_nodes = new_open
                heapify(self.open_nodes) 
            elif self.low_memory_mode and len(self.open_nodes) > rest:
                new_open = [heappop(self.open_nodes) for _ in xrange(rest+1)]
                self.max_heap_node_cost = new_open[-1][0]
                self.open_nodes = new_open
                heapify(self.open_nodes)
        return self.get_full_hypos_sorted()

