"""Implementation of the bow search strategy """

import copy
from heapq import heappush, heappop, heapify
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
import numpy as np


class BOWNode(object):
    """Helper class for ``BOWDecoder``` representing a child
    object in the search tree. 
    
    Attributes:
        hypo (PartialHypothesis): Hypothesis corresponding to this node
        posterior (dict): Scores on outgoing arcs
        score_breakdown (dict): Score breakdowns for outgoing arcs
        prev_nodes (list): Path from the root to this node
        active_arcs (dict): Dictionary of unexplored outgoing arcs
    """
    
    def __init__(self, hypo, posterior, score_breakdown, prev_nodes):
        """Creates a new search tree node. All outgoing arcs are set to
        active by default.
        
        Args:
            hypo (PartialHypothesis): Hypothesis corresponding to this node
            posterior (dict): Scores on outgoing arcs
            score_breakdown (dict): Score breakdowns for outgoing arcs
            prev_nodes (list): Path from the root to this node
        """
        self.hypo = hypo
        self.posterior = posterior
        self.score_breakdown = score_breakdown
        self.prev_nodes = prev_nodes
        self.active_arcs = {k: True for k in self.posterior.iterkeys()}


class BOWDecoder(Decoder):
    """This decoder is designed to work well for bag-of-words decoding
    experiments. It is similar to the restarting decoder as it back-
    traces to a node in the search tree and does greedy decoding from
    that point. However, the strategy to select the node from which to
    restart differs: we select the node with the highest expected total
    score, which is estimated by taking the score of a full hypothesis 
    with the same prefix plus the score in the posterior in the
    corresponding decoding run of the first deviating token. This 
    heuristic assumes that the score of the full hypothesis will be
    the same as the previous one, just using the better score at the
    deviating node in place. This assumption is more likely to hold in
    fixed length decoding problems, e.g. bag-of-words tasks.
    
    Note that this decoder does not support the ``max_length`` 
    parameter as it is designed for fixed length decoding problems.
    
    Also note that even if this decoder is designed for bag problems,
    it can be used even in case of variable length hypotheses.
    Therefore, you still need to use the bow predictor.
    """
    
    def __init__(self, 
                 decoder_args,
                 hypo_recombination,
                 max_expansions = 0,
                 stochastic=False,
                 early_stopping=True,
                 always_single_step=False):
        """Creates a new bag decoder.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
            stochastic (bool): If true, select the next node to restart
                               from randomly. If false, take the one
                               with the best node score
            early_stopping (bool): Activates inadmissible pruning. Do
                                   not use if you have positive scores
            always_single_step (bool): If true, we do only a single 
                                       expansion from the backtraced
                                       node and select a new node. If
                                       false, we perform greedy 
                                       decoding from that node until
                                       we reach a final node
        """
        super(BOWDecoder, self).__init__(decoder_args) 
        self.max_expansions_param = max_expansions
        self.early_stopping = early_stopping
        self.hypo_recombination = hypo_recombination
        self.always_single_step = always_single_step
        if stochastic:
            self.select_node = self._select_node_stochastic
        else:
            self.select_node = self._select_node_max
    
    def greedy_decode(self, node, word, single_step):
        """Helper function for greedy decoding from a certain point in
        the search tree."""
        
        prev_hypo = node.hypo.expand(word,
                                     None,
                                     node.posterior[word],
                                     node.score_breakdown[word])
        prev_nodes = node.prev_nodes + [node]
        
        best_word = word
        while ((prev_hypo.score > self.best_score or not self.early_stopping)
               and best_word != utils.EOS_ID):
            self.consume(best_word)
            posterior,score_breakdown = self.apply_predictors()
            if len(posterior) < 1:
                return
            self._update_best_word_scores(posterior)
            best_word = utils.argmax(posterior)
            best_word_score = posterior[best_word]
            new_hypo = prev_hypo.expand(best_word,
                                        None,
                                        best_word_score,
                                        score_breakdown[best_word])
            logging.debug("Expanded hypo: len=%d score=%f prefix= %s" % (
                            len(new_hypo.trgt_sentence),
                            new_hypo.score,
                            ' '.join([str(w) for w in new_hypo.trgt_sentence])))
            node = BOWNode(prev_hypo, 
                           posterior, 
                           score_breakdown, 
                           list(prev_nodes))
            if not single_step:
                del node.active_arcs[best_word]
                if len(node.active_arcs) > 0:
                    prev_hypo.predictor_states = copy.deepcopy(
                                                self.get_predictor_states())
            else:
                prev_hypo.predictor_states = self.get_predictor_states()
            prev_nodes.append(node)
            prev_hypo = new_hypo
            if single_step:
                break
        full_hypo_score = prev_hypo.score
        if best_word == utils.EOS_ID: # Full hypo
            self.add_full_hypo(prev_hypo.generate_full_hypothesis())
            self.best_score = max(self.best_score, full_hypo_score)
        else:
            full_hypo_score = self._estimate_full_hypo_score(prev_hypo)
        # Update the heap
        sen = prev_hypo.trgt_sentence
        l = len(sen)
        for pos in xrange(l):
            worst_scores = {}
            for p in xrange(pos, l):
                worst_scores[sen[p]] = min(worst_scores.get(sen[p], 0.0),
                                           prev_nodes[p].posterior[sen[p]])
            node = prev_nodes[pos]
            for w in node.active_arcs:
                expected_score = self._estimate_full_hypo_score(
                                node.hypo.cheap_expand(w, 
                                                       node.posterior[w], 
                                                       node.score_breakdown[w]))
                self._add_to_heap(node, w, expected_score) 
    
    def _update_best_word_scores(self, posterior):
        """Maintains the average unigram scores for each target word
        estimating future costs of partial hypothesis. This is required
        when using early stopping as we do not know the exact score of
        the full path.
        """
        for w,score in posterior.iteritems():
            if w in self.best_word_cnts:
                self.best_word_cnts[w] += 1.0
                self.best_word_sums[w] += score
            else:
                self.best_word_cnts[w] = 1.0
                self.best_word_sums[w] = score
            self.best_word_scores[w] = self.best_word_sums[w] / self.best_word_cnts[w]
            #self.best_word_scores[w] = max(self.best_word_scores.get(w, NEG_INF),
            #                               score)
            #self.best_word_scores[w] = score

    def _estimate_full_hypo_score(self, hypo):
        """Only needed if early stopping is set to true. Estimates the 
        the future cost of a partial hypothesis which have been pruned
        by admissible pruning
        """
        if not self.full_hypos:
            return hypo.score
        remaining_words = {utils.EOS_ID: 1}
        for w in self.full_hypos[0].trgt_sentence:
            remaining_words[w] = remaining_words.get(w, 0) + 1
        for w in hypo.trgt_sentence:
            remaining_words[w] -= 1
        acc = hypo.score
        for w,cnt in remaining_words.iteritems():
            if cnt > 0: # Sanity check
                acc += cnt*self.best_word_scores.get(w, 0)
        return acc

    def _add_to_heap(self, node, w, expected_score):
        """Add a node to the heap. Note that we may put the same node
        to the heap multiple times. This assures that the node is
        scored with its best score. Node selection makes sure that
        we do not traverse nodes twice.
        """
        heappush(self.open_nodes, (-expected_score, (node, w))) 
    
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
    
    def create_initial_node(self):
        """Create the root node for the search tree. """
        init_hypo = PartialHypothesis()
        posterior,score_breakdown = self.apply_predictors()
        best_word = utils.argmax(posterior)
        init_hypo.predictor_states = self.get_predictor_states()
        init_node = BOWNode(init_hypo, posterior, score_breakdown, [])
        self._add_to_heap(init_node, best_word, 0.0) # Expected score irrelevant 

    def decode(self, src_sentence):
        """Decodes a single source sentence using BOW search. """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        self.open_nodes = []
        self.best_score = self.get_lower_score_bound()
        self.best_word_scores = {}
        self.best_word_cnts = {}
        self.best_word_sums = {}
        # First, create a RestartingNode object for the initial state
        self.create_initial_node()
        # Then, restart from open nodes until the heap is empty
        single_step = False
        while (self.open_nodes 
                and self.max_expansions > self.apply_predictors_count):
            exp_score,tmp = self.select_node()
            node,word = tmp
            if not word in node.active_arcs: # Already expanded
                continue
            del node.active_arcs[word]
            if node.hypo.score + node.posterior[word] <= self.best_score:
                continue # Admissible pruning
            logging.debug(
                "Best: %f Expected: %f Expansions: %d Restart from %s -> %d" % (
                          self.best_score,
                          exp_score,
                          self.apply_predictors_count,
                          ' '.join([str(w) for w in node.hypo.trgt_sentence]),
                          word))
            if node.active_arcs:
                self.set_predictor_states(copy.deepcopy(
                                                node.hypo.predictor_states))
            else:
                self.set_predictor_states(node.hypo.predictor_states)
            self.greedy_decode(node, word, single_step)
            single_step = self.always_single_step
            if self.hypo_recombination:
                rest = self.max_expansions - self.apply_predictors_count
                new_open = []
                while len(new_open) < rest and self.open_nodes:
                    c_cost,candidate = heappop(self.open_nodes)
                    valid = True
                    for _,el in new_open:
                        if self.are_equal_predictor_states(
                                            candidate[0].hypo.predictor_states,
                                            el[0].hypo.predictor_states):
                            valid = False
                            break
                    if valid:
                        new_open.append((c_cost, candidate))
                        if len(new_open) > rest:
                            break
                self.open_nodes = new_open
                heapify(self.open_nodes)
        return self.get_full_hypos_sorted()
