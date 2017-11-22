import copy
import logging

from cam.sgnmt import utils
from cam.sgnmt.predictors.vocabulary import SkipvocabInternalHypothesis as InternalHypo
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt.utils import w2f
import pywrapfst as fst
import numpy as np


import collections
class ParsePredictor(Predictor):
    """This predictor builds a determinized lattice, given a grammar. 
    """
    
    def __init__(self, grammar_path, nmt_predictor, word_out,
                 normalize_scores=True, to_log=True, beam_size=12):
        """Creates a new parse predictor.
        Args:
            grammar_path (string): Path to the grammar file
        """
        super(ParsePredictor, self).__init__()
        self.grammar_path = grammar_path
        self.nmt = nmt_predictor
        self.weight_factor = -1.0 if to_log else 1.0
        self.word_out = word_out
        self.use_weights = True
        self.normalize_scores = normalize_scores
        self.beam_size = beam_size
        self.cur_fst = None
        self.add_bos_to_eos_score = False
        self.cur_node = -1
        self.stack = []
        self.prepare_grammar()
    
    def prepare_grammar(self):
        self.lhs_to_rule_ids = collections.defaultdict(list)
        rules = []
        with open(self.grammar_path) as f:
            for idx, line in enumerate(f):
                nt, rule = line.split(':')
                nt = int(nt.strip())
                self.lhs_to_rule_ids[nt].append(idx)
                rules.append(rule.strip().split())  
        self.rule_id_to_rhs = dict() # reversed rhs
        for rule_idx, rule in enumerate(rules):
          self.rule_id_to_rhs[rule_idx] = [int(r) for r in reversed(rule) 
                                           if int(r) in self.lhs_to_rule_ids]

    def get_unk_probability(self, posterior):
        """Always returns negative infinity: Words outside the 
        translation lattice are not possible according to this
        predictor.
        
        Returns:
            float. Negative infinity
        """
        if utils.UNK_ID in posterior:
            return posterior[utils.UNK_ID] 
        else:
            return utils.NEG_INF
    
    def predict_next(self, predicting_next_word=False):
        """Adds outgoing arcs from the current node as permitted by 
        the current stack and the grammar
        Uses these arcs to define next permitted rule.
              
        Returns:
            dict. Set of words on outgoing arcs from the current node
            together with their scores, or an empty set if we currently
            have no active node or fst.
        """
        if self.cur_node < 0:
            return {}
        nmt_posterior = self.nmt.predict_next()
        current_nt = self.stack.pop()
        outgoing_rules = self.lhs_to_rule_ids[current_nt]
        scores = {rule_id: 0 for rule_id in outgoing_rules}
        if utils.EOS_ID in scores and self.add_bos_to_eos_score:
            scores[utils.EOS_ID] += self.bos_score
        for rule_id in scores:
            scores[rule_id] += nmt_posterior[rule_id]
        if self.word_out and not predicting_next_word:
            scores = self.find_word(scores)
        parse_posterior = self.finalize_posterior(scores, 
                                                  self.use_weights,
                                                  self.normalize_scores)
        return parse_posterior

    def is_best_terminal(self, posterior):
        best_rule_id = utils.argmax(posterior)
        rhs = self.rule_id_to_rhs[best_rule_id]
        if not rhs or rhs[0] in (utils.EOS_ID, utils.UNK_ID):
          return True
        else:
          return False

    def find_word(self, posterior):
        """Check whether rhs of best option in posterior is a terminal
        if it is, return the posterior for decoding
        if not, take the best result and follow that path until a word is found
        this follows a greedy 1best path through non-terminals

        if using beam search n approach, need to loop:
        - deep copy of state
        - get top n in posterior. check if they're words. If they are, return. 
        - For each argmax_n:
          - consume word and predict next. 
          - store posterior
          - restore original state
        - combine the posteriors
        - 
        """
        if self.is_best_terminal(posterior):
          return posterior
        if self.beam_size == 0:
          best_rule_id = utils.argmax(posterior)
          self.consume(best_rule_id)
          return self.predict_next()
        else:
          pass # not yet implemented


    def initialize(self, src_sentence):
        """Loads the FST from the file system and consumes the start
        of sentence symbol. 
        
        Args:
            src_sentence (list)
        """
        self.nmt.initialize(src_sentence)
        self.cur_fst = fst.Fst()
        self.cur_fst.set_start(self.cur_fst.add_state())
        self.cur_node = self.cur_fst.start()
        self.stack = self.rule_id_to_rhs[utils.GO_ID]
    
    def consume(self, word):
        """Updates the current node by following the arc labelled with
        ``word``. If there is no such arc, we set ``cur_node`` to -1,
        indicating that the predictor is in an invalid state. In this
        case, all subsequent ``predict_next`` calls will return the
        empty set.
        Args:
            word (int): Word on an outgoing arc from the current node
        Returns:
            float. Weight on the traversed arc
        """
        if self.cur_node < 0:
            return
        n = self.cur_fst.add_state()
        self.cur_fst.add_arc(self.cur_node, fst.Arc(word, word, 1, n))
        self.nmt.consume(word) 
        from_state = self.cur_node
        self.cur_node = None
        for arc in self.cur_fst.arcs(from_state):
            if arc.olabel == word:
                self.cur_node = arc.nextstate
                self.stack.extend(self.rule_id_to_rhs[arc.olabel])
                return self.weight_factor*w2f(arc.weight)
    
    def get_state(self):
        """Returns the current node. """
        return self.cur_node, self.stack, self.nmt.get_state()
    
    def set_state(self, state):
        """Sets the current node. """
        self.cur_node, self.stack, nmt_state = state
        self.nmt.set_state(nmt_state)
        

    def reset(self):
        """Resets the loaded FST object and current node. """
        self.cur_fst = None
        self.cur_node = None
        self.stack = []
        self.nmt.reset()
    
    def initialize_heuristic(self, src_sentence):
        """Creates a matrix of shortest distances between nodes. """
        self.distances = fst.shortestdistance(self.cur_fst, reverse=True)
    
    def estimate_future_cost(self, hypo):
        """The FST predictor comes with its own heuristic function. We
        use the shortest path in the fst as future cost estimator. """
        if not self.cur_node:
            return 0.0
        last_word = hypo.trgt_sentence[-1]
        for arc in self.cur_fst.arcs(self.cur_node):
            if arc.olabel == last_word:
                return w2f(self.distances[arc.nextstate])
        return 0.0
    
    def is_equal(self, state1, state2):
        """Returns true if the current node is the same """
        return state1 == state2



class TokParsePredictor(Predictor):
    """This predictor builds a determinized lattice, given a grammar.
    Unlike ParsePredictor, the grammar predicts tokens, not rules
    """
    
    def __init__(self, grammar_path, nmt_predictor, word_out,
                 normalize_scores=True, to_log=True, beam_size=12,
                 consume_out_of_class=False):
        """Creates a new parse predictor.
        Args:
            grammar_path (string): Path to the grammar file
        """
        super(TokParsePredictor, self).__init__()
        self.grammar_path = grammar_path
        self.nmt = nmt_predictor
        self.UNK_ID = 3
        self.weight_factor = -1.0 if to_log else 1.0
        self.word_out = word_out
        self.use_weights = True
        self.normalize_scores = normalize_scores
        self.beam_size = beam_size
        self.add_bos_to_eos_score = False
        self.stack = []
        self.current_lhs = None
        self.current_rhs = []
        self.consume_ooc = consume_out_of_class
        self.prepare_grammar()

    def prepare_grammar(self):
        self.lhs_to_can_follow = {}
        with open(self.grammar_path) as f:
            for line in f:
                nt, rule = line.split(':')
                nt = int(nt.strip())
                self.lhs_to_can_follow[nt] = set([int(r) for r in rule.strip().split()])
        self.last_nt_in_rule = {nt: True for nt in self.lhs_to_can_follow}
        for nt, following in self.lhs_to_can_follow.items():
            if 0 in following:
                following.remove(0)
                self.last_nt_in_rule[nt] = False
                    
    def is_nt(self, word):
        if word in self.lhs_to_can_follow:
            return True
        return False

    def get_unk_probability(self, posterior):
        """Return probability of unk from most recent posterior
        """
        if self.UNK_ID in posterior:
            return posterior[self.UNK_ID] 
        else:
            return utils.NEG_INF
    
    def predict_next(self, predicting_next_word=False):
        """predict next tokens as permitted by 
        the current stack and the grammar
        """
        nmt_posterior = self.nmt.predict_next()
        outgoing_rules = self.lhs_to_can_follow[self.current_lhs]       
        scores = {rule_id: 0.0 for rule_id in outgoing_rules}
        if utils.EOS_ID in scores and self.add_bos_to_eos_score:
            scores[utils.EOS_ID] += self.bos_score
        for rule_id in scores:
            scores[rule_id] += nmt_posterior[rule_id]
        scores = self.finalize_posterior(scores, 
                                         self.use_weights,
                                         self.normalize_scores)

        if self.word_out and not predicting_next_word:
            scores = self.find_word(scores)
        return scores
    
    def find_word_greedy(self, posterior):
        while not self.is_best_terminal(posterior):
            best_rule_id = utils.argmax(posterior)
            self.consume(best_rule_id)
            posterior = self.predict_next(predicting_next_word=True)
        return posterior

    def find_word_beam(self, posterior):
        top_tokens = utils.argmax_n(posterior, self.beam_size)
        hypos = [InternalHypo(posterior[tok], self.get_state(), tok) for tok in top_tokens
                 if self.is_nt(tok)]
        best_hypo = InternalHypo(utils.NEG_INF, None, None)
        best_posterior = None
        while hypos and hypos[0].score > best_hypo.score:
            next_hypos = []
            for hypo in hypos:
                self.set_state(copy.deepcopy(hypo.predictor_state))
                self.consume(hypo.word_to_consume)
                new_post = self.predict_next(predicting_next_word=True)
                next_state = copy.deepcopy(self.get_state())
                if self.is_best_terminal(new_post) and hypo.score > best_hypo.score:
                    best_hypo.predictor_state = next_state
                    best_hypo.score = hypo.score
                    best_posterior = new_post
                else:
                    for tok, score in new_post.items(): # slow but more exhaustive
                        if self.is_nt(tok):
                            next_hypos.append(InternalHypo(score + hypo.score, next_state, tok))
            next_hypos.sort(key=lambda h: -h.score)
            hypos = next_hypos[:self.beam_size]
        self.set_state(best_hypo.predictor_state)
        #for tok in best_posterior:
        #    best_posterior[tok] += best_hypo.score # include path score 
        return best_posterior


    def find_word(self, posterior):
        """Check whether rhs of best option in posterior is a terminal
        if it is, return the posterior for decoding
        if not, take the best result and follow that path until a word is found
        this follows a greedy 1best or a beam path through non-terminals
        """
        if self.beam_size <= 1:
            return self.find_word_greedy(posterior)
        else:
          if self.is_best_terminal(posterior):
              return posterior
          else:
              return self.find_word_beam(posterior)

    def is_best_terminal(self, posterior):
        # Return true if most probable token in posterior is a terminal or EOS token
        best_rule_id = utils.argmax(posterior)
        if best_rule_id == utils.EOS_ID:
            return True
        return not self.is_nt(best_rule_id)

    def initialize(self, src_sentence):
        """Loads the FST from the file system and consumes the start
        of sentence symbol. 
        
        Args:
            src_sentence (list):  Not used
        """
        self.nmt.initialize(src_sentence)
        self.current_lhs = None
        self.current_rhs = []
        self.stack = [utils.EOS_ID]
        self.consume(utils.GO_ID)

    def replace_lhs(self):
        while self.current_rhs:
            self.stack.append(self.current_rhs.pop())
        if self.stack:
            self.current_lhs = self.stack.pop()
        else:
            self.current_lhs = utils.EOS_ID

    def consume(self, word, external=False):
        """Updates the current node by following the arc labelled with
        ``word``. If there is no such arc, we set ``cur_node`` to -1,
        indicating that the predictor is in an invalid state. In this
        case, all subsequent ``predict_next`` calls will return the
        empty set.
        Args:
            word (int): Word on an outgoing arc from the current node
        Returns:
            float. Weight on the traversed arc
        """
        #logging.info('consuming {}'.format(word))
        if self.current_lhs:
            current_allowed = self.lhs_to_can_follow[self.current_lhs]
        else:
            current_allowed = set([utils.GO_ID])
        if word == utils.UNK_ID:
            logging.info('changing unk consumption')
            word = self.UNK_ID
        if not self.consume_ooc and word not in current_allowed:
            #logging.info('currently allowed words like {}, not {}. Consuming unk instead.'.format(list(current_allowed)[-1], word))
            word = self.UNK_ID
        #if external:
            #logging.info('Parse consuming {}'.format(word))
        self.nmt.consume(word) 
        if self.is_nt(word):
            self.current_rhs.append(word)
            if self.last_nt_in_rule[word]:
                self.replace_lhs()
        else:
            self.replace_lhs()            
        return self.weight_factor
    
    def get_state(self):
        """Returns the current node. """
        return self.stack, self.current_lhs, self.current_rhs, self.nmt.get_state()
    
    def set_state(self, state):
        """Sets the current node. """
        self.stack, self.current_lhs, self.current_rhs, nmt_state = state
        self.nmt.set_state(nmt_state)
        

    def reset(self):
        self.stack = [utils.EOS_ID]
        self.current_lhs = None
        self.current_rhs = []
        self.nmt.reset()
    
    def initialize_heuristic(self, src_sentence):
        """Creates a matrix of shortest distances between nodes. """
        pass
    
    def estimate_future_cost(self, hypo):
        return 0.0
    
    def is_equal(self, state1, state2):
        """Returns true if the current node is the same """
        return state1 == state2



