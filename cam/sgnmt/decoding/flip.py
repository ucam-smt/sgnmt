"""Implementation of the flip search strategy """

import copy
from heapq import heappush, heappop
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
from cam.sgnmt.misc.trie import SimpleTrie


class FlipCandidate(object):
    """Helper class for ``FlipDecoder``. Represents a full but yet 
    unscored hypothesis which differs from an explored hypo by one
    flip or move operation.
    """

    def __init__(self, trgt_sentence, scores, bigram_scores, max_score):
        """Creates a new candidate hypothesis.
        
        Args:
            trgt_sentence (list): Full target sentence
            scores (list): Word level scores. Same length as
                           ``trgt_sentence``
            bigram_scores (dict): Bigram scores collected along the
                                  parent hypothesis
            max_score (float): Maximum possible score this candidate
                               can achieve
        """
        self.trgt_sentence = trgt_sentence
        self.scores = scores
        self.bigram_scores = bigram_scores
        self.max_score = max_score
        self.expected_score = sum(scores)


class FlipDecoder(Decoder):
    """The flip decoder explores the search space by permutating
    already explored hypotheses with a single permutation operation. We
    support two operations: 'flip' flips the position of two target 
    tokens. 'move' moves one target token to another location in the
    sentence.
    
    Note that this decoder does not support the ``max_length`` 
    parameter as it is designed for fixed length decoding problems.
    
    Also note that this decoder works only for bag-of-words problems.
    Do not use the bow predictor in combination with this decoder as
    it will hide the EOS scores which are important to estimate bigram
    scores.
    """
    
    def __init__(self, 
                 decoder_args,
                 trg_test_file, 
                 max_expansions=0,
                 early_stopping=True,
                 flip_strategy='move',
                 always_greedy=False):
        """Creates a new flip decoder. Do not use this decoder in 
        combination with the bow predictor as it inherently already
        satisfies the bag-of-word constrains.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            trg_test_file (string): Path to a plain text file which 
                                    defines the bag of words
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
            early_stopping (boolean): Activates admissible pruning
            flip_strategy (string): How the search space is traversed.
                                   'move' moves one token to another
                                   position, 'flip' switches the
                                   positions of two target tokens
            always_greedy (boolean): Per default, the flip decoder does
                                     forced decoding along the complete
                                     candidate sentence. Set to True to
                                     do greedy decoding from the
                                     backtraced node instead
        """
        super(FlipDecoder, self).__init__(decoder_args) 
        self.max_expansions_param = max_expansions
        self.early_stopping = early_stopping
        self.always_greedy = always_greedy
        with open(trg_test_file) as f:
            self.lines = f.read().splitlines()
        if flip_strategy == 'flip':
            self._extract_candidates = self._extract_candidates_flip
        elif flip_strategy == 'move':
            self._extract_candidates = self._extract_candidates_move
        else:
            logging.fatal("Unknown flip strategy!")
    
    def _greedy_decode(self):
        """Performs greedy decoding from the start node. Used to obtain
        the initial hypothesis.
        """
        hypo = PartialHypothesis()
        hypos = []
        posteriors = []
        score_breakdowns = []
        scores = []
        bag = dict(self.full_bag)
        while bag:
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = copy.deepcopy(self.get_predictor_states())
            hypos.append(hypo)
            posteriors.append(posterior)
            score_breakdowns.append(score_breakdown)
            best_word = utils.argmax({w: posterior[w] for w in bag})
            bag[best_word] -= 1
            if bag[best_word] < 1:
                del bag[best_word]
            self.consume(best_word)
            hypo = hypo.expand(best_word,
                               None,
                               posterior[best_word],
                               score_breakdown[best_word])
            scores.append(posterior[best_word])
        posterior,score_breakdown = self.apply_predictors()
        hypo.predictor_states = self.get_predictor_states()
        hypos.append(hypo)
        posteriors.append(posterior)
        score_breakdowns.append(score_breakdown)
        hypo = hypo.expand(utils.EOS_ID,
                           None,
                           posterior[utils.EOS_ID],
                           score_breakdown[utils.EOS_ID])
        logging.debug("Greedy hypo (%f): %s" % (
                          hypo.score,
                          ' '.join([str(w) for w in hypo.trgt_sentence])))
        scores.append(posterior[utils.EOS_ID])
        self.best_score = hypo.score
        self.add_full_hypo(hypo.generate_full_hypothesis())
        self._process_new_hypos(FlipCandidate(hypo.trgt_sentence,
                                               scores,
                                               self._create_dummy_bigrams(),
                                               hypo.score),
                                 len(hypo.trgt_sentence),
                                 hypos,
                                 posteriors,
                                 score_breakdowns)
    
    def _create_dummy_bigrams(self):
        """Creates a dictionary of optimistic bigram scores which is to
        be updated as the decoder steps through the target sentence
        """ 
        bigram_scores = {}
        words = [utils.GO_ID, utils.EOS_ID] + [w for w in self.full_bag]
        for w in words:
            bigram_scores[w] = {w2: 0.0 for w2 in words}
        return bigram_scores

    def _is_explored(self, trgt_sentence):
        """Returns true if this target sentence has been explored
        already
        """
        prefix = self.explored.get_prefix(trgt_sentence)
        return self.explored.get(prefix)
    
    def _process_new_hypos(self, 
                           candidate, 
                           max_pos, 
                           explored_hypos, 
                           explored_posteriors, 
                           explored_score_breakdowns):
        """This method is called after a new candidate has been 
        explored. It derives new candidates by updating the bigram
        scores and delegating to ``_extract_candidates`` for permutating
        the parent target sentence.
        """
        full_hypo_len = len(candidate.trgt_sentence)
        max_pos = min(full_hypo_len-1, # -1 because we don't permit moving EOS
                      max_pos)
        self.explored.add(candidate.trgt_sentence[0:max_pos+1], True)
        if (explored_hypos 
                and explored_hypos[-1].trgt_sentence[-1] == utils.EOS_ID):
            full_hypo = explored_hypos[-1].generate_full_hypothesis()
            self.best_score = max(self.best_score, full_hypo.total_score)
            self.add_full_hypo(full_hypo)
        # Update data structures
        bigram_scores = copy.deepcopy(candidate.bigram_scores)
        for idx,hypo in enumerate(explored_hypos):
            prev_word = hypo.trgt_sentence[-1] if hypo.trgt_sentence \
                                               else utils.GO_ID
            scores = candidate.scores[0:len(hypo.trgt_sentence)]
            for w in self.full_bag:
                score = explored_posteriors[idx][w]
                # Update bigram scores
                # not quite correct if we have two occurrences of the same word
                bigram_scores[prev_word][w] = score 
                # Update tries
                exp_hypo = hypo.cheap_expand(w, 
                                             score, 
                                             explored_score_breakdowns[idx][w])
                exp_hypo.scores = scores + [score]
                self.hypos.add(exp_hypo.trgt_sentence, exp_hypo)
        # Extract candidates
        self._extract_candidates(candidate, max_pos, bigram_scores)

    def _extract_candidates_move(self, candidate, max_pos, bigram_scores):
        """Implements the traversal strategy 'move' """
        full_hypo_len = len(candidate.trgt_sentence)
        for from_pos in xrange(full_hypo_len-1):
            from_word = candidate.trgt_sentence[from_pos]
            stub = candidate.trgt_sentence[0:from_pos] \
                        + candidate.trgt_sentence[from_pos+1:]
            for to_pos in xrange(full_hypo_len-1):
                change_pos = min(from_pos, to_pos)
                if change_pos >= max_pos:
                    continue
                trgt_sentence = list(stub)
                trgt_sentence.insert(to_pos, from_word)
                if self._is_explored(trgt_sentence):
                    continue
                scores = list(candidate.scores)
                prev_word = trgt_sentence[change_pos-1] if change_pos > 0 \
                                                        else utils.GO_ID
                for idx in xrange(change_pos, full_hypo_len):
                    word = trgt_sentence[idx]
                    scores[idx] = bigram_scores[prev_word][word]
                    prev_word = word
                self._add_candidate(FlipCandidate(trgt_sentence, 
                                                  scores, 
                                                  bigram_scores, 
                                                  sum(scores[0:change_pos+1])))

    def _extract_candidates_flip(self, candidate, max_pos, bigram_scores):
        """Implements the traversal strategy 'flip' """
        full_hypo_len = len(candidate.trgt_sentence)
        for from_pos in xrange(max_pos):
            max_score = sum(candidate.scores[0:from_pos])
            from_word = candidate.trgt_sentence[from_pos]
            for to_pos in xrange(from_pos+1, full_hypo_len-1):
                trgt_sentence = list(candidate.trgt_sentence)
                trgt_sentence[from_pos] = trgt_sentence[to_pos]
                trgt_sentence[to_pos] = from_word
                if self._is_explored(trgt_sentence):
                    continue
                scores = list(candidate.scores)
                prev_word = trgt_sentence[from_pos-1] if from_pos > 0 \
                                                      else utils.GO_ID
                for idx in xrange(from_pos, full_hypo_len):
                    word = trgt_sentence[idx]
                    scores[idx] = bigram_scores[prev_word][word]
                    prev_word = word
                self._add_candidate(FlipCandidate(trgt_sentence, 
                                                  scores, 
                                                  bigram_scores, 
                                                  max_score+scores[from_pos]))
        
    def _add_candidate(self, candidate):
        """Add a candidate to the heap """
        if candidate.max_score > self.best_score:
            heappush(self.open_candidates, (-candidate.expected_score, 
                                            candidate))
    
    def _explore_candidate(self, candidate):
        """Explores a candidate, adds it to the list of full hypotheses
        (if not pruned by early stopping) and calls  
        ``_process_new_hypos`` to derive new candidates. 
        """
        prefix = self.hypos.get_prefix(candidate.trgt_sentence)
        hypo = self.hypos.get(prefix)
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        for pos,score in enumerate(hypo.scores): # Update candidate scores
            candidate.scores[pos] = score
        self.consume(hypo.word_to_consume) 
        hypos = []
        posteriors = []
        score_breakdowns = []
        for pos in xrange(len(prefix), len(candidate.trgt_sentence)):
            if self.early_stopping and hypo.score <= self.best_score:
                break # admissible pruning
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = copy.deepcopy(self.get_predictor_states())
            hypos.append(hypo)
            posteriors.append(posterior)
            score_breakdowns.append(score_breakdown)
            logging.debug("Explored: %s (%f)" % (
                          ' '.join([str(w) for w in hypo.trgt_sentence]),
                          hypo.score))
            word = candidate.trgt_sentence[pos]
            score = posterior[word]
            if self.always_greedy: # change sentence st best word is at this pos
                best_pos = pos
                stub = candidate.trgt_sentence[0:pos]
                for p in xrange(pos+1, len(candidate.trgt_sentence)-1):
                    if (posterior[candidate.trgt_sentence[p]] > score
                           and not self._is_explored(
                                        stub + [candidate.trgt_sentence[p]])):
                        best_pos = p
                        word = candidate.trgt_sentence[p]
                        score = posterior[word]  
                if best_pos != pos:
                    del candidate.trgt_sentence[best_pos]
                    del candidate.scores[best_pos]
                    candidate.trgt_sentence.insert(pos, word)
                    candidate.scores.insert(pos, score)
            self.consume(word)
            hypo = hypo.expand(word,
                               None,
                               score,
                               score_breakdown[word])
            candidate.scores[pos] = posterior[word]
        if hypo.trgt_sentence[-1] == utils.EOS_ID:
            self.best_score = max(self.best_score, hypo.score)
            self.add_full_hypo(hypo.generate_full_hypothesis())
        acc = 0.0
        for max_pos,score in enumerate(candidate.scores):
            acc += score
            if acc <= self.best_score:
                break
        self._process_new_hypos(candidate,
                                 max_pos,
                                 hypos,
                                 posteriors,
                                 score_breakdowns)
    
    def _load_bag(self):
        """Load the current bag of words """
        self.full_bag = {}
        for w in self.lines[self.current_sen_id].strip().split(): 
            int_w = int(w)
            self.full_bag[int_w] = self.full_bag.get(int_w, 0) + 1

    def decode(self, src_sentence):
        """Decodes a single source sentence with the flip decoder """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        self._load_bag()
        self.hypos = SimpleTrie()
        self.explored = SimpleTrie()
        self.open_candidates = []
        self.best_score = self.get_lower_score_bound()
        self._greedy_decode()
        while (self.open_candidates 
                        and self.max_expansions > self.apply_predictors_count):
            _,candidate = heappop(self.open_candidates)
            if candidate.max_score <= self.best_score:
                continue
            if self._is_explored(candidate.trgt_sentence): # Already explored
                continue
            logging.debug(
                "Best: %f Expected: %f Expansions: %d Open: %d Explore %s" % (
                          self.best_score,
                          candidate.expected_score,
                          self.apply_predictors_count,
                          len(self.open_candidates),
                          ' '.join([str(w) for w in candidate.trgt_sentence])))
            self._explore_candidate(candidate)
        return self.get_full_hypos_sorted()
