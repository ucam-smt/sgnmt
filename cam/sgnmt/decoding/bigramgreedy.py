"""Implementation of the bigram greedy search strategy """

import copy
import logging
import operator

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
from cam.sgnmt.misc.trie import SimpleTrie


class BigramGreedyDecoder(Decoder):
    """The bigram greedy decoder collects bigram statistics at each
    node expansions. After each decoding pass, it constructs a new
    hypothesis to rescore by greedily selecting bigrams and gluing
    them together. Afterwards, the new hypothesis is rescored and new
    bigram statistics are collected.
    
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
                 early_stopping=True):
        """Creates a new bigram greedy decoder. Do not use this decoder
        in combination with the bow predictor as it inherently already
        satisfies the bag-of-word constrains.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            trg_test_file (string): Path to a plain text file which 
                                    defines the bag of words
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
            early_stopping (boolean): Activates admissible pruning
        """
        super(BigramGreedyDecoder, self).__init__(decoder_args)
        self.max_expansions_param = max_expansions
        self.early_stopping = early_stopping
        with open(trg_test_file) as f:
            self.lines = f.read().splitlines()
    
    def _greedy_decode(self):
        """Performs greedy decoding from the start node. Used to obtain
        initial bigram statistics.
        """
        hypo = PartialHypothesis()
        hypos = []
        posteriors = []
        score_breakdowns = []
        bag = dict(self.full_bag)
        while bag:
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = copy.deepcopy(self.get_predictor_states())
            bag_posterior = {w: posterior[w] for w in self.full_bag_with_eos}
            bag_breakdown = {w: score_breakdown[w] 
                                        for w in self.full_bag_with_eos}
            posteriors.append(bag_posterior)
            score_breakdowns.append(bag_breakdown)
            hypos.append(hypo)
            best_word = utils.argmax({w: bag_posterior[w] for w in bag})
            bag[best_word] -= 1
            if bag[best_word] < 1:
                del bag[best_word]
            self.consume(best_word)
            hypo = hypo.expand(best_word,
                               None,
                               bag_posterior[best_word],
                               score_breakdown[best_word])
        posterior,score_breakdown = self.apply_predictors()
        hypo.predictor_states = copy.deepcopy(self.get_predictor_states())
        bag_posterior = {w: posterior[w] for w in self.full_bag_with_eos}
        bag_breakdown = {w: score_breakdown[w] for w in self.full_bag_with_eos}
        posteriors.append(bag_posterior)
        score_breakdowns.append(bag_breakdown)
        hypos.append(hypo)
        
        hypo = hypo.cheap_expand(utils.EOS_ID,
                                 bag_posterior[utils.EOS_ID],
                                 score_breakdown[utils.EOS_ID])
        logging.debug("Greedy hypo (%f): %s" % (
                          hypo.score,
                          ' '.join([str(w) for w in hypo.trgt_sentence])))
        self._process_new_hypos(hypos, posteriors, score_breakdowns, hypo)
    
    def _process_new_hypos(self,
                           hypos,
                           posteriors, 
                           score_breakdowns, 
                           complete_hypo = None):
        """This method is called after a decoding pass. It updates 
        bigram statistics, stores partial hypotheses for restarting
        from them later, and creates full hypotheses if a hypo
        ends with EOS
        """
        if complete_hypo:
            self.best_score = max(self.best_score, complete_hypo.score)
            self.add_full_hypo(complete_hypo.generate_full_hypothesis())
        for idx,hypo in enumerate(hypos):
            posterior = posteriors[idx]
            prefix = hypo.trgt_sentence
            self._register_bigram_scores(prefix[-1] if prefix else utils.GO_ID, 
                                         posterior)
            self.posteriors.add(prefix, posterior)
            self.score_breakdowns.add(prefix, score_breakdowns[idx])
            self.hypos.add(prefix, hypo)
        self._sort_bigram_scores()
    
    def _get_next_sentence(self):
        """Get the next sentence to rescore
        """
        bag0 = dict(self.full_bag)
        bag0[utils.GO_ID] = 1
        bag1 = dict(self.full_bag_with_eos)
        return self._get_next_sentence_recursive([], bag0, bag1)
    
    def _get_next_sentence_recursive(self,
                                     bigrams,
                                     remaining_bag0,
                                     remaining_bag1):
        """Recursive helper function for _get_next_sentence
        
        Args:
            bigrams (list): List of already selected bigrams
            remaining_bag0 (dict): Remaining words in the bag for the
                                   first word in the bigram
            remaining_bag1 (dict): Remaining words in the bag for the
                                   second word in the bigram
        
        Returns:
            Tuple. hypo, sen tuple where sen is an unexplored sentence
            and hypo corresponds to the largest explored prefix of sen.
            Returns None if no consistent sentence was found
        """
        if len(bigrams) == self.num_words + 1: # Collected enough bigrams
            sens = self._get_sentences_from_bigrams(bigrams)
            if not sens: # Bigrams are not consistent
                return None
            for sen in sens:
                hypo = self._get_largest_prefix_hypo(sen)
                if hypo and hypo.score > self.best_score:
                    return hypo, sen
            return None
        for bigram in self.sorted_bigrams:
            if remaining_bag0[bigram[0]] > 0 and remaining_bag1[bigram[1]] > 0:
                remaining_bag0[bigram[0]] -= 1
                remaining_bag1[bigram[1]] -= 1
                ret = self._get_next_sentence_recursive(bigrams + [bigram],
                                                        remaining_bag0,
                                                        remaining_bag1)
                if ret:
                    return ret
                remaining_bag0[bigram[0]] += 1
                remaining_bag1[bigram[1]] += 1
        return None
    
    def _get_largest_prefix_hypo(self, sen):
        """Get the explored hypothesis with the largest common prefix
        with ``sen``.
        """
        prefix = self.hypos.get_prefix(sen)
        if len(prefix) == len(sen): # hypo is already fully explored
            return None
        hypo = self.hypos.get(prefix)
        posterior = self.posteriors.get(prefix)
        score_breakdown = self.score_breakdowns.get(prefix)
        next_word = sen[len(prefix)]
        return hypo.cheap_expand(next_word,
                                 posterior[next_word],
                                 score_breakdown[next_word])
    
    def _get_sentences_from_bigrams(self, bigrams):
        """Constructs all full consistent sentences from a list of 
        bigrams. The search is implemented as BFS. """
        candidates = [([utils.GO_ID], bigrams)]
        for _ in xrange(len(bigrams)):
            next_candidates = []
            for candidate in candidates:
                # Select the next consistent bigram
                cand_sen,cand_bigrams = candidate 
                last_word = cand_sen[-1]
                for idx,bigram in enumerate(cand_bigrams):
                    if bigram[0] == last_word: # Consistent
                        new_bigrams = list(cand_bigrams)
                        del new_bigrams[idx]
                        next_candidates.append((cand_sen + [bigram[1]],
                                                new_bigrams))
            candidates = next_candidates
            if not candidates:
                break
        return [candidate[0][1:] for candidate in candidates]
    
    def _forced_decode(self, start_hypo, sen):
        """Performs forced decoding from a the node in the search tree.
        
        Args:
            start_hypo (PartialHypothesis): This is a partial hypothesis
                                            for a prefix of sen from 
                                            which we start decoding
            sen (list): Sentence to rescore
        """
        logging.debug("best=%f prefix=%s prefix_score=%f sen=%s" % (
                                                 self.best_score, 
                                                 start_hypo.trgt_sentence, 
                                                 start_hypo.score, 
                                                 sen))
        self.set_predictor_states(copy.deepcopy(start_hypo.predictor_states))
        if not start_hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(start_hypo.word_to_consume)
        hypos = []
        posteriors = []
        score_breakdowns = []
        hypo = start_hypo
        cancelled = False
        for forced_w in sen[len(start_hypo.trgt_sentence):]:
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = copy.deepcopy(self.get_predictor_states())
            bag_posterior = {w: posterior[w] for w in self.full_bag_with_eos}
            bag_breakdown = {w: score_breakdown[w] 
                                        for w in self.full_bag_with_eos}
            posteriors.append(bag_posterior)
            score_breakdowns.append(bag_breakdown)
            hypos.append(hypo)
            hypo = hypo.expand(forced_w,
                               None,
                               bag_posterior[forced_w],
                               score_breakdown[forced_w])
            if self.early_stopping and hypo.score < self.best_score:
                cancelled = True
                break
            self.consume(forced_w)
        self._process_new_hypos(hypos, posteriors, score_breakdowns,
                                hypo if not cancelled else None)
    
    def _load_bag(self):
        """Load the current bag of words """
        self.full_bag = {}
        for w in self.lines[self.current_sen_id].strip().split(): 
            int_w = int(w)
            self.full_bag[int_w] = self.full_bag.get(int_w, 0) + 1
        self.num_words = sum(self.full_bag.itervalues())
        self.full_bag_with_eos = dict(self.full_bag)
        self.full_bag_with_eos[utils.EOS_ID] = 1
    
    def _register_bigram_scores(self, last_word, posterior):
        for w,score in posterior.iteritems():
            self.bigram_scores[last_word][w] = min(
                                    self.bigram_scores[last_word][w], score)
    
    def _sort_bigram_scores(self):
        self.sorted_bigrams = []
        for w1,scores in self.bigram_scores.iteritems():
            self.sorted_bigrams.extend([(w1, w2, score)
                                        for w2,score in scores.iteritems()])
        self.sorted_bigrams.sort(key=operator.itemgetter(2), reverse=True)
    
    def _initialize_bigram_scores(self):
        default_scores = {w: 0.0 for w in self.full_bag_with_eos}
        self.bigram_scores = {w: dict(default_scores) for w in self.full_bag}
        self.bigram_scores[utils.GO_ID] = default_scores

    def decode(self, src_sentence):
        """Decodes a single source sentence with the flip decoder """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        self._load_bag()
        self.hypos = SimpleTrie()
        self.posteriors = SimpleTrie()
        self.score_breakdowns = SimpleTrie()
        self.best_score = self.get_lower_score_bound()
        self._initialize_bigram_scores()
        self._greedy_decode()
        while self.max_expansions > self.apply_predictors_count:
            ret = self._get_next_sentence()
            if not ret:
                break
            self._forced_decode(ret[0], ret[1])
        return self.get_full_hypos_sorted()
        