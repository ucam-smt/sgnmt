"""This module contains predictors for bag of words experiments. This
is the standard bow predictor and the bowsearch predictor which first
does an unrestricted search to construct a skeleton and then restricts
the order of words by that skeleton (in addition to the bag 
restriction).
"""

import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.core import CLOSED_VOCAB_SCORE_NORM_NONE
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.misc.unigram import FileUnigramTable, \
    BestStatsUnigramTable, FullStatsUnigramTable, AllStatsUnigramTable
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt.utils import INF, NEG_INF, MESSAGE_TYPE_FULL_HYPO, \
    MESSAGE_TYPE_DEFAULT


class BagOfWordsPredictor(Predictor):
    """This predictor is similar to the forced predictor, but it does
    not enforce the word order in the reference. Therefore, it assigns
    1 to all hypotheses which have the words in the reference in any 
    order, and -inf to all other hypos.
    """
    
    def __init__(self, 
                 trg_test_file, 
                 accept_subsets=False,
                 accept_duplicates=False,
                 heuristic_scores_file="",
                 collect_stats_strategy='best',
                 heuristic_add_consumed = False, 
                 heuristic_add_remaining = True,
                 diversity_heuristic_factor = -1.0,
                 equivalence_vocab=-1):
        """Creates a new bag-of-words predictor.
        
        Args:
            trg_test_file (string): Path to the plain text file with 
                                    the target sentences. Must have the
                                    same number of lines as the number
                                    of source sentences to decode. The 
                                    word order in the target sentences
                                    is not relevant for this predictor.
            accept_subsets (bool): If true, this predictor permits
                                       EOS even if the bag is not fully
                                       consumed yet
            accept_duplicates (bool): If true, counts are not updated
                                      when a word is consumed. This
                                      means that we allow a word in a
                                      bag to appear multiple times
            heuristic_scores_file (string): Path to the unigram scores 
                                            which are used if this 
                                            predictor estimates future
                                            costs
            collect_stats_strategy (string): best, full, or all. Defines 
                                             how unigram estimates are 
                                             collected for heuristic 
            heuristic_add_consumed (bool): Set to true to add the 
                                           difference between actual
                                           partial score and unigram
                                           estimates of consumed words
                                           to the predictor heuristic
            heuristic_add_remaining (bool): Set to true to add the sum
                                            of unigram scores of words
                                            remaining in the bag to the
                                            predictor heuristic
            diversity_heuristic_factor (float): Factor for diversity
                                                heuristic which 
                                                penalizes hypotheses
                                                with the same bag as
                                                full hypos
            equivalence_vocab (int): If positive, predictor states are
                                     considered equal if the the 
                                     remaining words within that vocab
                                     and OOVs regarding this vocab are
                                     the same. Only relevant when using
                                     hypothesis recombination
        """
        super(BagOfWordsPredictor, self).__init__()
        with open(trg_test_file) as f:
            self.lines = f.read().splitlines()
        if heuristic_scores_file:
            self.estimates = FileUnigramTable(heuristic_scores_file)
        elif collect_stats_strategy == 'best':
            self.estimates = BestStatsUnigramTable()
        elif collect_stats_strategy == 'full':
            self.estimates = FullStatsUnigramTable()
        elif collect_stats_strategy == 'all':
            self.estimates = AllStatsUnigramTable()
        else:
            logging.error("Unknown statistics collection strategy")
        self.accept_subsets = accept_subsets
        self.accept_duplicates = accept_duplicates
        self.heuristic_add_consumed = heuristic_add_consumed
        self.heuristic_add_remaining = heuristic_add_remaining
        self.equivalence_vocab = equivalence_vocab
        if accept_duplicates and not accept_subsets:
            logging.error("You enabled bow_accept_duplicates but not bow_"
                          "accept_subsets. Therefore, the bow predictor will "
                          "never accept end-of-sentence and could cause "
                          "an infinite loop in the search strategy.")
        self.diversity_heuristic_factor = diversity_heuristic_factor
        self.diverse_heuristic = (diversity_heuristic_factor > 0.0)
          
    def get_unk_probability(self, posterior):
        """Returns negative infinity unconditionally: Words which are
        not in the target sentence have assigned probability 0 by
        this predictor.
        """
        return NEG_INF
    
    def predict_next(self):
        """If the bag is empty, the only allowed symbol is EOS. 
        Otherwise, return the list of keys in the bag.
        """
        if not self.bag: # Empty bag
            return {utils.EOS_ID : 0.0}
        ret = {w : 0.0 for w in self.bag.iterkeys()}
        if self.accept_subsets:
            ret[utils.EOS_ID] = 0.0
        return ret
    
    def initialize(self, src_sentence):
        """Creates a new bag for the current target sentence..
        
        Args:
            src_sentence (list):  Not used
        """
        self.best_hypo_score = NEG_INF
        self.bag = {}
        for w in self.lines[self.current_sen_id].strip().split(): 
            int_w = int(w)
            self.bag[int_w] = self.bag.get(int_w, 0) + 1
        self.full_bag = dict(self.bag)
        
    def consume(self, word):
        """Updates the bag by deleting the consumed word.
        
        Args:
            word (int): Next word to consume
        """
        if word == utils.EOS_ID:
            self.bag = {}
            return
        if not word in self.bag:
            logging.warn("Consuming word which is not in bag-of-words!")
            return
        cnt = self.bag.pop(word)
        if cnt > 1 and not self.accept_duplicates:
            self.bag[word] = cnt - 1
    
    def get_state(self):
        """State of this predictor is the current bag """
        return self.bag
    
    def set_state(self, state):
        """State of this predictor is the current bag """
        self.bag = state

    def reset(self):
        """Empty method. """
        pass
    
    def initialize_heuristic(self, src_sentence):
        """Calls ``reset`` of the used unigram table with estimates
        ``self.estimates`` to clear all statistics from the previous
        sentence
        
        Args:
            src_sentence (list): Not used
        """
        self.estimates.reset()
        if self.diverse_heuristic:
            self.explored_bags = SimpleTrie()
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """This gets called if this predictor observes the decoder. It
        updates unigram heuristic estimates via passing through this
        message to the unigram table ``self.estimates``.
        """
        self.estimates.notify(message, message_type)
        if self.diverse_heuristic and message_type == MESSAGE_TYPE_FULL_HYPO:
            self._update_explored_bags(message)
    
    def _update_explored_bags(self, hypo):
        """This is called if diversity heuristic is enabled. It updates
        ``self.explored_bags``
        """
        sen = hypo.trgt_sentence
        for l in xrange(len(sen)):
            key = sen[:l]
            key.sort()
            cnt = self.explored_bags.get(key)
            if not cnt:
                cnt = 0.0
            self.explored_bags.add(key, cnt + 1.0)
                    
    def estimate_future_cost(self, hypo):
        """The bow predictor comes with its own heuristic function. We
        use the sum of scores of the remaining words as future cost 
        estimator. 
        """
        acc = 0.0
        if self.heuristic_add_remaining:
            remaining = dict(self.full_bag)
            remaining[utils.EOS_ID] = 1
            for w in hypo.trgt_sentence:
                remaining[w] -= 1
            acc -= sum([cnt*self.estimates.estimate(w) 
                            for w,cnt in remaining.iteritems()])
        if self.diverse_heuristic:
            key = list(hypo.trgt_sentence)
            key.sort()
            cnt = self.explored_bags.get(key)
            if cnt:
                acc += cnt * self.diversity_heuristic_factor
        if self.heuristic_add_consumed:
            acc -= hypo.score - sum([self.estimates.estimate(w, -1000.0)
                            for w in hypo.trgt_sentence])
        return acc
    
    def _get_unk_bag(self, org_bag):
        if self.equivalence_vocab <= 0:
            return org_bag
        unk_bag = {}
        for word,cnt in org_bag.iteritems():
            idx = word if word < self.equivalence_vocab else utils.UNK_ID
            unk_bag[idx] = unk_bag.get(idx, 0) + cnt
        return unk_bag
    
    def is_equal(self, state1, state2):
        """Returns true if the bag is the same """
        return self._get_unk_bag(state1) == self._get_unk_bag(state2) 


class BagOfWordsSearchPredictor(BagOfWordsPredictor):
    """Combines the bag-of-words predictor with a proxy decoding pass
    which creates a skeleton translation.
    """
    
    def __init__(self,
                 main_decoder,
                 hypo_recombination,
                 trg_test_file, 
                 accept_subsets=False,
                 accept_duplicates=False,
                 heuristic_scores_file="",
                 collect_stats_strategy='best',
                 heuristic_add_consumed = False, 
                 heuristic_add_remaining = True,
                 diversity_heuristic_factor = -1.0,
                 equivalence_vocab=-1):
        """Creates a new bag-of-words predictor with pre search
        
        Args:
            main_decoder (Decoder): Reference to the main decoder
                                    instance, used to fetch the predictors
            hypo_recombination (bool): Activates hypo recombination for the
                                       pre decoder 
            trg_test_file (string): Path to the plain text file with 
                                    the target sentences. Must have the
                                    same number of lines as the number
                                    of source sentences to decode. The 
                                    word order in the target sentences
                                    is not relevant for this predictor.
            accept_subsets (bool): If true, this predictor permits
                                       EOS even if the bag is not fully
                                       consumed yet
            accept_duplicates (bool): If true, counts are not updated
                                      when a word is consumed. This
                                      means that we allow a word in a
                                      bag to appear multiple times
            heuristic_scores_file (string): Path to the unigram scores 
                                            which are used if this 
                                            predictor estimates future
                                            costs
            collect_stats_strategy (string): best, full, or all. Defines 
                                             how unigram estimates are 
                                             collected for heuristic 
            heuristic_add_consumed (bool): Set to true to add the 
                                           difference between actual
                                           partial score and unigram
                                           estimates of consumed words
                                           to the predictor heuristic
            heuristic_add_remaining (bool): Set to true to add the sum
                                            of unigram scores of words
                                            remaining in the bag to the
                                            predictor heuristic
            equivalence_vocab (int): If positive, predictor states are
                                     considered equal if the the 
                                     remaining words within that vocab
                                     and OOVs regarding this vocab are
                                     the same. Only relevant when using
                                     hypothesis recombination
        """
        self.main_decoder = main_decoder
        self.pre_decoder = BeamDecoder(CLOSED_VOCAB_SCORE_NORM_NONE,
                                       main_decoder.max_len_factor,
                                       hypo_recombination,
                                       10)
        self.pre_decoder.combine_posteriors = main_decoder.combine_posteriors 
        super(BagOfWordsSearchPredictor, self).__init__(trg_test_file, 
                                                        accept_subsets,
                                                        accept_duplicates,
                                                        heuristic_scores_file,
                                                        collect_stats_strategy,
                                                        heuristic_add_consumed, 
                                                        heuristic_add_remaining,
                                                        diversity_heuristic_factor,
                                                        equivalence_vocab)
        self.pre_mode = False
    
    def predict_next(self):
        """If in ``pre_mode``, pass through to super class. Otherwise,
        scan skeleton 
        """
        if self.pre_mode:
            return super(BagOfWordsSearchPredictor, self).predict_next()
        if not self.bag: # Empty bag
            return {utils.EOS_ID : 0.0}
        ret = {w : 0.0 for w in self.missing.iterkeys()}
        if self.accept_subsets:
            ret[utils.EOS_ID] = 0.0
        if self.skeleton_pos < len(self.skeleton):
            ret[self.skeleton[self.skeleton_pos]] = 0.0
        return ret
    
    def initialize(self, src_sentence):
        """If in ``pre_mode``, pass through to super class. Otherwise,
        initialize skeleton. 
        """
        if self.pre_mode:
            return super(BagOfWordsSearchPredictor, self).initialize(src_sentence)
        self.pre_mode = True
        old_accept_subsets = self.accept_subsets
        old_accept_duplicates = self.accept_duplicates
        self.accept_subsets = True
        self.accept_duplicates = True
        self.pre_decoder.predictors = self.main_decoder.predictors
        self.pre_decoder.current_sen_id = self.main_decoder.current_sen_id - 1
        hypos = self.pre_decoder.decode(src_sentence)
        score = INF
        if not hypos:
            logging.warn("No hypothesis found by the pre decoder. Effectively "
                         "reducing bowsearch predictor to bow predictor.")
            self.skeleton = []
        else:
            self.skeleton = hypos[0].trgt_sentence
            score = hypos[0].total_score
            if self.skeleton and self.skeleton[-1] -- utils.EOS_ID:
                self.skeleton = self.skeleton[:-1] # Remove EOS
        self.skeleton_pos = 0
        self.accept_subsets = old_accept_subsets
        self.accept_duplicates = old_accept_duplicates
        self._set_up_full_mode()
        logging.debug("BOW Skeleton (score=%f missing=%d): %s" % (
                                          score,
                                          sum(self.missing.values()),
                                          self.skeleton))
        self.main_decoder.current_sen_id -= 1
        self.main_decoder.initialize_predictors(src_sentence)
        self.pre_mode = False
    
    def _set_up_full_mode(self):
        """This method initializes ``missing`` by using
        ``self.skeleton`` and ``self.full_bag`` and removes
        duplicates from ``self.skeleton``.
        """
        self.bag = dict(self.full_bag)
        missing = dict(self.full_bag)
        skeleton_no_duplicates = []
        for word in self.skeleton:
            if missing[word] > 0:
                missing[word] -= 1
                skeleton_no_duplicates.append(word)
        self.skeleton = skeleton_no_duplicates
        self.missing = {w: cnt for w, cnt in missing.iteritems() if cnt > 0}
        
    def consume(self, word):
        """Calls super class ``consume``. If not in ``pre_mode``,
        update skeleton info. 
        
        Args:
            word (int): Next word to consume
        """
        super(BagOfWordsSearchPredictor, self).consume(word)
        if self.pre_mode:
            return
        if (self.skeleton_pos < len(self.skeleton) 
                 and word == self.skeleton[self.skeleton_pos]):
            self.skeleton_pos += 1
        elif word in self.missing:
            self.missing[word] -= 1
            if self.missing[word] <= 0:
                del self.missing[word]
    
    def get_state(self):
        """If in pre_mode, state of this predictor is the current bag
        Otherwise, its the bag plus skeleton state
        """
        if self.pre_mode:
            return super(BagOfWordsSearchPredictor, self).get_state()
        return self.bag, self.skeleton_pos, self.missing
    
    def set_state(self, state):
        """If in pre_mode, state of this predictor is the current bag
        Otherwise, its the bag plus skeleton state
        """
        if self.pre_mode:
            return super(BagOfWordsSearchPredictor, self).set_state(state)
        self.bag, self.skeleton_pos, self.missing = state
    
    def is_equal(self, state1, state2):
        """Returns true if the bag and the skeleton states are the same
        """
        if self.pre_mode:
            return super(BagOfWordsSearchPredictor, self).is_equal(state1, 
                                                                   state2)
        return super(BagOfWordsSearchPredictor, self).is_equal(state1[0], 
                                                               state2[0])
        
