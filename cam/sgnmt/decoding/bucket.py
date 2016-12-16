"""Implementation of the bucket search strategy """

import copy
import logging
import operator

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
from cam.sgnmt.utils import INF, NEG_INF
import numpy as np


class BucketDecoder(Decoder):
    """The bucket decoder maintains separate buckets for each sentence
    length. The buckets contain partial hypotheses. In each iteration,
    the decoder selects a bucket, and expands the best hypothesis in
    this bucket by one token. The core of the bucket decoder is the
    bucket selection strategy. The following strategies are available:
    
    * 'iter': Puts all buckets in a big loop and iterates through it.
              With this strategy, the number of hypothesis expansions
              is equally distributed over the buckets
    * 'random': (with stochastic=true and bucket_selecto!=difference)
                Randomly select a non-empty bucket
    * 'difference': Similar to the heuristic used by the restarting
                    decoder. Select the bucket in which the difference
                    between best and second best hypothesis is minimal
    * 'maxdiff': Like 'iter', but filters buckets in which the 
                 difference between first and second hypo is larger
                 than epsilon. If no such buckets exist, increase 
                 epsilon
    """
    
    def __init__(self, 
                 decoder_args,
                 hypo_recombination, 
                 max_expansions=0,
                 low_memory_mode = True,
                 beam=1,
                 pure_heuristic_scores = False,
                 diversity_factor = -1.0,
                 early_stopping=True,
                 stochastic=False,
                 bucket_selector='maxscore',
                 bucket_score_strategy='difference',
                 collect_stats_strategy='best'):
        """Create a new bucket decoder
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (boolean): Activates hypothesis 
                                          recombination. Hypos are
                                          tested only within a bucket 
            max_expansions (int): Maximum number of node expansions for
                                  inadmissible pruning.
            low_memory_mode (bool): Switch on low memory mode at cost 
                                    of some computational overhead.
                                    Limits the number of hypotheses in
                                    each bucket to the number of 
                                    remaining node expansions
            beam (int): Number of hypotheses which get expanded at once
                        after selecting a bucket.
            pure_heuristic_scores (bool): If false, hypos are scored 
                                          with partial score plus
                                          future cost estimates. If
                                          true, only the future cost
                                          estimates are used
            diversity_factor (float): If this is set to a positive 
                                      value, we reorder hypos in a 
                                      bucket by adding a term which 
                                      counts how many hypos with the
                                      same parent have been expanded
                                      already
            early_stopping (boolean): Admissible pruning (works only if
                                      scores are non-positive)
            stochastic (boolean): Stochastic bucket selection. If the 
                                  bucket selector is not 'difference',
                                  this results in random bucket 
                                  selection. If ``bucket_selector`` is
                                  set to 'difference', buckets are 
                                  randomly selected with probability
                                  according the bucket score
            bucket_selector (string): Bucket selection strategy. 'iter',
                                      'maxscore'. 'score'. See the
                                      class docstring for more info
            bucket_score_strategy (string): Determines the way the 
                                            buckets are scored.
                                            'difference' between current
                                            best word score and best 
                                            hypo in bucket, 'absolute'
                                            hypo score, 'heap' score of
                                            top scoring hypo in bucket ,
                                            'constant' score of 0.0.
            collect_stats_strategy (string): best, full, or all. Defines 
                                             how unigram estimates are 
                                             collected for heuristic
                                            
        """
        super(BucketDecoder, self).__init__(decoder_args)
        self.max_expansions_param = max_expansions
        self.low_memory_mode = low_memory_mode
        self.beam = beam
        self.pure_heuristic_scores = pure_heuristic_scores
        self.diversity_factor = diversity_factor
        self.diverse_decoding = (diversity_factor > 0.0)
        self.early_stopping = early_stopping
        if stochastic:
            if bucket_selector == 'score':
                self.get_bucket = self._get_bucket_stochastic
            else:
                self.get_bucket = self._get_bucket_random
        else:
            if 'iter' in bucket_selector:
                self.get_bucket = self._get_bucket_iter
                self.max_iter = 1000000
                if '-' in bucket_selector:
                    _,n = bucket_selector.split("-")
                    self.max_iter = int(n)
            elif bucket_selector == 'maxscore':
                self.get_bucket = self._get_bucket_maxscore
            elif bucket_selector == 'score-end':
                self.get_bucket = self._get_bucket_score_end
            elif bucket_selector == 'score':
                self.get_bucket = self._get_bucket_score
            else:
                logging.fatal("Unknown bucket selector")
        if bucket_score_strategy == 'difference':
            self.get_bucketscore = self._get_bucketscore_difference
        elif bucket_score_strategy == 'heap':
            self.get_bucketscore = self._get_bucketscore_heap
        elif bucket_score_strategy == 'absolute':
            self.get_bucketscore = self._get_bucketscore_absolute
        elif bucket_score_strategy == 'constant':
            self.get_bucketscore = self._get_bucketscore_constant
        else:
            logging.fatal("Unknown bucket score strategy")
        self.collect_stats_from_partial = (collect_stats_strategy == 'all')
        if collect_stats_strategy == 'best':
            self.collect_stats = self._collect_stats_best
        elif collect_stats_strategy == 'full':
            self.collect_stats = self._collect_stats_full
        self.hypo_recombination = hypo_recombination
        
    def _get_bucketscore_difference(self, length):
        return self.best_word_scores[length] - self.buckets[length][0][1].score
    
    def _get_bucketscore_heap(self, length):
        return self.buckets[length][0][0]
    
    def _get_bucketscore_absolute(self, length):
        return -self.buckets[length][0][1].score
    
    def _get_bucketscore_constant(self, length):
        return 0.0

    def _get_bucket_iter(self):
        """Implements the bucket selector 'iter' """
        if self.cur_iter > self.max_iter:
            if self.guaranteed_optimality:
                logging.info("max iter. Optimality not guaranteed for ID %d" %
                      (self.current_sen_id + 1))
            self.guaranteed_optimality = False
            return -1
        last_length = self.last_bucket
        for length in xrange(last_length+1, self.max_len):
            if self.buckets[length]:
                self.last_bucket = length
                return length
        # Restart with first bucket
        self.cur_iter += 1
        for length in xrange(last_length+1):
            if self.buckets[length]:
                self.last_bucket = length
                return length
        return -1

    def _get_bucket_maxscore(self):
        """Implements the bucket selector 'maxscore' """
        for max_score in range(0, 500, 5):
            length = self._get_bucket_maxscore_helper(max_score) 
            if length >= 0:
                return length
        return -1

    def _get_bucket_maxscore_helper(self, max_score):
        """Helper method for maxscore """
        last_length = self.last_bucket
        for length in xrange(last_length+1, self.max_len):
            if self.buckets[length]:
                score = self.get_bucketscore(length)
                if score < max_score:
                    self.last_bucket = length
                    return length
        for length in xrange(last_length+1):
            if self.buckets[length]:
                score = self.get_bucketscore(length)
                if score < max_score:
                    self.last_bucket = length
                    return length
        return -1

    def _get_bucket_score(self):
        """Implements the bucket selector 'score' """
        best_score = INF
        best_length = -1
        for length in xrange(self.max_len):
            if self.buckets[length]:
                score = self.get_bucketscore(length)
                if score <= best_score:
                    best_score = score
                    best_length = length 
        return best_length

    def _get_bucket_score_end(self):
        """Implements the bucket selector 'score-end' """
        last_length = self.last_bucket
        for length in xrange(last_length+1, self.max_len):
            if self.buckets[length]:
                self.last_bucket = length
                return length
        # Restart with best bucket
        best_score = INF
        best_length = -1
        for length in xrange(self.max_len):
            if self.buckets[length]:
                score = self.get_bucketscore(length)
                if score <= best_score:
                    best_score = score
                    best_length = length 
        self.last_bucket = best_length
        return best_length

    def _get_bucket_random(self):
        """Implements random bucket selection """
        lengths = [l for l in xrange(self.max_len) if self.buckets[l]]
        return np.random.choice(lengths)
        
    def _get_bucket_stochastic(self):
        """Implements the stochastic bucket selector 'difference' """
        lengths = []
        scores = []
        for length in xrange(self.max_len):
            if self.buckets[length]:
                score = self.get_bucketscore(length)
                if score == NEG_INF:
                    return self._get_bucket_difference()
                lengths.append(length)
                scores.append(score)
        if not lengths:
            return -1
        exps = np.exp([-d for d in scores])
        total = sum(exps)
        return np.random.choice(lengths, p=[e/total for e in exps])
    
    def _initialize_decoding(self, src_sentence):
        """Helper function for ``decode`` to which initializes all the
        class attributes
        """
        self.initialize_predictors(src_sentence)
        self.max_expansions = self.get_max_expansions(self.max_expansions_param,
                                                      src_sentence) 
        init_hypo = PartialHypothesis()
        init_hypo.predictor_states = self.get_predictor_states()
        init_hypo.scores = []
        init_hypo.parent_hypo_array_idx = 0 # point to guardian
        self.buckets = [[] for _ in xrange(self.max_len+1)]
        self.expanded_hypos = [[] for _ in xrange(self.max_len+1)]
        self.buckets[0].append((0.0, init_hypo))
        self.expand_counts = [0.0] # with guardian
        self.expand_backpointers = [0] # with guardian
        self.last_bucket = 0
        self.best_score = self.get_lower_score_bound()
        self.best_word_scores = [NEG_INF] * (self.max_len+1)
        self.compressed = [True] * (self.max_len+1)
        self.guaranteed_optimality = True
        self.cur_iter = 0
    
    def _activate_hypo(self, hypo, length, heap_score):
        """Prepares the decoder for expanding the given hypothesis. 
        This may include updating global word scores, loading the
        predictor states and consume the last word of the hypothesis
        if necessary. After this method, ``apply_predictors`` computes
        the next posterior vector
        """
        if (self.collect_stats_from_partial 
                and hypo.score > self.best_word_scores[length]):
            self.best_word_scores[length] = hypo.score
            self._update_heap_scores()
        logging.debug("Expand (best_glob=%f diff=%f (%f-%f) heap=%f exp=%d): %s"
                            % (self.best_score,
                               self.best_word_scores[length] - hypo.score,
                               self.best_word_scores[length],
                               hypo.score,
                               heap_score,
                               self.apply_predictors_count,
                               ' '.join([str(w) for w in hypo.trgt_sentence])))
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
    
    def _get_next_hypo_array_idx(self, hypo):
        """Used to assign a new hypothesis array index used in 
        ``expand_counts`` and ``expand_backpointers``. This also 
        updates the count and backpointer array. Only called if
        ``diverse_decoding`` is true.
        """
        idx = hypo.parent_hypo_array_idx
        while idx != 0:
            self.expand_counts[idx] += 1.0
            idx = self.expand_backpointers[idx]
        hypo_array_idx = len(self.expand_counts)
        self.expand_counts.append(0.0)
        self.expand_backpointers.append(hypo.parent_hypo_array_idx)
        return hypo_array_idx
    
    def _collect_stats_best(self, hypo):
        if hypo.score > self.best_score:
            self.best_score = hypo.score
            self.best_word_scores[:len(hypo.scores) ] = hypo.scores
            self._update_heap_scores()
    
    def _collect_stats_full(self, hypo):
        update = False
        for idx, score in enumerate(hypo.scores):
            if score > self.best_word_scores[idx]:
                self.best_word_scores[idx] = score
                update = True
        if update:
            self._update_heap_scores()
    
    def _register_full_hypo(self, hypo):
        """Update all class attributes according a new full hypothesis.
        ``hypo`` is a ``PartialHypothesis`` ending with EOS.
        """
        self.add_full_hypo(hypo.generate_full_hypothesis())
        if hypo.score > self.best_score:
            logging.info("New_best (ID: %d): score=%f exp=%d hypo=%s" 
                            % (self.current_sen_id + 1,
                               hypo.score, 
                               self.apply_predictors_count,
                               ' '.join([str(w) for w in hypo.trgt_sentence])))
        if not self.collect_stats_from_partial:
            self.collect_stats(hypo)
        if hypo.score > self.best_score:
            self.best_score = hypo.score
    
    def _update_heap_scores(self):
        """Called when ``best_word_scores`` has changed and heap scores
        need to be updated
        """
        for length in xrange(len(self.buckets)):
            self._update_heap_score(length)
    
    def _update_heap_score(self, length):
        """``_update_heap_score`` for a single bucket
        """
        new_bucket = [(-self._get_combined_score(h), h) 
                                for _,h in self.buckets[length]]
        self.buckets[length] = new_bucket
        self.buckets[length].sort(key=operator.itemgetter(0))
    
    def _get_max_bucket_size(self):
        if not self.low_memory_mode:
            return 10000000
        max_size = max(1, 1 + self.max_expansions - self.apply_predictors_count)
        if self.hypo_recombination:
            max_size = 5 * max_size
        return max_size
    
    def _add_new_hypos_to_bucket(self, length, new_hypos):
        max_size = self._get_max_bucket_size()
        if self.diverse_decoding:
            self._update_heap_score(length)
        if (not self.hypo_recombination or
                not self.guaranteed_optimality or 
                max_size >= len(new_hypos) + len(self.buckets[length])):
            self.buckets[length].extend(new_hypos)
            self.buckets[length].sort(key=operator.itemgetter(0))
            self.buckets[length] = self.buckets[length][:max_size]
            self.compressed[length] = False
        elif self.compressed[length]: # Equivalence check only for new
            logging.debug("Add %d hypos to compressed bucket of size %d" % (
                                    len(new_hypos), len(self.buckets[length])))
            new_hypos.sort(key=operator.itemgetter(0))
            new_bucket = []
            oidx = 0
            nidx = 0
            olen = len(self.buckets[length])
            nlen = len(new_hypos)
            while len(new_bucket) < max_size:
                oscore = INF if oidx >= olen else self.buckets[length][oidx][0]
                nscore = INF if nidx >= nlen else new_hypos[nidx][0]
                if oscore == INF and nscore == INF:
                    break
                if oscore < nscore: # Add hypos from old bucket without checks
                    new_bucket.append(self.buckets[length][oidx])
                    oidx += 1
                else: # Check equivalence
                    hypo = new_hypos[nidx][1]
                    self.set_predictor_states(copy.deepcopy(
                                                    hypo.predictor_states))
                    if not hypo.word_to_consume is None:
                        self.consume(hypo.word_to_consume)
                        hypo.word_to_consume = None
                    hypo.predictor_states = self.get_predictor_states()
                    valid = True
                    for other_hypo in [b for _,b in new_bucket]:
                        if other_hypo.score >= hypo.score and self.are_equal_predictor_states(
                                                hypo.predictor_states,
                                                other_hypo.predictor_states):
                            valid = False
                            logging.debug("Hypo recombination: %s > %s (compress)"
                                                  % (other_hypo.trgt_sentence,
                                                     hypo.trgt_sentence))
                            break
                    if valid:
                        new_bucket.append((nscore, hypo))
                    nidx += 1
            self.buckets[length] = new_bucket
            self.compressed[length] = True
        else: # Compress from scratch
            hypos = self.buckets[length] + new_hypos
            logging.debug("Compress bucket of size %d" % len(hypos))
            new_hypos.sort(key=operator.itemgetter(0))
            new_bucket = []
            idx = 0
            while len(new_bucket) < max_size and idx < len(hypos):
                hypo = hypos[idx][1]
                self.set_predictor_states(copy.deepcopy(
                                                    hypo.predictor_states))
                if not hypo.word_to_consume is None:
                    self.consume(hypo.word_to_consume)
                    hypo.word_to_consume = None
                hypo.predictor_states = self.get_predictor_states()
                valid = True
                for other_hypo in [b for _,b in new_bucket]:
                    if other_hypo.score >= hypo.score and self.are_equal_predictor_states(
                                                hypo.predictor_states,
                                                other_hypo.predictor_states):
                        valid = False
                        logging.debug("Hypo recombination: %s > %s"
                                                  % (other_hypo.trgt_sentence,
                                                     hypo.trgt_sentence))
                        break
                if valid:
                    new_bucket.append((hypos[idx][0], hypo))
                idx += 1
            self.buckets[length] = new_bucket
            self.compressed[length] = True
        if (self.hypo_recombination
                and self.guaranteed_optimality 
                and len(self.buckets[length]) >= max_size):
            logging.info("Shrunk bucket. Optimality not guaranteed for ID %d" %
                      (self.current_sen_id + 1))
            self.guaranteed_optimality = False

    def _get_combined_score(self, hypo):
        est_score = -self.estimate_future_cost(hypo)
        if self.diverse_decoding:
            cnt = 0.0
            idx = hypo.parent_hypo_array_idx
            while idx != 0:
                cnt += self.expand_counts[idx]
                idx = self.expand_backpointers[idx]
            est_score -= self.diversity_factor * cnt
        if not self.pure_heuristic_scores:
            est_score += hypo.score
            if self.best_score != NEG_INF:
                est_score -= self.best_score
        return est_score
    
    def _get_min_bucket_score(self, length):
        max_bucket_size = self._get_max_bucket_size() 
        if len(self.buckets[length]) >= max_bucket_size:
            return -self.buckets[length][max_bucket_size-1][0]
        return NEG_INF
    
    def _get_hypo(self, length):
        hypo = None
        while self.buckets[length] and hypo is None:
            s,hypo = self.buckets[length].pop(0)
            if self.early_stopping and hypo.score <= self.best_score:
                hypo = None
            else:
                self._activate_hypo(hypo, length, s)
                if self.hypo_recombination:
                    hypo.predictor_states = self.get_predictor_states()
                    for other_hypo in self.expanded_hypos[length]:
                        if other_hypo.score >= hypo.score and self.are_equal_predictor_states(
                                                hypo.predictor_states,
                                                other_hypo.predictor_states):
                            logging.debug("Hypo recombination: %s > %s (activate)"
                                                  % (other_hypo.trgt_sentence,
                                                     hypo.trgt_sentence))
                            hypo = None
                            break
                    if not hypo is None:
                        self.expanded_hypos[length].append(hypo)
        return hypo

    def decode(self, src_sentence):
        """Decodes a single source sentence. 
        """
        self._initialize_decoding(src_sentence)
        while self.max_expansions > self.apply_predictors_count:
            length = self.get_bucket()
            if length < 0: # No more full buckets
                break
            min_next_bucket_score = self._get_min_bucket_score(length+1)
            hypos_to_add = []
            for _ in xrange(self.beam): # Expand beam_size hypos in this bucket
                hypo = self._get_hypo(length)
                if hypo is None:
                    break
                posterior,score_breakdown = self.apply_predictors()
                hypo.predictor_states = self.get_predictor_states()
                if self.diverse_decoding:
                    hypo_array_idx = self._get_next_hypo_array_idx(hypo)
                for w,score in posterior.iteritems():
                    exp_hypo = hypo.cheap_expand(w, score, score_breakdown[w])
                    exp_hypo.scores = hypo.scores + [hypo.score] 
                    if self.diverse_decoding:
                        exp_hypo.parent_hypo_array_idx = hypo_array_idx
                    combi_score = self._get_combined_score(exp_hypo) 
                    if w == utils.EOS_ID:
                        self._register_full_hypo(exp_hypo)
                    elif (combi_score >= min_next_bucket_score
                          and (exp_hypo.score > self.best_score 
                               or not self.early_stopping)):
                        hypos_to_add.append((-combi_score, exp_hypo))
            self._add_new_hypos_to_bucket(length+1, hypos_to_add)
        if self.guaranteed_optimality and self.max_expansions <= self.apply_predictors_count:
            logging.info("Reached max_node_expansions. Optimality not guaranteed for ID %d" %
                      (self.current_sen_id + 1))
        if not self.full_hypos: # Add incomplete longest hypos if no complete
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in self.buckets[self.max_len]:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()
