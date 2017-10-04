"""This module contains predictors that deal wit the length of the
target sentence. The ``NBLengthPredictor`` assumes a negative binomial
distribution on the target sentence lengths, where the parameters r and
p are linear combinations of features extracted from the source 
sentence. The ``WordCountPredictor`` adds the number of words as cost,
which can be used to prevent hypotheses from getting to short when 
using a language model.
"""

import logging
import math
from scipy.misc import logsumexp
from scipy.special import gammaln

from cam.sgnmt import utils
from cam.sgnmt.misc.trie import SimpleTrie
from cam.sgnmt.predictors.core import Predictor
import numpy as np


NUM_FEATURES = 5
EPS_R = 0.1;
EPS_P = 0.00001;


class NBLengthPredictor(Predictor):
    """This predictor assumes that target sentence lengths are 
    distributed according a negative binomial distribution with 
    parameters r,p. r is linear in features, p is the logistic of a
    linear function over the features. Weights can be trained using 
    the Matlab script ``estimate_length_model.m`` 
    
    Let w be the model_weights. All features are extracted from the
    src sentence::
    
      r = w0 * #char
      + w1 * #words
      + w2 * #punctuation
      + w3 * #char/#words
      + w4 * #punct/#words
      + w10
      
      p = logistic(w5 * #char
      + w6 * #words
      + w7 * #punctuation
      + w8 * #char/#words
      + w9 * #punct/#words
      + w11)
      
      target_length ~ NB(r,p)
      
    The biases w10 and w11 are optional.
    
    The predictor predicts EOS with NB(#consumed_words,r,p)
    """
    
    def __init__(self, text_file, model_weights, use_point_probs, offset = 0):
        """Creates a new target sentence length model predictor.
        
        Args:
            text_file (string): Path to the text file with the 
                                unindexed source sentences, i.e. not
                                using word ids
            model_weights (list): Weights w0 to w11 of the length 
                                  model. See class docstring for more
                                  information
            use_point_probs (bool): Use point estimates for EOS token,
                                    0.0 otherwise 
            offset (int): Subtract this from hypothesis length before
                          applying the NB model
        """
        super(NBLengthPredictor, self).__init__()
        self.use_point_probs = use_point_probs
        self.offset = offset
        if len(model_weights) == 2*NUM_FEATURES: # add biases
            model_weights.append(0.0)
            model_weights.append(0.0)
        if len(model_weights) != 2*NUM_FEATURES+2:
            logging.fatal("Number of length model weights has to be %d or %d"
                    % (2*NUM_FEATURES, 2*NUM_FEATURES+2))
        self.r_weights = model_weights[0:NUM_FEATURES] + [model_weights[-2]]
        self.p_weights = model_weights[NUM_FEATURES:2*NUM_FEATURES] + [model_weights[-1]]
        self.src_features = self._extract_features(text_file)
        self.n_consumed = 0 

    def _extract_features(self, file_name):
        """Extract all features from the source sentences. """
        feats = []
        with open(file_name) as f:
            for line in f:
                feats.append(self._analyse_sentence(line.strip()))
        return feats
    
    def _analyse_sentence(self, sentence):
        """Extract features for a single source sentence.
        
        Args:
            sentence (string): Source sentence string
        
        Returns:
            5-tuple of features as described in the class docstring
        """
        n_char = len(sentence) + 0.0
        n_words = len(sentence.split()) + 0.0
        n_punct = sum([sentence.count(s) for s in ",.:;-"]) + 0.0
        return [n_char, n_words, n_punct, n_char/n_words, n_punct/n_words]
        
    def get_unk_probability(self, posterior):
        """If we use point estimates, return 0 (=1). Otherwise, return
        the 1-p(EOS), with p(EOS) fetched from ``posterior``
        """
        if self.use_point_probs:
            if self.n_consumed == 0:
                return self.max_eos_prob
            return 0.0
        if self.n_consumed == 0:
            return 0.0
        return np.log(1.0 - np.exp(posterior[utils.EOS_ID]))
    
    def predict_next(self):
        """Returns a dictionary with single entry for EOS. """
        if self.n_consumed == 0:
            return {utils.EOS_ID : utils.NEG_INF}
        return {utils.EOS_ID : self._get_eos_prob()}
    
    def _get_eos_prob(self):
        """Get loglikelihood according cur_p, cur_r, and n_consumed """
        eos_point_prob = self._get_eos_point_prob(max(
                                              1, 
                                              self.n_consumed - self.offset))
        if self.use_point_probs:
            return eos_point_prob - self.max_eos_prob
        if not self.prev_eos_probs:
            self.prev_eos_probs.append(eos_point_prob)
            return eos_point_prob
        # bypass utils.log_sum because we always want to use logsumexp here 
        prev_sum = logsumexp(np.asarray([p for p in self.prev_eos_probs])) 
        self.prev_eos_probs.append(eos_point_prob)
        # Desired prob is eos_point_prob / (1-last_eos_probs_sum)
        return eos_point_prob - np.log(1.0-np.exp(prev_sum))
    
    def _get_eos_point_prob(self, n):
        return gammaln(n + self.cur_r) \
                - gammaln(n + 1) \
                - gammaln(self.cur_r) \
                + n * np.log(self.cur_p) \
                + self.cur_r * np.log(1.0-self.cur_p)
    
    def _get_max_eos_prob(self):
        """Get the maximum loglikelihood according cur_p, cur_r 
        TODO: replace this brute force impl. with something smarter
        """
        max_prob = utils.NEG_INF
        n_prob = max_prob
        n = 0
        while n_prob == max_prob:
            n += 1
            n_prob = self._get_eos_point_prob(n)
            max_prob = max(max_prob, n_prob)
        return max_prob
    
    def initialize(self, src_sentence):
        """Extract features for the source sentence. Note that this
        method does not use ``src_sentence`` as we need the string
        representation of the source sentence to extract features.
        
        Args:
            src_sentence (list): Not used
        """
        feat = self.src_features[self.current_sen_id] + [1.0]
        self.cur_r  = max(EPS_R, np.dot(feat, self.r_weights));
        p = np.dot(feat, self.p_weights)
        p = 1.0 / (1.0 + math.exp(-p))
        self.cur_p = max(EPS_P, min(1.0 - EPS_P, p))
        self.n_consumed = 0
        self.prev_eos_probs = []
        if self.use_point_probs:
            self.max_eos_prob = self._get_max_eos_prob()
    
    def consume(self, word):
        """Increases the current history length
        
        Args:
            word (int): Not used
        """
        self.n_consumed = self.n_consumed + 1
    
    def get_state(self):
        """State consists of the number of consumed words, and the
        accumulator for previous EOS probability estimates if we 
        don't use point estimates.
        """
        return self.n_consumed,self.prev_eos_probs
    
    def set_state(self, state):
        """Set the predictor state """
        self.n_consumed,self.prev_eos_probs = state

    def reset(self):
        """Empty method. """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true if the number of consumed words is the same """
        n1,_ = state1
        n2,_ = state2
        return n1 == n2


class WordCountPredictor(Predictor):
    """This predictor adds the number of words as feature. """
    
    def __init__(self, word = -1):
        """Creates a new word count predictor instance.
        
        Args:
            word (int): If this is non-negative we count only the
                        number of the specified word. If its
                        negative, count all words
        """
        super(WordCountPredictor, self).__init__()
        if word < 0:
            self.posterior = {utils.EOS_ID : 0.0}
            self.unk_prob = -1.0
        else:
            self.posterior = {word : -1.0}
            self.unk_prob = 0.0 
        
    def get_unk_probability(self, posterior):
        return self.unk_prob
    
    def predict_next(self):
        """Set score for EOS to the number of consumed words """
        return self.posterior
    
    def initialize(self, src_sentence):
        """Empty
        """
        pass
    
    def consume(self, word):
        """Empty
        """
        pass
    
    def get_state(self):
        """Returns true """
        return True
    
    def set_state(self, state):
        """Empty """
        pass

    def reset(self):
        """Empty method. """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true """
        return True


class ExternalLengthPredictor(Predictor):
    """This predictor loads the distribution over target sentence
    lengths from an external file. The file contains blank separated
    length:score pairs in each line which define the length 
    distribution. The predictor adds the specified scores directly
    to the EOS score.
    """
    
    def __init__(self, path):
        """Creates a external length distribution predictor.
        
        Args:
            path (string): Path to the file with target sentence length
                           distributions.
        """
        super(ExternalLengthPredictor, self).__init__()
        self.trg_lengths = []
        with open(path) as f:
            for line in f:
                scores = {}
                for pair in line.strip().split():
                    length,score = pair.split(':')
                    scores[int(length)] = float(score)
                self.trg_lengths.append(scores)
        
    def get_unk_probability(self, posterior):
        """Returns 0=log 1 if the partial hypothesis does not exceed
        max length. Otherwise, predict next returns an empty set,
        and we set everything else to -inf.
        """
        if self.n_consumed < self.max_length:
            return 0.0
        return utils.NEG_INF
    
    def predict_next(self):
        """Returns a dictionary with one entry and value 0 (=log 1). The
        key is either the next word in the target sentence or (if the
        target sentence has no more words) the end-of-sentence symbol.
        """
        if self.n_consumed in self.cur_scores: 
            return {utils.EOS_ID : self.cur_scores[self.n_consumed]}
        return {utils.EOS_ID : utils.NEG_INF} 
    
    def initialize(self, src_sentence):
        """Fetches the corresponding target sentence length 
        distribution and resets the word counter.
        
        Args:
            src_sentence (list):  Not used
        """
        self.cur_scores = self.trg_lengths[self.current_sen_id]
        self.max_length = max(self.cur_scores)
        self.n_consumed = 0

    def consume(self, word):
        """Increases word counter by one.
        
        Args:
            word (int): Not used
        """
        self.n_consumed = self.n_consumed + 1
    
    def get_state(self):
        """Returns the number of consumed words """
        return self.n_consumed
    
    def set_state(self, state):
        """Set the number of consumed words """
        self.n_consumed = state

    def reset(self):
        """Empty method. """
        pass
    
    def is_equal(self, state1, state2):
        """Returns true if the number of consumed words is the same """
        return state1 == state2


class NgramCountPredictor(Predictor):
    """This predictor counts the number of n-grams in hypotheses. n-gram
    posteriors are loaded from a file. The predictor score is the sum of
    all n-gram posteriors in a hypothesis. """
    
    def __init__(self, path, order=0, discount_factor=-1.0):
        """Creates a new ngram count predictor instance.
        
        Args:
            path (string): Path to the n-gram posteriors. File format:
                           <ngram> : <score> (one ngram per line). Use
                           placeholder %d for sentence id.
            order (int): If positive, count n-grams of the specified
                         order. Otherwise, count all n-grams
            discount_factor (float): If non-negative, discount n-gram
                                     posteriors by this factor each time 
                                     they are consumed 
        """
        super(NgramCountPredictor, self).__init__()
        self.path = path 
        self.order = order
        self.discount_factor = discount_factor
        
    def get_unk_probability(self, posterior):
        """Always return 0.0 """
        return 0.0
    
    def predict_next(self):
        """Composes the posterior vector by collecting all ngrams which
        are consistent with the current history.
        """
        posterior = {}
        for i in reversed(range(len(self.cur_history)+1)):
            scores = self.ngrams.get(self.cur_history[i:])
            if scores:
                factors = False
                if self.discount_factor >= 0.0:
                    factors = self.discounts.get(self.cur_history[i:])
                if not factors:
                    for w,score in scores.iteritems():
                        posterior[w] = posterior.get(w, 0.0) + score
                else:
                    for w,score in scores.iteritems():
                        posterior[w] = posterior.get(w, 0.0) +  \
                                       factors.get(w, 1.0) * score
        return posterior
    
    def _load_posteriors(self, path):
        """Sets up self.max_history_len and self.ngrams """
        self.max_history_len = 0
        self.ngrams = SimpleTrie()
        with open(path) as f:
            for line in f:
                ngram,score = line.split(':')
                words = [int(w) for w in ngram.strip().split()]
                if self.order > 0 and len(words) != self.order:
                    continue
                hist = words[:-1]
                last_word = words[-1]
                if last_word == utils.GO_ID:
                    continue
                self.max_history_len = max(self.max_history_len, len(hist))
                p = self.ngrams.get(hist)
                if p:
                    p[last_word] = float(score.strip())
                else:
                    self.ngrams.add(hist, {last_word: float(score.strip())})
    
    def initialize(self, src_sentence):
        """Loads n-gram posteriors and resets history.
        
        Args:
            src_sentence (list): not used
        """
        self._load_posteriors(utils.get_path(self.path, self.current_sen_id+1))
        self.cur_history = [utils.GO_ID]
        self.discounts = SimpleTrie()
    
    def consume(self, word):
        """Adds ``word`` to the current history. Shorten if the extended
        history exceeds ``max_history_len``.
        
        Args:
            word (int): Word to add to the history.
        """
        self.cur_history.append(word)
        if len(self.cur_history) > self.max_history_len:
            self.cur_history = self.cur_history[-self.max_history_len:]
        if self.discount_factor >= 0.0:
            for i in range(len(self.cur_history)):
                key = self.cur_history[i:-1]
                factors = self.discounts.get(key)
                if not factors:
                    factors = {word: self.discount_factor}
                else:
                    factors[word] = factors.get(word, 1.0)*self.discount_factor
                self.discounts.add(key, factors)
    
    def get_state(self):
        """Current history is the predictor state """
        return self.cur_history,self.discounts
    
    def set_state(self, state):
        """Current history is the predictor state """
        self.cur_history,self.discounts = state

    def reset(self):
        """Empty method. """
        pass
    
    def is_equal(self, state1, state2):
        """Hypothesis recombination is
        not supported if discounting is enabled.
        """
        if self.discount_factor >= 0.0:
            return False
        hist1 = state1[0]
        hist2 = state2[0]
        if hist1 == hist2: # Return true if histories match
            return True
        if len(hist1) > len(hist2):
            hist_long = hist1
            hist_short = hist2
        else:
            hist_long = hist2
            hist_short = hist1
        min_len = len(hist_short)
        for n in xrange(1, min_len+1): # Look up non matching in self.ngrams
            key1 = hist1[-n:]
            key2 = hist2[-n:]
            if key1 != key2:
                if self.ngrams.get(key1) or self.ngrams.get(key2):
                    return False
        for n in xrange(min_len+1, len(hist_long)+1):
            if self.ngrams.get(hist_long[-n:]):
                return False
        return True


class UnkCountPredictor(Predictor):
    """This predictor regulates the number of UNKs in the output. We 
    assume that the number of UNKs in the target sentence is Poisson 
    distributed. This predictor is configured with n lambdas for
    0,1,...,>=n-1 UNKs in the source sentence. """
    
    def __init__(self, src_vocab_size, lambdas):
        """Initializes the UNK count predictor.

        Args:
            src_vocab_size (int): Size of source language vocabulary.
                                  Indices greater than this are 
                                  considered as UNK.
            lambdas (list): List of floats. The first entry is the 
                            lambda parameter given that the number of
                            unks in the source sentence is 0 etc. The
                            last float is lambda given that the source
                            sentence has more than n-1 unks.
        """
        self.lambdas = lambdas
        self.l = lambdas[0]
        self.src_vocab_size = src_vocab_size
        super(UnkCountPredictor, self).__init__()
        
    def get_unk_probability(self, posterior):
        """Always returns 0 (= log 1) except for the first time """
        if self.n_consumed == 0:
            return self.max_prob
        return 0.0
    
    def predict_next(self):
        """Set score for EOS to the number of consumed words """
        if self.n_consumed == 0:
            return {utils.EOS_ID : self.unk_prob}
        if self.n_unk < self.max_prob_idx:
            return {utils.EOS_ID : self.unk_prob - self.max_prob}
        return {utils.UNK_ID : self.unk_prob - self.consumed_prob}
    
    def initialize(self, src_sentence):
        """Count UNKs in ``src_sentence`` and reset counters.
        
        Args:
            src_sentence (list): Count UNKs in this list
        """
        src_n_unk = len([w for w in src_sentence if w == utils.UNK_ID 
                                                    or w > self.src_vocab_size])
        self.l = self.lambdas[min(len(self.lambdas)-1, src_n_unk)]
        self.n_consumed = 0
        self.n_unk = 0
        self.unk_prob = self._get_poisson_prob(1)
        # Mode at lambda is the maximum of the poisson function
        self.max_prob_idx = int(self.l)
        self.max_prob = self._get_poisson_prob(self.max_prob_idx)
        ceil_prob = self._get_poisson_prob(self.max_prob_idx + 1)
        if ceil_prob > self.max_prob:
            self.max_prob = ceil_prob
            self.max_prob_idx = self.max_prob_idx + 1
        self.consumed_prob = self.max_prob

    def _get_poisson_prob(self, n):
        """Get the log of the poisson probability for n events. """
        return n * np.log(self.l) - self.l - sum([np.log(i+1) for i in xrange(n)])
    
    def consume(self, word):
        """Increases unk counter by one if ``word`` is unk.
        
        Args:
            word (int): Increase counter if ``word`` is UNK
        """
        self.n_consumed += 1
        if word == utils.UNK_ID:
            if self.n_unk >= self.max_prob_idx:
                self.consumed_prob = self.unk_prob
            self.n_unk += 1
            self.unk_prob = self._get_poisson_prob(self.n_unk+1)
    
    def get_state(self):
        """Returns the number of consumed words """
        return self.n_unk,self.n_consumed,self.unk_prob,self.consumed_prob
    
    def set_state(self, state):
        """Set the number of consumed words """
        self.n_unk,self.n_consumed,self.unk_prob,self.consumed_prob = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the state is the same"""
        return state1 == state2


