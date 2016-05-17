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

from cam.sgnmt.decoding.core import Predictor
import numpy as np
from cam.sgnmt import utils
from scipy.special import gammaln
from scipy.misc import logsumexp


NEG_INF = float("-inf")


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
    src sentence.
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
    
    def __init__(self, text_file, model_weights, use_point_probs):
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
        """
        super(NBLengthPredictor, self).__init__()
        self.use_point_probs = use_point_probs
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
            return 0.0
        if self.n_consumed == 0:
            return 0.0
        return np.log(1.0 - np.exp(posterior[utils.EOS_ID]))
    
    def predict_next(self):
        """Returns a dictionary with single entry for EOS. """
        if self.n_consumed == 0:
            return {utils.EOS_ID : NEG_INF}
        return {utils.EOS_ID : self._get_eos_prob()}
    
    def _get_eos_prob(self):
        """Get loglikelihood according cur_p, cur_r, and n_consumed """
        eos_point_prob = gammaln(self.n_consumed+self.cur_r) \
            - gammaln(self.n_consumed+1) \
            - gammaln(self.cur_r) \
            + self.n_consumed * np.log(self.cur_p) \
            + self.cur_r * np.log(1.0-self.cur_p);
        if self.use_point_probs:
            return eos_point_prob
        if not self.prev_eos_probs:
            self.prev_eos_probs.append(eos_point_prob)
            return eos_point_prob
        # bypass utils.log_sum because we always want to use logsumexp here 
        prev_sum = logsumexp(np.asarray([p for p in self.prev_eos_probs])) 
        self.prev_eos_probs.append(eos_point_prob)
        # Desired prob is eos_point_prob / (1-last_eos_probs_sum)
        return eos_point_prob - np.log(1.0-np.exp(prev_sum))
    
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



class WordCountPredictor(Predictor):
    """This predictor adds the number of words as feature """
    
    def __init__(self):
        super(WordCountPredictor, self).__init__()
        
    def get_unk_probability(self, posterior):
        """Always returns 0 (= log 1) """
        return 0.0
    
    def predict_next(self):
        """Set score for EOS to the number of consumed words """
        return {utils.EOS_ID : self.n_consumed}
    
    def initialize(self, src_sentence):
        """Resets the internal counter for consumed words.
        
        Args:
            src_sentence (list): Not used
        """
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
