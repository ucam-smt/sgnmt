"""Contains all the basic interfaces and abstract classes for decoding.
This is mainly ``Predictor`` and ``Decoder``. Functionality should be
implemented mainly in the ``predictors`` package for predictors and in
the ``decoding.decoder`` module for decoders.
"""

from abc import abstractmethod
import numpy as np
from cam.sgnmt import utils

NEG_INF = float("-inf")

"""The ``CLOSED_VOCAB_SCORE_NORM_*`` constants define the normalization
behavior for closed vocabulary predictor scores. Closed vocabulary 
predictors (e.g. NMT) have a predefined (and normally very limited) 
vocabulary. In contrast, open vocabulary predictors (see 
``UnboundedPredictor``) are defined over a much larger vocabulary 
(e.g. FST) s.t. it is easier to consider them as having an open 
vocabulary. When combining open and closed vocabulary predictors, we use
the UNK probability of closed vocabulary predictors for words outside 
their vocabulary. The following flags decide (as argument to 
``Decoder``) what to do with the closed vocabulary predictor scores
when combining them with open vocabulary predictors in that way. This
can be changed with the --closed_vocab_norm argument """


CLOSED_VOCAB_SCORE_NORM_NONE = 1
"""None: Do not apply any normalization. """


CLOSED_VOCAB_SCORE_NORM_EXACT = 2
"""Exact: Normalize by 1 plus the number of words outside the 
vocabulary to make it a valid distribution again"""


CLOSED_VOCAB_SCORE_NORM_REDUCED = 3
"""Reduced: Always normalize the closed vocabulary scores to the 
vocabulary which is defined by the open vocabulary predictors at each
time step. """


class Predictor(object):
    """A predictor produces the predictive probability distribution of
    the next word given the state of the predictor. The state may 
    change during ``predict_next()`` and ``consume()``. The functions
    ``get_state()`` and ``set_state()`` can be used for non-greedy 
    decoding. Note: The state describes the predictor with the current
    history. It does not encapsulate the current source sentence, i.e. 
    you cannot recover a predictor state if ``initialize()`` was called
    in between. ``predict_next()`` and ``consume()`` must be called 
    alternately. This holds even when using ``get_state()`` and 
    ``set_state()``: Loading/saving states is transparent to the
    predictor instance.
    """
    
    def __init__(self):
        """Initializes ``current_sen_id`` with 0. """
        self.current_sen_id = 0
        pass

    def set_current_sen_id(self, cur_sen_id):
        """This function is called between ``initialize()`` calls to 
        increment the sentence id counter. It can also be used to skip 
        sentences for the --range argument.
        
        Args:
            cur_sen_id (int):  Sentence id for the next call of
                               ``initialize()``
        """
        self.current_sen_id = cur_sen_id
    
    @abstractmethod
    def predict_next(self):
        """Returns the predictive distribution over the target 
        vocabulary for the next word given the predictor state. Note 
        that the prediction itself can change the state of the 
        predictor. For example, the neural predictor updates the 
        decoder network state and its attention to predict the next 
        word. Two calls of ``predict_next()`` must be separated by a 
        ``consume()`` call.
        
        Returns:
            dictionary,array,list. Word log probabilities for the next 
            target token. All ids which are not set are assumed to have
            probability ``get_unk_probability()``
        """
        raise NotImplementedError
    
    @abstractmethod
    def consume(self, word):
        """Expand the current history by ``word`` and update the 
        internal predictor state accordingly. Two calls of ``consume()``
        must be separated by a ``predict_next()`` call.
        
        Args:
            word (int):  Word to add to the current history
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self):
        """Get the current predictor state. The state can be any object
        or tuple of objects which makes it possible to return to the
        predictor state with the current history.
        
        Returns:
          object. Predictor state
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_state(self, state):
        """Loads a predictor state from an object created with 
        ``get_state()``. Note that this does not copy the argument but
        just references the given state. If ``state`` is going to be
        used in the future to return to that point again, you should
        copy the state with ``copy.deepcopy()`` before.
        
        Args:
           state (object): Predictor state as returned by 
                           ``get_state()``
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """Reset the predictor state to the initial configuration. This
        is required when a new set of sentences is to be decoded, e.g.
        to reset the sentence counter in the fst predictor to load the
        correct lattice. This function is NOT called each time before
        decoding a single sentence. See ``initialize()`` for this.
        """
        raise NotImplementedError
    
    def estimate_future_cost(self, hypo):
        """Predictors can implement their own look-ahead cost functions.
        They are used in A* if the --heuristics parameter is set to 
        predictor. This function should return the future log *cost* 
        (i.e. the lower the better) given the current predictor state, 
        assuming that the last word in the partial hypothesis 'hypo' is
        consumed next. This function must not change the internal 
        predictor state.
        
        Args:
            hypo (PartialHypothesis): Hypothesis for which to estimate
                                      the future cost given the current
                                      predictor state
        
        Returns
            float. Future cost
        """
        return 0.0
    
    def get_unk_probability(self, posterior):
        """This function defines the probability of all words which are
        not in ``posterior``. This is usually used to combine open and
        closed vocabulary predictors. The argument ``posterior`` should 
        have been produced with ``predict_next()``
        
        Args:
            posterior (list,array,dict): Return value of the last call
                                         of ``predict_next``
        
        Returns:
            float: Score to use for words outside ``posterior``
        """
        return NEG_INF
    
    def initialize(self, src_sentence):
        """Initialize the predictor with the given source sentence. 
        This resets the internal predictor state and loads everything 
        which is constant throughout the processing of a single source
        sentence. For example, the NMT decoder runs the encoder network
        and stores the source annotations.
        
        Args:
            src_sentence (list): List of word IDs which form the source
                                 sentence without <S> or </S>
        """
        pass
    
    def initialize_heuristic(self, src_sentence):
        """This is called after ``initialize()`` if the predictor is
        registered as heuristic predictor (i.e. 
        ``estimate_future_cost()`` will be called in the future).
        Predictors can implement this function for initialization of 
        their own heuristic mechanisms.
        
        Args:
            src_sentence (list): List of word IDs which form the source
                                 sentence without <S> or </S>
        """
        pass
    
    def finalize_posterior(self, scores, use_weights, normalize_scores):
        """This method can be used to enforce the parameters use_weights
        normalize_scores in predictors with dict posteriors.
        
        Args:
            scores (dict): unnormalized log valued scores
            use_weights (bool): Set to false to replace all values in 
                                ``scores`` with 0 (= log 1)
            normalize_scores: Set to true to make the exp of elements 
                              in ``scores`` sum up to 1"""
        if not scores: # empty scores -> pass through
            return scores
        if not use_weights:
            scores = dict.fromkeys(scores, 0.0)
        if normalize_scores:
            log_sum = utils.log_sum(scores.itervalues())
            ret = {k: v - log_sum for k, v in scores.iteritems()}
            return ret
        return scores


class UnboundedVocabularyPredictor(Predictor):
    """Predictors under this class implement models with very large 
    target vocabularies, for which it is too inefficient to list the 
    entire posterior. Instead, they are evaluated only for a given list
    of target words. This list is usually created by taking all non-zero
    probability words from the bounded vocabulary predictors. An 
    example of a unbounded vocabulary predictor is the ngram predictor:
    Instead of listing the entire ngram vocabulary, we run srilm only
    on the words which are possible according other predictor (e.g. fst
    or nmt). This is realized by introducing the ``trgt_words``
    argument to ``predict_next``. """

    def __init__(self):
        """ Initializes ``current_sen_id`` with 0. """
        super(UnboundedVocabularyPredictor, self).__init__()

    @abstractmethod
    def predict_next(self, trgt_words):
        """Like in ``Predictor``, returns the predictive distribution
        over target words given the predictor state. Note 
        that the prediction itself can change the state of the 
        predictor. For example, the neural predictor updates the 
        decoder network state and its attention to predict the next 
        word. Two calls of ``predict_next()`` must be separated by a 
        ``consume()`` call.
        
        Args:
            trgt_words (list): List of target word ids.
        
        Returns:
            dictionary,array,list. Word log probabilities for the next 
            target token. All ids which are not set are assumed to have
            probability ``get_unk_probability(). The returned set should
            not contain any ids which are not in ``trgt_words``, but it
            does not have to score all of them
        """
        raise NotImplementedError


class Heuristic(object):
    """A ``Heuristic`` instance can be used to estimate the future 
    costs for a given word in a given state. See the ``heuristics``
    module for implementations."""
    
    def __init__(self):
        """Creates a heuristic without predictors. """
        super(Heuristic, self).__init__()
        self.predictors = []

    def set_predictors(self, predictors):
        """Set the predictors used by this heuristic. 
        
        Args:
            predictors (list):  Predictors and their weights to be
                                used with this heuristic. Should be in
                                the same form as ``Decoder.predictors``,
                                i.e. a list of (predictor, weight)
                                tuples
        """
        self.predictors = predictors
    
    def initialize(self, src_sentence):
        """Initialize the heuristic with the given source sentence.
        This is not passed through to the heuristic predictors
        automatically but handles initialization outside the
        predictors.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        pass

    @abstractmethod
    def estimate_future_cost(self, hypo):
        """Estimate the future cost (i.e. negative score) given the 
        states of the predictors set by ``set_predictors`` for a
        partial hypothesis ``hypo``. Note that this function is not 
        supposed to change predictor states. If (e.g. for the greedy 
        heuristic) this is not possible, the predictor states must be
        changed back after execution by the implementing method.
        
        Args:
            hypo (PartialHypo): Hypothesis for which to estimate the
                                future cost
        
        Returns:
            float. The future cost estimate for this heuristic
        """
        raise NotImplementedError
    

def breakdown2score_sum(working_score, score_breakdown):
    """Implements the combination scheme 'sum' by always returning
    ``working_score``. This function is designed to be assigned to
    the globals ``breakdown2score_partial`` or ``breakdown2score_full``
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
    
    Returns:
        float. Returns ``working_score``
    """
    return working_score


def breakdown2score_length_norm(working_score, score_breakdown):
    """Implements the combination scheme 'length_norm' by normalizing
    the sum of the predictor scores by the length of the current 
    sequence (i.e. the length of ``score_breakdown``. This function is
    designed to be assigned to the globals ``breakdown2score_partial``
    or ``breakdown2score_full``. 
    TODO could make more efficient use of ``working_score``
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``. Not used.
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
    
    Returns:
        float. Returns a length normalized ``working_score``
    """
    score = sum([Decoder.combi_arithmetic_unnormalized(s) 
                        for s in score_breakdown])
    return score / len(score_breakdown)


def breakdown2score_bayesian(working_score, score_breakdown):
    """This realizes score combination following the Bayesian LM 
    interpolation scheme from (Allauzen and Riley, 2011)
    
      Bayesian Language Model Interpolation for Mobile Speech Input
    
    By setting K=T we define the predictor weights according the score
    the predictors give to the current partial hypothesis. The initial
    predictor weights are used as priors. This function is designed to 
    be assigned to the globals ``breakdown2score_partial`` or 
    ``breakdown2score_full``. 
    TODO could make more efficient use of ``working_score``
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``. Not used.
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
    
    Returns:
        float. Bayesian interpolated predictor scores
    """
    if not score_breakdown:
        return working_score
    acc = []
    prev_alphas = [] # list of all alpha_i,k
    # Write priors to alphas
    for (p,w) in score_breakdown[0]:
        prev_alphas.append(np.log(w))
    for pos in score_breakdown: # for each position in the hypothesis
        alphas = []
        sub_acc = []
        # for each predictor (p: p_k(w_i|h_i), w: prior p(k))
        for k,(p,w) in enumerate(pos): 
            alpha = prev_alphas[k] + p
            alphas.append(alpha)
            sub_acc.append(p + alpha)
        acc.append(utils.log_sum(sub_acc) - utils.log_sum(alphas))
        prev_alphas = alphas
    return sum(acc)


"""The function breakdown2score_partial is called at each hypothesis
expansion. This should only be changed if --combination_scheme is not 
'sum' and --apply_combination_scheme_to_partial_hypos is set to true.
""" 
breakdown2score_partial = breakdown2score_sum


"""The function breakdown2score_full is called at each creation of a 
full hypothesis, i.e. only once per hypothesis
"""
breakdown2score_full = breakdown2score_sum


class Decoder(object):    
    """A ``Decoder`` instance represents a particular search strategy
    such as A*, beam search, greedy search etc. Decisions are made 
    based on the outputs of one or many predictors, which are 
    maintained by the ``Decoder`` instance.
    """
    
    def __init__(self, closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE):
        """Initializes the decoder instance with no predictors or 
        heuristics.
        
        Args:
            closed_vocab_norm (int): Defines the normalization behavior
                                     for closed vocabulary predictor
                                     scores. See the documentation to
                                     the ``CLOSED_VOCAB_SCORE_NORM_*``
                                     variables for more information
        """
        self.predictors = [] # Tuples (predictor, weight)
        self.heuristics = []
        self.heuristic_predictors = []
        self.predictor_names = []
        self.nbest = 1 # length of n-best list
        self.combi_predictor_method = Decoder.combi_arithmetic_unnormalized
        self.combine_posteriors = self._combine_posteriors_norm_none
        if closed_vocab_norm == CLOSED_VOCAB_SCORE_NORM_EXACT:
            self.combine_posteriors = self._combine_posteriors_norm_exact
        elif closed_vocab_norm == CLOSED_VOCAB_SCORE_NORM_REDUCED:
            self.combine_posteriors = self._combine_posteriors_norm_reduced
        self.current_sen_id = -1
        self.start_sen_id = 0
        self.apply_predictors_count = 0
    
    def add_predictor(self, name, predictor, weight=1.0):
        """Adds a predictor to the decoder. This means that this 
        predictor is going to be used to predict the next target word
        (see ``predict_next``)
        
        Args:
            name (string): Predictor name like 'nmt' or 'fst'
            predictor (Predictor): Predictor instance
            weight (float): Predictor weight
        """
        self.predictors.append((predictor, weight))
        self.predictor_names.append(name)
    
    def set_heuristic_predictors(self, heuristic_predictors):
        """Define the list of predictors used by heuristics. This needs
        to be called before adding heuristics with ``add_heuristic()``

        Args:
            heuristic_predictors (list):  Predictors and their weights 
                                          to be used with heuristics. 
                                          Should be in the same form 
                                          as ``Decoder.predictors``,
                                          i.e. a list of 
                                          (predictor, weight) tuples
        """
        self.heuristic_predictors = heuristic_predictors
    
    def add_heuristic(self, heuristic):
        """Add a heuristic to the decoder. For future cost estimates,
        the sum of the estimates from all heuristics added so far will
        be used. The predictors used in this heuristic have to be set
        before via ``set_heuristic_predictors()``
        
        Args:
            heuristic (Heuristic): A heuristic to use for future cost
                                   estimates
        """
        heuristic.set_predictors(self.heuristic_predictors)
        self.heuristics.append(heuristic)
    
    def estimate_future_cost(self, hypo):
        """Uses all heuristics which have been added with 
        ``add_heuristic`` to estimate the future cost for a given
        partial hypothesis. The estimates are used in heuristic based
        searches like A*. This function returns the future log *cost* 
        (i.e. the lower the better), assuming that the last word in the
        partial hypothesis ``hypo`` is consumed next.
        
        Args:
            hypo (PartialHypothesis): Hypothesis for which to estimate
                                      the future cost given the current
                                      predictor state
        
        Returns
            float. Future cost
        """
        return sum([h.estimate_future_cost(hypo) for h in  self.heuristics])
    
    def consume(self, word):
        """Calls ``consume()`` on all predictors. """
        for (p, _) in self.predictors:
            p.consume(word) # May change predictor state
    
    def _get_non_zero_words(self, bounded_predictors, posteriors):
        """Get the set of words from the predictor posteriors which 
        have non-zero probability. This set of words is then passed
        through to the open vocabulary predictors.
        """
        words = None
        for idx, posterior in enumerate(posteriors):
            (p, _) = bounded_predictors[idx]
            if p.get_unk_probability(posterior) == NEG_INF: # Restrict to this
                if not words:
                    words = set(utils.common_viewkeys(posterior))
                else:
                    words = words & set(utils.common_viewkeys(posterior))
                if not words: # Special case empty set: no word is possible
                    return set([utils.EOS_ID])
        if not words: # If no restricting predictor, use union
            words = set(utils.common_viewkeys(posteriors[0]))
            for posterior in posteriors[1:]:
                words = words | set(utils.common_viewkeys(posterior))
        return words
    
    def apply_predictors(self):
        """Get the distribution over the next word by combining the
        predictor scores.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        self.apply_predictors_count += 1
        bounded_predictors = [el for el in self.predictors 
                        if not isinstance(el[0], UnboundedVocabularyPredictor)]
        # Get bounded posteriors
        bounded_posteriors = [p.predict_next() for (p, _) in bounded_predictors]
        non_zero_words = self._get_non_zero_words(bounded_predictors,
                                                  bounded_posteriors)
        # Add unbounded predictors and unk probabilities
        posteriors = []
        unk_probs = []
        bounded_idx = 0
        for (p, _) in self.predictors:
            if isinstance(p, UnboundedVocabularyPredictor):
                posterior = p.predict_next(non_zero_words)
            else: # Take it from the bounded_* variables
                posterior = bounded_posteriors[bounded_idx]
                bounded_idx += 1
            posteriors.append(posterior)
            unk_probs.append(p.get_unk_probability(posterior))
        return self.combine_posteriors(non_zero_words, posteriors, unk_probs)
    
    def _combine_posteriors_norm_none(self,
                                      non_zero_words,
                                      posteriors,
                                      unk_probs):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_NONE``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        combined = {}
        score_breakdown = {}
        for trgt_word in non_zero_words:
            preds = [(utils.common_get(posteriors[idx],
                                       trgt_word, unk_probs[idx]), w)
                        for idx, (_,w) in enumerate(self.predictors)]
            combined[trgt_word] = self.combi_predictor_method(preds) 
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown
    
    def _combine_posteriors_norm_exact(self,
                                       non_zero_words,
                                       posteriors,
                                       unk_probs):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_EXACT``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        unk_counts = [0] * n_predictors
        for trgt_word in non_zero_words:
            preds = []
            for idx, (_,w) in enumerate(self.predictors):
                if utils.common_contains(posteriors[idx], trgt_word):
                    preds.append((posteriors[idx][trgt_word], w))
                else:
                    preds.append((unk_probs[idx], w))
                    unk_counts[idx] += 1
            score_breakdown_raw[trgt_word] = preds
        renorm_factors = [0.0] * n_predictors
        for idx in xrange(n_predictors):
            if unk_counts[idx] > 1:
                renorm_factors[idx] = np.log(
                            1.0 
                            + (unk_counts[idx] - 1.0) * np.exp(unk_probs[idx]))  
        return self._combine_posteriors_with_renorm(score_breakdown_raw,
                                                    renorm_factors)
    
    def _combine_posteriors_norm_reduced(self,
                                         non_zero_words,
                                         posteriors,
                                         unk_probs):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_REDUCED``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        for trgt_word in non_zero_words: 
            score_breakdown_raw[trgt_word] = [(utils.common_get(
                                                posteriors[idx],
                                                trgt_word, unk_probs[idx]), w)
                        for idx, (_,w) in enumerate(self.predictors)]
        sums = []
        for idx in xrange(n_predictors):
            sums.append(utils.log_sum([preds[idx][0] 
                            for preds in score_breakdown_raw.itervalues()]))
        return self._combine_posteriors_with_renorm(score_breakdown_raw, sums)
    
    def _combine_posteriors_with_renorm(self,
                                        score_breakdown_raw,
                                        renorm_factors):
        """Helper function for ``_combine_posteriors_norm_*`` functions
        to renormalize score breakdowns by predictor specific factors.
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        combined = {}
        score_breakdown = {}
        for trgt_word,preds_raw in score_breakdown_raw.iteritems():
            preds = [(preds_raw[idx][0] - renorm_factors[idx],
                      preds_raw[idx][1]) for idx in xrange(n_predictors)]
            combined[trgt_word] = self.combi_predictor_method(preds) 
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown
    
    def set_start_sen_id(self, start_sen_id):
        """Set the internal sentence id counter `self.current_sen_id``
        to ``start_sen_id`` and resets all predictors."""
        self.start_sen_id = start_sen_id
        self.reset_predictors()

    def reset_predictors(self):
        """Calls ``reset()`` on all predictors and resets the sentence
        id counter ``self.current_sen_id``. """
        for (p, _) in self.predictors:
            p.reset()
        # -1 because its incremented in initialize_predictors
        self.current_sen_id = self.start_sen_id-1
            
    def initialize_predictors(self, src_sentence):
        """First, increases the sentence id counter and calls
        ``initialize()`` on all predictors. Then, ``initialize()`` is
        called for all heuristics.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        self.current_sen_id += 1
        for (p, _) in self.predictors:
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)
    
    def set_predictor_states(self, states):
        """Calls ``set_state()`` on all predictors. """
        i = 0
        for (p, _) in self.predictors:
            p.set_state(states[i])
            i = i + 1
    
    def get_predictor_states(self):
        """Calls ``get_state()`` on all predictors. """
        return [p.get_state() for (p, _) in self.predictors]
    
    def set_predictor_combi_method(self, method):
        """Defines how to accumulate scores over the sequence. Should
        be one of the ``combi_`` methods defined below
        
        Args:
            method (function):  A function which accepts a list of
                                tuples [(out1, weight1), ...] and
                                calculates a combined score, e.g.
                                one of the ``combi_*`` methods
        """
        self.predictor_combi_method = method
    
    @staticmethod
    def combi_arithmetic_unnormalized(x):
        """Calculates the weighted sum (or geometric mean of log 
        values). Do not use with empty lists.
        
        Args:
            x (list): List of tuples [(out1, weight1), ...]
        
        Returns:
            float. Weighted sum out1*weight1+out2*weight2...
        """
        (fAcc, _) = reduce(lambda (f1,w1), (f2,w2):(f1*w1 + f2*w2, 1.0),
                           x,
                           (0.0, 1.0))
        return fAcc
    
    @staticmethod
    def combi_geometric_unnormalized(x):
        """Calculates the weighted geometric mean. Do not use empty 
        lists.
        
        Args:
            x (list): List of tuples [(out1, weight1), ...]
        
        Returns:
            float. Weighted geo. mean: out1^weight1*out2^weight2...
        """
        (fAcc, _) = reduce(lambda (f1, w1), (f2, w2): (pow(f1,w1) * pow(f2*w2),
                                                       1.0),
                           x,
                           (1.0, 1.0))
        return fAcc

    @abstractmethod
    def decode(self, src_sentence):
        """Decodes a single source sentence. This method has to be 
        implemented by subclasses. It contains the core of the 
        implemented search strategy ``src_sentence`` is a list of
        source word ids representing the source sentence without
        <S> or </S> symbols. This method returns a list of hypotheses,
        order descending by score such that the first entry is the best
        decoding result. Implementations should delegate the scoring of
        hypotheses to the predictors via ``apply_predictors()``, and
        organize predictor states with the methods ``consume()``,
        ``get_predictor_states()`` and ``set_predictor_states()``. In
        this way, the decoder is decoupled from the scoring modules.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        raise NotImplementedError
