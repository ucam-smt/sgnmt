"""This module contains the two basic predictor interfaces
for bounded and unbounded vocabulary predictors.
"""

from abc import abstractmethod

from cam.sgnmt import utils
from cam.sgnmt.utils import Observer, NEG_INF, MESSAGE_TYPE_DEFAULT


class Predictor(Observer):
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
        super(Predictor, self).__init__()
        self.current_sen_id = 0

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
    
    def is_equal(self, state1, state2):
        """Returns true if two predictor states are equal, i.e. both
        states will always result in the same scores. This is used for
        hypothesis recombination
        
        Args:
            state1 (object): First predictor state
            state2 (object): Second predictor state
        
        Returns:
            bool. True if both states are equal, false if not
        """
        return False
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """We implement the ``notify`` method from the ``Observer``
        super class with an empty method here s.t. predictors do not
        need to implement it.
        
        Args:
            message (object): The posterior sent by the decoder
        """
        pass


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
            probability ``get_unk_probability()``. The returned set should
            not contain any ids which are not in ``trgt_words``, but it
            does not have to score all of them
        """
        raise NotImplementedError
