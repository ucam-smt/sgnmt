"""This module contains classes which are able to store unigram
probabilities and potentially collect them by observing a
decoder instance. This can be used for heuristics.
"""

from cam.sgnmt.decoding.core import Decoder
from cam.sgnmt.utils import Observer, MESSAGE_TYPE_DEFAULT, \
    MESSAGE_TYPE_POSTERIOR, NEG_INF, MESSAGE_TYPE_FULL_HYPO


class UnigramTable(Observer):
    """A unigram table stores unigram probabilities for a certain
    vocabulary. These statistics can be loaded from an external
    file (``FileUnigramTable``) or collected during decoding.
    """
    
    def __init__(self):
        """Creates a unigram table without entries. """
        self.heuristic_scores = {}
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Unigram tables usually observe the decoder, but some
        do not process messages from the decoder. This is an empty
        implementation of ``notify`` for those implementations.
        """
        pass
    
    def estimate(self, word, default=0.0):
        """Estimate the unigram score for the given word.
        
        Args:
            word (int): word ID
            default (float): Default value to be returned if ``word``
                             cannot be found in the table
        Returns:
            float. Unigram score or ``default`` if ``word`` is not in 
            table
        """
        return self.heuristic_scores.get(word, default)
    
    def reset(self):
        """This is called to reset collected statistics between each
        sentence pair.
        """
        self.heuristic_scores = {}
    

class FileUnigramTable(UnigramTable):
    """Loads a unigram table from an external file. """
    
    def __init__(self, path):
        """Loads the unigram table from ``path``. """
        super(FileUnigramTable, self).__init__()
        with open(path) as f:
            for line in f:
                w,s = line.split()
                self.heuristic_scores[int(w.strip())] = float(s.strip())
    
    def reset(self):
        pass


class AllStatsUnigramTable(UnigramTable):
    """This unigram table collect statistics from all partial hypos. """
    
    def __init__(self):
        """Pass through to super class constructor. """
        super(AllStatsUnigramTable, self).__init__()

    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Update unigram statistics. We assume to observe a Decoder
        instance. We update the unigram table if the message type 
        is ``MESSAGE_TYPE_POSTERIOR``.
        
        Args:
            message (object): Message from an observable ``Decoder``
            message_type (int): Message type
        """ 
        if message_type == MESSAGE_TYPE_POSTERIOR:
            posterior,_ = message
            for w, score in posterior.iteritems():
                self.heuristic_scores[w] = max(
                                        self.heuristic_scores.get(w, NEG_INF),
                                        score)


class FullStatsUnigramTable(UnigramTable):
    """This unigram table collect statistics from all full hypos. """
    
    def __init__(self):
        """Pass through to super class constructor. """
        super(FullStatsUnigramTable, self).__init__()

    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Update unigram statistics. We assume to observe a Decoder
        instance. We update the unigram table if the message type 
        is ``MESSAGE_TYPE_FULL_HYPO``.
        
        Args:
            message (object): Message from an observable ``Decoder``
            message_type (int): Message type
        """ 
        if message_type == MESSAGE_TYPE_FULL_HYPO:
            breakdowns = message.score_breakdown
            for pos,w in enumerate(message.trgt_sentence):
                self.heuristic_scores[w] = max(
                        self.heuristic_scores.get(w, NEG_INF),
                        Decoder.combi_arithmetic_unnormalized(breakdowns[pos]))
                
                
class BestStatsUnigramTable(UnigramTable):
    """This unigram table collect statistics from the best full hypo. """
    
    def __init__(self):
        """Pass through to super class constructor. """
        super(BestStatsUnigramTable, self).__init__()
        self.best_hypo_score = NEG_INF

    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Update unigram statistics. We assume to observe a Decoder
        instance. We update the unigram table if the message type 
        is ``MESSAGE_TYPE_FULL_HYPO``.
        
        Args:
            message (object): Message from an observable ``Decoder``
            message_type (int): Message type
        """ 
        if (message_type == MESSAGE_TYPE_FULL_HYPO 
                    and message.total_score > self.best_hypo_score):
            self.best_hypo_score = message.total_score
            self.heuristic_scores = {}
            breakdowns = message.score_breakdown
            for pos,w in enumerate(message.trgt_sentence):
                self.heuristic_scores[w] = \
                    Decoder.combi_arithmetic_unnormalized(breakdowns[pos])
    
    def reset(self):
        """This is called to reset collected statistics between each
        sentence pair.
        """
        self.heuristic_scores = {}
        self.best_hypo_score = NEG_INF