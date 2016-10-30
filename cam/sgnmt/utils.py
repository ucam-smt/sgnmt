"""This file contains common basic functionality which can be used from
anywhere. This includes the definition of reserved word indices, some
mathematical functions, and helper functions to deal with the small
quirks python sometimes has.
"""

from abc import abstractmethod
import numpy
import operator
from scipy.misc import logsumexp


# Reserved IDs
PAD_ID = None
""" Reserved word ID for padding. Relict from the TensorFlow 
implementation. """


GO_ID = 1
""" Reserved word ID for the start-of-sentence symbol. """


EOS_ID = 2
""" Reserved word ID for the end-of-sentence symbol. """


UNK_ID = 0
""" Reserved word ID for the unknown word (UNK). """


NOTAPPLICABLE_ID = 3
""" Reserved word ID which is currently not used. """


NEG_INF = float("-inf")


INF = float("inf")


def switch_to_old_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``PAD_ID``, ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the old scheme. The old scheme is used in older NMT models, 
    and is legacy from TensorFlow. """
    global PAD_ID
    global GO_ID
    global EOS_ID
    global UNK_ID
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3


# Log summation


def log_sum_tropical_semiring(vals):
    """Approximates summation in log space with the max.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return max(vals)


def log_sum_log_semiring(vals):
    """Uses the ``logsumexp`` function in scipy to calculate the log of
    the sum of a set of log values.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return logsumexp(numpy.asarray([val for val in vals]))


#log_sum = log_sum_log_semiring
log_sum = log_sum_log_semiring
"""Defines which log summation function to use. """


def oov_to_unk(seq, vocab_size, unk_idx=UNK_ID):
    return [x if x < vocab_size else unk_idx for x in seq]

# Maximum functions

def argmax_n(arr, n):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)[:n]
    else:
        return numpy.argpartition(arr, -n)[-n:]


def argmax(arr):
    """Get the index of the maximum entry in ``arr``. The parameter can
    be a dictionary.
    
    Args:
        arr (list,array,dict):  Set of numerical values
    
    Returns:
        Index or key of the maximum entry in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.iteritems(), key=operator.itemgetter(1))[0]
    else:
        return numpy.argmax(arr)


# Functions for common access to numpy arrays, lists, and dicts
    

def common_viewkeys(obj):
    """Can be used to iterate over the keys or indices of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        return obj.viewkeys()
    else:
        return xrange(len(obj))


def common_iterable(obj):
    """Can be used to iterate over the key-value pairs of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        for key, value in obj.iteritems():
            yield key, value
    else:
        for index, value in enumerate(obj):
            yield index, value


def common_get(obj, key, default):
    """Can be used to access an element via the index or key.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
        ``default`` (object): Default return value if ``key`` not found
    
    Returns:
        ``obj[key]`` if ``key`` in ``obj``, otherwise ``default``
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default

def common_contains(obj, key):
    """Checks the existence of a key or index in a mapping.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
    
    Returns:
        ``True`` if ``key`` in ``obj``, otherwise ``False``
    """
    if isinstance(obj, dict):
        return key in obj
    else:
        return key < len(obj)


# Miscellaneous


def get_path(tmpl, sub = 1):
    """Replaces the %d placeholder in ``tmpl`` with ``sub``. If ``tmpl``
    does not contain %d, return ``tmpl`` unmodified.
    
    Args:
        tmpl (string): Path, potentially with %d placeholder
        sub (int): Substitution for %d
    
    Return:
        string. ``tmpl`` with %d replaced with ``sub`` if present
    """
    if "%d" in tmpl:
        return tmpl % sub
    return tmpl


MESSAGE_TYPE_DEFAULT = 1
"""Default message type for observer messages """


MESSAGE_TYPE_POSTERIOR = 2
"""This message is sent by the decoder after ``apply_predictors`` was
called. The message includes the new posterior distribution and the
score breakdown. 
"""


MESSAGE_TYPE_FULL_HYPO = 3
"""This message type is used by the decoder when a new complete 
hypothesis was found. Note that this is not necessarily the best hypo
so far, it is just the latest hypo found which ends with EOS.
"""


class Observer(object):
    """Super class for classes which observe (GoF design patten) other
    classes.
    """
    
    @abstractmethod
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Get a notification from an observed object.
        
        Args:
            message (object): the message sent by observed object
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        raise NotImplementedError
    

class Observable(object):
    """For the GoF design pattern observer """
    
    def __init__(self):
        """Initializes the list of observers with an empty list """
        self.observers = []
    
    def add_observer(self, observer):
        """Add a new observer which is notified when this class fires
        a notification
        
        Args:
            observer (Observer): the observer class to add
        """
        self.observers.append(observer)
    
    def notify_observers(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Sends the given message to all registered observers.
        
        Args:
            message (object): The message to send
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        for observer in self.observers:
            observer.notify(message, message_type)
    
    