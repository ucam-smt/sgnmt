"""This file contains common basic functionality which can be used from
anywhere. This includes the definition of reserved word indices, some
mathematical functions, and helper functions to deal with the small
quirks python sometimes has.
"""

from scipy.misc import logsumexp
import numpy
import operator

# Reserved IDs


""" Reserved word ID for padding. Relict from the TensorFlow 
implementation. """
PAD_ID = None


""" Reserved word ID for the start-of-sentence symbol. """
GO_ID = 1


""" Reserved word ID for the end-of-sentence symbol. """
EOS_ID = 2


""" Reserved word ID for the unknown word (UNK). """
UNK_ID = 0


""" Reserved word ID which is currently not used. """
NOTAPPLICABLE_ID = 3


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
"""Defines which log summation function to use. """
log_sum = log_sum_log_semiring


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


# Basic Trie implementation


class SimpleNode:
    """Helper class representing a node in a ``SimpleTrie`` """
    def __init__(self):
        """Creates an empty node without children. """
        self.edges = {} # outgoing edges with terminal symbols
        self.element = None # Elements stored at this node


class SimpleTrie:
    """This is a very simple Trie implementation. It is simpler than 
    the one in ``cam.sgnmt.predictors.grammar`` because it does not 
    support non-terminals or removal. The only supported operations are
    ``add`` and ``get``, but those are implemented very efficiently. 
    For many applications (e.g. the cache in the greedy heuristic) this
    is already enough.
    """
    
    def __init__(self):
        """Creates an empty Trie data structure. """
        self.root = SimpleNode()
    
    def _get_node(self, seq):
        """Get the ```SimpleNode``` for the given sequence ``seq``. If
        the path for ``seq`` does not exist yet in the Trie, add it and
        return a reference to the newly created node. """
        cur_node = self.root
        for token_id in seq:
            children = cur_node.edges
            if not token_id in children:
                children[token_id] = SimpleNode()
            cur_node = children[token_id]
        return cur_node
    
    def add(self, seq, element):
        """Add an element to the Trie for the key ``seq``. If ``seq`` 
        already exists, override.
        
        Args:
            seq (list): Key
            element (object): The object to store for key ``seq``
        """
        self._get_node(seq).element = element
        
    def get(self, seq):
        """Retrieve the element for a key ``seq``. If the key does not
        exist, return ``None``. """
        return self._get_node(seq).element
