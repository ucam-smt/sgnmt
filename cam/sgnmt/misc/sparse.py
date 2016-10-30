"""This module adds support for sparse input or output features. In
standard NMT we normally use a one-hot-representation, and input and
output layers are lookup tables (embedding matrices). SGNMT supports
explicit definition of word representations as sparse features, in 
which more than one neuron can be activated at a time.
"""

from abc import abstractmethod
from cam.sgnmt.misc.trie import SimpleTrie
import copy
import logging
import numpy as np
import operator


def sparse_euclidean2(v1, v2):
    """Calculates the squared Euclidean distance between two sparse 
    vectors.
    
    Args:
        v1 (dict): First sparse vector
        v2 (dict): Second sparse vector
    
    Returns:
        float. Squared distance between ``v1`` and ``v2``.
    """
    d = 0.0
    for i,v in v1.iteritems():
        v = v - v2.get(i, 0.0)
        d = d + v*v
    d = d + sum([v*v for i,v in v2.iteritems() if not i in v1])
    return d


def sparse_euclidean(v1, v2):
    """Calculates the Euclidean distance between two sparse 
    vectors.
    
    Args:
        v1 (dict): First sparse vector
        v2 (dict): Second sparse vector
    
    Returns:
        float. Distance between ``v1`` and ``v2``.
    """
    return np.sqrt(sparse_euclidean2(v1, v2))


def dense_euclidean2(v1, v2):
    """Calculates the squared Euclidean distance between two dense 
    vectors.
    
    Args:
        v1 (dict): First dense vector
        v2 (dict): Second dense vector
    
    Returns:
        float. Squared distance between ``v1`` and ``v2``.
    """
    return sum([(a-b)*(a-b) for a,b in zip(v1,v2)])


def dense_euclidean(v1, v2):
    """Calculates the Euclidean distance between two sparse 
    vectors.
    
    Args:
        v1 (dict): First dense vector
        v2 (dict): Second dense vector
    
    Returns:
        float. Distance between ``v1`` and ``v2``.
    """
    return np.sqrt(dense_euclidean2(v1, v2))


class SparseFeatMap(object):
    """This is the super class for mapping strategies between sparse
    feature representations and symbolic word indices. The translation
    needs to be implemented in ``sparse2word`` and ``word2sparse``.
    """
    
    def __init__(self, dim):
        """Initializes this map.
        
        Args:
            dim (int): Dimensionality of the feature representation
        """
        self.dim = dim
        self.null_vec = np.zeros(self.dim, dtype=np.float32)
    
    @abstractmethod
    def sparse2word(self, feat):
        """Gets the word id for a sparse feature. The sparse feature 
        format is a list of tuples [(dim1,val1),..,(dimN,valN)]
        
        Args:
            feat (list): Sparse feature to look up
        Returns:
            int. Word ID of a match for ``feat`` or ``None`` is no
                 match could be found
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError
    
    @abstractmethod
    def word2sparse(self, word):
        """Gets the sparse feature representation for a word.
        
        Args:
            word (int): Word ID
        Returns:
            list. Sparse feature representation for ``word`` or ``None``
                  if the word could not be converted
        Raises:
            NotImplementedError.
        """
        raise NotImplementedError
    
    def sparse2nwords(self, feat, n=1):
        """Returns the n closest words to ``feat``. Subclasses can
        implement this method to implement search for words. The 
        default implementation returns the single best word only.
        
        Args:
            feat (list): Sparse feature
            n (int): Number of words to retrieve
        
        Returns:
            list: List of (wordid, distance) tuples with words which
                  are close to ``feat``.
        
        Note:
            The default implementation does not use the ``n`` argument
            and always returns distance 0.
        """
        return [(self.sparse2word(feat), 0.0)]
    
    def word2dense(self, word):
        """Gets the feature representation in dense format, i.e. a
        ``self.dim``-dimensional vector as list.
        
        Args:
            word(int): Word ID
        
        Returns:
            list. Dense vector corresponding to ``word`` or the null
            vector if no representation found
        """
        return self.sparse2dense(self.word2sparse(word))
    
    def words2dense(self, seq):
        """Applies ``word2dense`` to a word sequence. """
        return [self.word2dense(w) for w in seq]
    
    def dense2sparse(self, dense, eps = 0.5):
        """Converts a dense vector to a sparse vector.
        
        Args:
            dense (list): Dense vector (list of length n)
            eps (float): Values smaller than this are set to zero
                         in the sparse representation
        
        Returns:
            list. List of (dimension, value) tuples (sparse vector)
        """
        return [(d,v) for d,v in enumerate(dense) if v > eps]
    
    def sparse2dense(self, sparse):
        """Converts a sparse vector to its dense representation.
        
        Args:
            sparse (list): Sparse vector (list of tuples)
            
        Raises:
            IndexError. If the input vector exceeds the dimensionality
                        of this map
        """
        vec = copy.copy(self.null_vec)
        if sparse:
            for (d,v) in sparse:
                vec[d] = float(v)
        return vec
    
    def dense2word(self, feat):
        """Gets the word id for a dense feature.
        
        Args:
            feat (list): Dense feature vector to look up
        Returns:
            int. Word ID of a match for ``feat`` or ``None`` is no
                 match could be found
        Raises:
            NotImplementedError.
        """
        return self.sparse2word(self.dense2sparse(feat))
    
    def dense2nwords(self, feat, n=1):
        """Returns the n closest words to ``feat``. 
        
        Args:
            feat (list): Dense feature vector
            n (int): Number of words to retrieve
        
        Returns:
            list: List of (wordid, distance) tuples with words which
                  are close to ``feat``.
        
        Note:
            The default implementation does not use the ``n`` argument
            and always returns distance 0.
        """
        return self.sparse2nwords(self.dense2sparse(feat), n)
    
    def dense2words(self, seq):
        """Applies ``dense2word`` to a sequence. """
        return [self.dense2word(w) for w in seq]


class FlatSparseFeatMap(SparseFeatMap):
    """Can be used as replacement if a ``SparseFeatMap`` is required
    but you wish to use flat word ids. It overrides the dense methods
    with the identities.
    """
    
    def __init__(self, dim=0):
        """
        Args:
            dim (int): not used
        """
        super(FlatSparseFeatMap, self).__init__(0)
    
    def sparse2word(self, feat):
        """
        Raise:
            NotImplementedError.
        """
        return NotImplementedError
    
    def word2sparse(self, word):
        """
        Raise:
            NotImplementedError.
        """
        return NotImplementedError
    
    def word2dense(self, word):
        """Identity. """
        return word
    
    def words2dense(self, seq):
        """Identity. """
        return seq
    
    def dense2sparse(self, dense, eps = 0.3):
        """
        Raise:
            NotImplementedError.
        """
        return NotImplementedError
    
    def sparse2dense(self, sparse):
        """
        Raise:
            NotImplementedError.
        """
        return NotImplementedError
    
    def dense2word(self, feat):
        """Identity. """
        return feat


class TrivialSparseFeatMap(SparseFeatMap):
    """This is the null-object (GoF) implementation for
    ``SparseFeatMap``. It corresponds to the usual one-hot
    representation.
    """
    
    def __init__(self, dim):
        """Pass through to ``SparseFeatMap``.
        
        Args:
            dim (int): Dimensionality of the feature representation
                       (should be the vocabulary size)
        """
        super(TrivialSparseFeatMap, self).__init__(dim)
    
    def sparse2word(self, feat):
        """Returns feat[0][0] """
        return feat[0][0]
    
    def word2sparse(self, word):
        """Returns [(word, 1)] """
        return [(word, 1)]
    
    
class FileBasedFeatMap(SparseFeatMap):
    """This class loads the mapping from word to sparse feature from
    a file (see ``--src_sparse_feat_map`` and ``--trg_sparse_feat_map``)
    The mapping from word to feature is a simple dictionary lookup.
    
    The mapping from feature to word is implemented with a Trie based
    nearest neighbor implementation and does not require an exact
    match. However, in case of an exact match, is runs linearly in the
    number of non-zero entries in the vector.
    """
    
    def __init__(self, dim, path):
        """Loads the feature map from the file system.
        
        Args:
            dim (int). Dimensionality of the feature space
            path (string). Path to the mapping file
        
        Raises:
            IOError. If the file could not be loaded
        """
        super(FileBasedFeatMap, self).__init__(dim)
        self.f2w = None
        self.w2f = {}
        logging.info("Loading sparse feat map from %s" % path)
        with open(path) as f:
            for line in reversed(f.readlines()):
                (word_str, feat_str) = line.strip().split(None, 1)
                word = int(word_str)
                feat = []
                for e in feat_str.split(","):
                    d,v = e.split(":", 1)
                    feat.append((int(d), float(v)))
                self.w2f[word] = feat
        logging.info("Loaded %d entries from %s" % (len(self.w2f), path))

    def _load_f2w(self):
        logging.info("Building Trie with %d elements for sparse vector lookup"
                     % len(self.w2f))
        self.f2w = SimpleTrie()
        for w,f in sorted(self.w2f.items(), 
                          key=operator.itemgetter(0), 
                          reverse=True):
            # We iterate through the file in reversed ordered such that if 
            # vector representations clash we keep the first one
            # in f2w
            self.f2w.add_sparse(f, w)
    
    def sparse2word(self, feat):
        if not self.f2w:
            self._load_f2w()
        w,_ = self.f2w.nearest_sparse(feat)
        return w
    
    def sparse2nwords(self, feat, n=1):
        if not self.f2w:
            self._load_f2w()
        return self.f2w.n_nearest_sparse(feat, n)
    
    def word2sparse(self, word):
        return self.w2f.get(word, None)
