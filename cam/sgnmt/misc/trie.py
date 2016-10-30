"""This module contains ``SimpleTrie`` which is a generic trie
implementation based on strings of integers.
"""

from operator import itemgetter

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
    
    The implementation also supports keys in sparse representation,
    in which most of the elements in the sequence are zero (see 
    ``add_sparse``, ``get_sparse``, and ``nearest_sparse``. In this
    case, the key is a list of tuples [(dim1,val1),...(dimN,valN)].
    Internally, we store them as sequence "dim1 val1 dim2 val2..."
    Note that we assume that the tuples are ordered by dimension!
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
        """Retrieve the element for a key ``seq``.
        
        Args:
            seq (list): Query key
            
        Returns:
            object. The element which has been added along with ``seq``
            or ``None`` if the key does not exist. 
        """
        return self._get_node(seq).element
    
    def get_prefix(self, seq):
        """Get the key in the Trie with the longest common prefix with
        ``seq``.
        
        Args:
            seq (list): Query sequence
        
        Returns:
            list. The longest key in the Trie which is a prefix of 
            ``seq``.
        """
        cur_node = self.root
        prefix = []
        best_prefix = []
        for token_id in seq:
            children = cur_node.edges
            if not token_id in children:
                break
            prefix.append(token_id)
            cur_node = children[token_id]
            if cur_node.element:
                best_prefix = list(prefix)
        return best_prefix

    def _sparse2seq(self, key):
        """Transforms a key in sparse representation to a sequence
        which can be used as key in the Trie.
        """
        seq = []
        for (d,v) in key:
            seq.append(d)
            seq.append(v)
        return seq
        
    def add_sparse(self, key, element):
        """Adds an element with a key in sparse representation.
        
        Args:
            seq (list): Sparse key (list of tuples)
            element (object): The object to store for key ``seq`` 
        """
        self.add(self._sparse2seq(key), element)
    
    def get_sparse(self, key, element):
        """Retrieves an element with a key in sparse representation.
        
        Args:
            seq (list). Query key in sparse format
            
        Returns:
            object. The element which has been added along with ``seq``
            or ``None`` if the key does not exist. 
        """
        return self.get(self._sparse2seq(key), element)
    
    def nearest_sparse(self, query):
        """This method returns the element in the Trie with the closest
        key to ``query`` in terms of Euclidean distance. The efficiency
        relies on sparseness: The more zeros in the vector, the more 
        efficient. If the Trie contains an exact match, this method
        runs linear in the length of the query (i.e. independent of
        number of elements in the Trie).
        
        Args:
            query (list): Query key in sparse format
        
        Returns:
            Tuple. (object,dist) pair with the nearest element to 
            ``query`` in terms of L2 norm and the squared L2 distance.
        """
        self.best_dist = float("inf")
        self.best_element = None
        self._register_best_element = self._register_best_element_single 
        self._nearest_sparse_recursive(self._sparse2seq(query), self.root, 0.0)
        return self.best_element,self.best_dist
    
    def n_nearest_sparse(self, query, n=1):
        """This method returns the n element in the Trie with the closest
        key to ``query`` in terms of Euclidean distance. The efficiency
        relies on sparseness: The more zeros in the vector, the more 
        efficient.
        
        Args:
            query (list): Query key in sparse format
            n (int): Number of elements to retrieve
        
        Returns:
            List. List of (object,dist) pairs with the nearest element to 
            ``query`` in terms of L2 norm and the squared L2 distance.
        """
        if n <= 1:
            return [self.nearest_sparse(query)]
        self.best_dist = float("inf")
        self.best_elements = [(None, self.best_dist)] # guardian element
        self.n = n
        self._register_best_element = self._register_best_element_multi
        self._nearest_sparse_recursive(self._sparse2seq(query), self.root, 0.0)
        return self.best_elements
    
    def _register_best_element_single(self, dist, el):
        self.best_dist = dist
        self.best_element = el
        
    def _register_best_element_multi(self, dist, el):
        self.best_elements = self.best_elements[:self.n-1] + [(el, dist)]
        self.best_elements.sort(key=itemgetter(1))
        self.best_dist = self.best_elements[-1][1] 
            
    def _nearest_sparse_recursive(self, seq, root, dist):
        if dist > self.best_dist:
            return
        if not seq:
            self._dfs_for_nearest(root, dist)
            return
        if root.element:
            add_dist = sum([seq[idx]**2 for idx in xrange(1, len(seq), 2)]) 
            if dist + add_dist < self.best_dist:
                self._register_best_element(dist + add_dist, root.element)
        dim = seq[0]
        # Explore close matches first
        children = sorted(root.edges.iterkeys(), key=lambda el: (el-dim)**2)
        for child_dim in children:
            child_node = root.edges[child_dim]
            next_seq = seq[0:]
            next_dist = dist
            try:
                while child_dim > next_seq[0]:
                    next_dist += next_seq[1]**2
                    next_seq = next_seq[2:]
                if child_dim == next_seq[0]: # Exact match :)
                    c_discount = next_seq[1]
                    next_seq = next_seq[2:]
                else:
                    c_discount = 0.0
                for c,node in child_node.edges.iteritems():
                    self._nearest_sparse_recursive(next_seq, 
                                                   node,
                                                   next_dist+(c-c_discount)**2)
            except IndexError:
                for c,node in child_node.edges.iteritems():
                    self._dfs_for_nearest(node, next_dist + c*c)
    
    def _dfs_for_nearest(self, root, dist):
            """Scans the subtree under ``root`` for nearest elements. 
            ``dist`` is the distance which has already been 
            accumulated.  
            """
            if dist > self.best_dist:
                return
            if root.element:
                self._register_best_element(dist, root.element)
                return
            for child in root.edges.itervalues():
                for c,next_child in child.edges.iteritems(): 
                    self._dfs_for_nearest(next_child, dist + c*c)

