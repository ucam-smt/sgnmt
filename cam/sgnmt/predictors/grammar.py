"""This module contains everything related to the hiero predictor. This
predictor allows applying rules from a syntactical SMT system directly
in SGNMT. The main interface is ``RuleXtractPredictor`` which can be 
used like other predictors during decoding. 
The Hiero predictor follows are the LRHiero implementation from 

https://github.com/sfu-natlang/lrhiero

  Efficient Left-to-Right Hierarchical Phrase-based Translation with 
  Improved Reordering. 
  Maryam Siahbani, Baskaran Sankaran and Anoop Sarkar. 
  EMNLP 2013. Oct 18-21, 2013. Seattle, USA.

However, note that we modified the code to 
a) deal with an arbitrary number of non-terminals
b) work with ruleXtract
c) allow spurious ambiguity

ATTENTION: This implementation is experimental!!
"""

from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils
import logging
import re
import gzip

class Cell:
    """Comparable to a CYK cell: A set of hypotheses. If duplicates are
    added, we do hypo combination by combining the costs and retraining
    only one of them. Internally, the hypotheses are stored in a list
    sorted by the sum of the translation prefix
    """
    
    def __init__(self, init_hypo=None):
        """Creates a new ``Cell`` with only one hypothesis.
        
        Args:
            init_hypo (LRHieroHypothesis): Initial hypothesis
        """
        self.hypos = [init_hypo] if init_hypo else []
    
    def findIdx(self, key, a, b):
        """Find index of first element with given key. If there is no
        such key, return last element with largest key smaller than key
        This is a recursive function which only searches in the 
        interval [a,b]
        """
        if b == a:
            return a
        idx = int(a + (b-a)/2)
        idx_key = self.hypos[idx].key
        if key > idx_key:
            return self.findIdx(key, idx+1, b)
        else: # key <= idx_key
            return self.findIdx(key, a, idx)

    def add(self, hypo):
        """Add a new hypothesis to the cell. If an equivalent 
        hypothesis already exists, combine both hypotheses.
        
        Args:
            hypo (LRHieroHypothesis): Hypothesis to add under the key
                                      ``hypo.key``
        """
        n_hypos = len(self.hypos)
        idx = self.findIdx(hypo.key, 0, n_hypos)
        while idx < n_hypos and self.hypos[idx].key == hypo.key:
            if hypo == self.hypos[idx]: # Hypo combination
                self.hypos[idx].cost = max(self.hypos[idx].cost, hypo.cost)
                #print("HYPO COMBINATION")
                return
            idx += 1
        self.hypos.insert(idx, hypo)
    
    def filter(self, pos, symb):
        """Remove all hypotheses which do not have ``symb`` at ``pos``
        in their ``trgt_prefix``. Breaks if ``pos`` is out of range for
        some ``trgt_prefix``
        """
        self.hypos = [hypo for hypo in self.hypos if hypo.trgt_prefix[pos] == symb]
    
    def pop(self):
        """Removes a hypothesis from the cell.
        
        Returns:
            LRHieroHypothesis. The removed hypothesis
        """
        return self.hypos.pop()
    
    def __nonzero__(self):
        """Cell is zero if its empty. """
        return True if self.hypos else False


class Node:
    """Represents a node in the Trie. """
    def __init__(self):
        self.terminal_edges = {} # outgoing edges with terminal symbols
        self.nonterminal_edges = {} # outgoing edges with non-terminal symbols
        self.elements = [] # rules at this node
    
        
class Trie:
    """This trie implementation allows matching NT symbols with arbitrary 
    symbol sequences with certain lengths when searching.
    Note: This trie does not implement edge collapsing - each edge is
    labeled with exactly one word
    """
    
    def __init__(self, span_len_range):
        """Creates an empty trie data structure.
        
        Args:
            span_len_range (tuple): minimum and maximum span lengths
                                    for non-terminal symbols
        """
        self.root = Node()
        self.span_len_range = span_len_range # Explicitly no deep copy
    
    def _get_node(self, seq):
        """Search for the node in the data structure which matches the
        key ``seq``. This allows for non-terminals in ``seq`` which
        are marked with negative IDs.
        """
        cur_node = self.root
        for token_id in seq:
            children = cur_node.terminal_edges
            if token_id < 0:
                children = cur_node.nonterminal_edges
                token_id = -token_id
            if not token_id in children:
                children[token_id] = Node()
            cur_node = children[token_id]
        return cur_node
    
    def add(self, seq, element):
        """Add an element to the trie data structure. The key sequence
        ``seq`` can contain non-terminals with negative IDs. If a
        element with the same key already exists in the data structure,
        we do not delete it but store both items.
        
        Args:
            seq (list): Sequence of terminals and non-terminals used as
                        key in the trie
            element (object): Object to associate with ``seq``
        """
        self._get_node(seq).elements.append(element)
    
    def replace(self, seq, element):
        """Replaces all elements stored at a ``seq`` with a new single
        element ``element``. This is equivalent to first removing all
        items with key ``seq``, and then add the new element with
        ``add(seq, element)``
        
        Args:
            seq (list): Sequence of terminals and non-terminals used as
                        key in the trie
            element (object): Object to associate with ``seq``
        """
        self._get_node(seq).elements = [element]
        
    def get_all_elements(self):
        """Retrieve all elements stored in the trie """
        return self._get_all_elements_recursive(self.root)
    
    def _get_all_elements_recursive(self, node):
        """Recursive helper function for ``get_all_elements`` which
        traverses the trie ignoring the arc labels.
        """
        els = node.elements
        for child in node.nonterminal_edges.values():
            els = els + self._get_all_elements_recursive(child)
        for child in node.terminal_edges.values():
            els = els + self._get_all_elements_recursive(child)
        return els
    
    def get_elements(self, src_seq):
        """Get all elements (e.g. rules) which match the given sequence
        of source tokens.
        
        Args:
            seq (list): Sequence of terminals and non-terminals used as
                        key in the trie
        
        Returns:
            two dicts: ``(rules, nt_span_lens)``. The first dictionary
            contains all applying rules. ``nt_span_lens`` lists the 
            number of symbols each of the NTs on the source side 
            covers. Make sure that ``self.span_len_range`` is updated
        """
        self.matching_elements = {}
        self.matching_nt_span_lens = {}
        self._get_elements_recursive(self.root, src_seq, [])
        return (self.matching_elements, self.matching_nt_span_lens)
    
    def _get_elements_recursive(self, node, src_seq, nt_span_lens):
        """Recursive helper function for ``get_elements``. Fills up the
        ``matching_elements`` variable. """
        if not src_seq:
            for rule in node.elements:
                if not rule.id in self.matching_elements:
                    self.matching_elements[rule.id] = rule
                    self.matching_nt_span_lens[rule.id] = []
                self.matching_nt_span_lens[rule.id].append(nt_span_lens)
            return
        token_id = src_seq[0]
        if token_id in node.terminal_edges: # Exact matches
            self._get_elements_recursive(node.terminal_edges[token_id],
                                         src_seq[1:],
                                         nt_span_lens)
        for nt_id, child in node.nonterminal_edges.iteritems():
            (min_span_len, max_span_len) = self.span_len_range[nt_id]
            max_span_len = min(len(src_seq), max_span_len)
            for span_len in xrange(min_span_len, max_span_len + 1):
                self._get_elements_recursive(child,
                                             src_seq[span_len:],
                                             nt_span_lens + [span_len])
        

class Span:
    """Span is defined by the start and end position and the 
    corresponding sequence of terminal and non-terminal symbols p. 
    Normally, p is just a single NT symbol. However, if there is 
    ambiguity with how to apply a rule to a span (e.g. 
    rule X -> X the X to span foo the bar the baz) we allow to resolve
    them later on demand. In this case, p = X the X
    """
    
    def __init__(self, p, borders):
        """Fully initializes a new ``Span`` instance.
        
        Args:
            p (list): See class docstring for ``Span``
            borders (tuple): (begin, end) with begin inclusive and end
                             exclusive
        """
        self.p = p
        self.borders = borders
    
    def __repr__(self):
        """Returns a string representation of the span/ """
        return "%s:%s" % (self.p, self.borders)
    
    def __eq__(self, other):
        """Two spans are equivalent if ``p`` and ``borders`` match, and
        if ``trgt_src_map`` matches in case they are defined.
        """
        map_match = True
        try:
            map_match = (self.trgt_src_map == other.trgt_src_map)
        except AttributeError:
            map_match = True
        return self.p == other.p and self.borders == other.borders and map_match


class LRHieroHypothesis:
    """Represents a LRHiero hypothesis, which is defined by the 
    accumulated cost, the target prefix, and open source spans.
    """
    
    def __init__(self, trgt_prefix, spans, cost):
        """Creates a new LRHiero hypothesis
        
        Args:
            trgt_prefix (list): Target side translation prefix, i.e.
                                the partial target sentence which is
                                translated so far
            spans (list): List of spans which are not covered yet, in 
                          left-to-right order on target side
            cost (float): Cost of this partial hypothesis
        """
        self.trgt_prefix = trgt_prefix 
        self.spans = spans # 
        self.cost = cost
        self.key = sum(trgt_prefix)
        
    def is_final(self):
        """Returns true if this hypothesis has no open spans """
        return len(self.spans) == 0
    
    def __repr__(self):
        """Returns a string representation of the hypothesis """
        return "%s: %s (%d)" % (
                self.trgt_prefix, self.spans, self.cost)
        
    def __eq__(self, other):
        """True if translation prefix and set of spans are equal """
        return self.trgt_prefix == other.trgt_prefix and self.spans == other.spans


class Rule:
    """A rule consists of ``rhs_src`` and ``rhs_trgt``, both are 
    sequences of integers. NTs are indicated with negative sign. The 
    ``trgt_src_map`` defines which NT on the target side belongs to 
    which NT on the source side.
    """
    
    last_id = 0 # Used for assigning unique rule indices
    
    def __init__(self, rhs_src, rhs_trgt, trgt_src_map, cost):
        """Creates a new rule.
        
        Args:
            rhs_src (list): Source on the right hand side of the rule
            rhs_trgt (list): Target on the right hand side of the rule
            trgt_src_map (dict): Defines which NT on the target side
                                 belongs to which NT on the source side
        """
        self.rhs_src = rhs_src
        self.rhs_trgt = rhs_trgt
        self.trgt_src_map = trgt_src_map
        self.cost = cost
        Rule.last_id += 1
        self.id = Rule.last_id 
    
    def __repr__(self):
        """Returns a string representation of the rule. """
        return "%d (%d): < %s , %s > (%s)" % (self.id,
                                              self.cost,
                                              self.rhs_src,
                                              self.rhs_trgt,
                                              self.trgt_src_map)


class RuleSet:
    """This class stores the set of rules and provides efficient retrieval and
    matching functionality
    """
        
    INF = 10000
    
    def __init__(self):
        """Initializes the set by setting up the trie data structure 
        for storing the rules.
        """
        # Note: NT ids start with 1, so we need to add dummy element to lists
        # Stores minimum and maximum length of each span under non-terminal
        self.span_len_range = [(1,1)] 
        # Productions. Maps LHS to rule object.
        self.tries = [Trie(self.span_len_range)]
        # Maps non-terminal to its NT id (not to be confused with word ids!) 
        self.nt2id = {'dummy': 0} 
        self.regex = re.compile("^[^0-9]+")
        self.span_len_range_updated = True
        # Number of parsed but discarded rules (because not in GNF)
        self.n_discarded = 0 
        self.n_rules = 0 # Number of rules
    
    def update_span_len_range(self):
        """This method updates the ``span_len_range`` variable by 
        finding boundaries for the spans each non terminal can cover. 
        This is done iteratively: First, guess the range for each NT to
        (0, inf). Then, iterate through all rules for a specific NT and
        adjust the boundaries given the ranges for all other NTs. Do 
        this until ranges do not change anymore. This is an expensive 
        operation should be done after adding all rules. Note also that
        the tries store a reference to ``self.span_len_range``, i.e. 
        the variable is propagated to all tries automatically.
        """
        # Build a list of multipliers for each rule for each NT
        n_nt = len(self.tries)
        multipliers = [[]] # with dummy element at index 0
        ranges = [(1,1)] # Guardian for lexical counts
        empty_src_side = False
        for nt_id in xrange(1, n_nt):
            # We'll use it as normal trie here without NT matching
            multis = Trie(self.span_len_range) 
            for rule in self.tries[nt_id].get_all_elements():
                multiplier = n_nt * [0]
                if not rule.rhs_src:
                    empty_src_side = True
                # lexical counts are stored at index 0
                for token_id in rule.rhs_src: 
                    multiplier[0 if token_id > 0 else (-token_id)] += 1
                multis.replace(multiplier, multiplier) # Add if not there yet
            multipliers.append(multis.get_all_elements())
            ranges.append((0, RuleSet.INF))
        # No production goes to <eps>: minimum length is 1
        if not empty_src_side: 
            # cannot be inferred sometimes by the while loop below
            ranges = [(1, RuleSet.INF) for _ in ranges] 
        changed = True
        while changed:
            changed = False
            for nt_id in xrange(1, n_nt):
                (old_min_len, old_max_len) = ranges[nt_id]
                if old_min_len == old_max_len:
                    continue
                min_len = RuleSet.INF
                max_len = 0
                for multiplier in multipliers[nt_id]:
                    min_len = min(min_len, 
                                  sum([ranges[idx][0]*w 
                                        for idx, w in enumerate(multiplier)]))
                    max_len = max(max_len,
                                  sum([ranges[idx][1]*w 
                                        for idx, w in enumerate(multiplier)]))
                max_len = min(max_len, RuleSet.INF)
                if min_len != old_min_len or max_len != old_max_len:
                    ranges[nt_id] = (min_len, max_len)
                    changed = True
        for nt_id in xrange(1, n_nt):
            logging.info("Range for non-terminal %d: %s" % (nt_id,
                                                            ranges[nt_id]))
            self.span_len_range[nt_id] = ranges[nt_id]
        self.span_len_range_updated = True
        
    def expand_hypo(self, hypo, src_seq):
        """Similar to ``getSpanRules()`` and ``GrowHypothesis()`` in 
        Alg. 1 in (Siahbani, 2013) combined. Gets all rules which match
        the given span. 
        
        * If the p parameter of the span is a single non-terminal, we 
          return hypotheses resulting from productions of this non-
          terminal. Note that rules might be applicable in many different
          ways: X-> A the B can be applied to foo the bar the baz in two 
          ways. In this case, we add the translation prefix, but leave the
          borders of the span untouched, and change the ``p`` value to 
          ``thr rhs`` of the production (i.e. "A the B"). If p consists
          of multiple characters, the spans store the minimum and maximum
          *length*, not the begin and end since the exact begin and end
          positions are variable.
        * If the p parameter of the span has length > 1, we return a 
          set of hypotheses in which the first subspan has a single NT
          as p parameter.
        
        
        Through this contract we can e.g. handle spurious ambiguity, if 
        two NT are on the source side. However, resolving this 
        ambiguity is implemented in a lazy fashion: we delay fixing the 
        span boundaries until we need to expand the hypothesis once 
        more, and then we fix only the first boundaries for the first 
        span.
        
        Args:
            hypo (LRHieroHypothesis): Hypothesis to expand
            src_seq (list): Source sequence to match
        """
        if not self.span_len_range_updated:
            self.update_span_len_range()
        
        span = hypo.spans.pop(0)
        if len(span.p) == 1:
            return self._expand_hypo_single_p(hypo, span, src_seq)
        else:
            return self._expand_hypo_multi_p(hypo, span, src_seq)
    
    def _expand_hypo_single_p(self, hypo, span, src_seq):
        """Helper function for ``expand_hypo`` if p has length 1 """
        base_spans = hypo.spans 
        (begin, end) = span.borders
        (rules, nt_span_lens) = self.tries[-span.p[0]].get_elements(
                                                            src_seq[begin:end])
        new_hypos = []
        for rule_id in rules:
            rule = rules[rule_id]
            span_lens = nt_span_lens[rule_id]
            trgt_prefix = [word for word in rule.rhs_trgt if word >= 0]
            if len(span_lens) == 1: # Phew, rule application not ambiguous
                src_sorted_spans = []
                cur_pos = begin
                cur_idx = 0
                for span_len in span_lens[0]:
                    while rule.rhs_src[cur_idx] >= 0:
                        cur_pos += 1
                        cur_idx += 1
                    src_sorted_spans.append(Span([rule.rhs_src[cur_idx]],
                                                 (cur_pos,
                                                  cur_pos + span_len)))
                    cur_pos += span_len
                    cur_idx += 1
                spans = [src_sorted_spans[src_pos] 
                                for src_pos in rule.trgt_src_map]
            else: # Ambiguity. Set p to rhs(src)
                span = Span(rule.rhs_src, (begin, end))
                # Gonna need trgt_src_map in expand_hypo_multi_p
                span.trgt_src_map = rule.trgt_src_map 
                spans = [span]
            new_hypos.append(LRHieroHypothesis(
                                    hypo.trgt_prefix + trgt_prefix,
                                    spans + base_spans,
                                    rule.cost + hypo.cost))
        return new_hypos
    
    def _expand_hypo_multi_p(self, hypo, span, src_seq):
        """This method creates hypotheses where p of the first span is
        a single non-terminal. We try to resolve as least ambiguity as 
        possible, i.e. to return hypotheses with as many spans with 
        multi-symbol p as possible. We can leave ambiguity unresolved 
        as long as a *continuous* sequence on the source side is mapped
        to a *continuous* sequence on the target side, as long as both 
        do not contain the very first span of the root span according
        to target side ordering."""
        # Check terminal postfix in p if there is any, and remove it
        span_from, span_to = span.borders
        term_postfix_len = next((idx for idx, el in enumerate(reversed(span.p)) 
                                        if el < 0), len(span.p))
        #print("term_postfix_len: %d " % term_postfix_len)
        if term_postfix_len > 0: 
            if not span.p[-term_postfix_len:] == src_seq[
                                            span_to-term_postfix_len:span_to]:
                return [] 
            span_to -= term_postfix_len
            span.p = span.p[0:len(span.p)-term_postfix_len]
            span.borders = (span_from, span_to)
        # trivial case: p contains no NTs
        if not span.p and span_to - span_from == 0: 
            return [hypo] # We return the original hypo because p matched span
        # First, create spans array containing all new sub-spans retaining as
        # much ambiguity as possible - borders store min and max span len
        minmax_spans, trgt_src_map, prefixes = self._factorize_first_nt(span)
        src_trgt_map = [0] * len(trgt_src_map)
        for idx, val in enumerate(trgt_src_map):
            src_trgt_map[val]= idx 
        # Then, get all applicable combinations of sub spans with concrete 
        # begin and end
        spans_list = self._get_spans_from_minmax_recursive(
                                minmax_spans, 
                                prefixes,
                                src_trgt_map,
                                span_to - span_from, src_seq, span_from, 0, [])
        new_hypos = []
        base_spans = hypo.spans
        for src_ordered_spans in spans_list:
            new_hypos.append(LRHieroHypothesis(
                hypo.trgt_prefix,
                [src_ordered_spans[idx] 
                    for idx in trgt_src_map] + base_spans, hypo.cost))
        return new_hypos
    
    def _get_spans_from_minmax_recursive(self,
                                         minmax_spans,
                                         prefixes,
                                         src_trgt_map, 
                                         span_len_sum,
                                         src_seq,
                                         src_idx,
                                         span_idx,
                                         previous_spans):
        """Recursive helper function to get concrete spans
        from minmax spans
        """ 
        # Frist check prefix
        pref = prefixes[src_trgt_map[span_idx]]
        pref_len = len(pref)
        if not pref == src_seq[src_idx:src_idx+pref_len]:
            return []
        src_idx += pref_len 
        span_len_sum -= pref_len
        minmax_span = minmax_spans[src_trgt_map[span_idx]]
        p = minmax_span.p
        if span_idx == len(src_trgt_map) - 1: # At last span
            span = Span(p, (src_idx, src_idx+span_len_sum))
            if len(p) > 1:
                span.trgt_src_map = minmax_span.trgt_src_map
            if (span_len_sum >= minmax_span.borders[0] 
                    and span_len_sum <= minmax_span.borders[1]
                    and self._is_compatible(
                                    p,
                                    src_seq[src_idx:src_idx+span_len_sum])):
                return [previous_spans + [span]]
            return []
        else:
            ret = []
            min_len, max_len = minmax_span.borders[0], min(span_len_sum,
                                                           minmax_span.borders[1])
            for span_len in xrange(min_len, max_len+1):
                if self._is_compatible(p, src_seq[src_idx:src_idx+span_len]):
                    span = Span(p, (src_idx, src_idx+span_len))
                    if len(p) > 1:
                        span.trgt_src_map = minmax_span.trgt_src_map
                    ret += self._get_spans_from_minmax_recursive(
                                                    minmax_spans,
                                                    prefixes,
                                                    src_trgt_map,
                                                    span_len_sum-span_len,
                                                    src_seq,
                                                    src_idx+span_len,
                                                    span_idx+1,
                                                    previous_spans + [span])
            return ret 
    
    def _is_compatible(self, p, src_seq):
        """Checks if terminals in p can be matched in ``src_seq`` """
        src_idx = 0
        for symb in p:
            if symb >= 0: # search for it in src_seq
                if src_idx >= len(src_seq):
                    return False
                while src_seq[src_idx] != symb:
                    src_idx += 1
                    if src_idx >= len(src_seq):
                        return False
                src_idx += 1
        return True         
        
    def _factorize_first_nt(self, span):
        """Given span must have multi-symbol p. Returns a set of spans
        which can replace the given span if the first non-terminal
        (according target side ordering) is to be isolated. Borders of 
        returned span objects stand for minimum and maximum span 
        lengths
        """
        p_len = len(span.p)
        p_nts = [symb for symb in span.p if symb < 0]
        p_nt_pos = [pos for pos, symb in enumerate(span.p) if symb < 0]
        first_nt = -p_nts[span.trgt_src_map[0]]
        first_span = Span([-first_nt], self.span_len_range[first_nt])
        spans = [first_span]
        trgt_idx = 1
        # number of non-terminals
        n_nt = len(span.trgt_src_map) 
        # trgt_src map describing the ordering of the newly created spans
        new_trgt_src_map = [span.trgt_src_map[0]]
        # Stores parts of p which are covered by spans 
        p_covered = [RuleSet.INF] * p_len 
        p_covered[p_nt_pos[span.trgt_src_map[0]]] = 0
        while trgt_idx < n_nt:
            span_min_len, span_max_len = (0, 0)
            from_src_pos, to_src_pos = (RuleSet.INF, 0)
            from_src_idx, to_src_idx = (RuleSet.INF, 0)
            new_internal_trgt_src_map = []
            while True:
                src_idx = span.trgt_src_map[trgt_idx]
                new_internal_trgt_src_map.append(src_idx)
                nt = -p_nts[src_idx]
                nt_min_len, nt_max_len = self.span_len_range[nt]
                span_min_len, span_max_len = (span_min_len + nt_min_len,
                                              span_max_len + nt_max_len)
                src_pos = p_nt_pos[src_idx]
                from_src_pos, to_src_pos = (min(from_src_pos, src_pos),
                                            max(to_src_pos, src_pos))
                from_src_idx, to_src_idx = (min(from_src_idx, src_idx),
                                            max(to_src_idx, src_idx))
                
                trgt_idx += 1
                if trgt_idx >= n_nt:
                    break
                src_idx = span.trgt_src_map[trgt_idx]
                if src_idx < from_src_idx - 1 or src_idx > to_src_idx + 1:
                    break
            # len of p - no. of NT
            n_terminals = to_src_pos+1-from_src_pos-len(new_internal_trgt_src_map) 
            for i in xrange(from_src_pos,to_src_pos+1):
                p_covered[i] = len(spans)
            new_span = Span(span.p[from_src_pos:to_src_pos+1],
                            (span_min_len+n_terminals,
                             min(span_max_len+n_terminals, RuleSet.INF)))
            new_span.trgt_src_map = [idx-from_src_idx 
                                        for idx in new_internal_trgt_src_map]
            spans.append(new_span)
            new_trgt_src_map.append(from_src_idx)
        # Create prefix array using p_covered
        prefixes = [[] for _ in spans]
        i = 0
        while i < p_len:
            prefix_len = 0
            while i+prefix_len < p_len and p_covered[i+prefix_len] == RuleSet.INF:
                prefix_len += 1
            if i+prefix_len < p_len:
                prefixes[p_covered[i+prefix_len]] = span.p[i:i+prefix_len]
            i += prefix_len+1 
        # new_trgt_src_map still stores source indices (with holes). Remove 
        # holes s.t. it is compatible with the created spans list
        return spans, self._remove_holes_in_list(new_trgt_src_map), prefixes
    
    def _remove_holes_in_list(self, l):
        # could also use dictionary here, but holes are rather small
        d = [0] * (max(l)+1) 
        for idx, val in enumerate(sorted(l)):
            d[val] = idx
        return [d[val] for val in l]
        
    def _get_nt_id(self, nt_name):
        if nt_name in self.nt2id:
            return self.nt2id[nt_name]
        # Introduce new NT id
        nt_id = len(self.tries)
        self.tries.append(Trie(self.span_len_range))
        self.nt2id[nt_name] = nt_id
        logging.info("Found new non-terminal symbol %s (id: %d)" % (nt_name,
                                                                    nt_id))
        self.span_len_range.append((0, RuleSet.INF))
        return nt_id
        
    def create_rule(self, rhs_src, rhs_trgt, weight):
        """Creates a rule object (factory method)
        
        Args:
            rhs_src (list): String sequence describing the source of 
                            the right-hand-side of the rule
            rhs_trgt (list): String sequence describing the target of 
                             the right-hand-side of the rule
            weight (float): Rule weight
        
        Returns:
            ``Rule`` or ``None`` if something went wrong
        """
        src_seq = []
        nt_pos = {}
        nt_ids = {}
        for token in rhs_src:
            if token.isdigit():
                src_seq.append(int(token))
            elif token == "<oov>":
                src_seq.append(utils.UNK_ID)
            else:
                m = self.regex.match(token)
                nt_id = self._get_nt_id(m.group())
                src_seq.append(-nt_id)
                nt_pos[token] = len(nt_pos)
                nt_ids[token] = nt_id
        trgt_seq = []
        trgt_src_map = []
        nt_seen = False
        for token in rhs_trgt:
            if token == "<oov>":
                token = str(utils.UNK_ID)
            if not token.isdigit():
                trgt_src_map.append(nt_pos[token])
                trgt_seq.append(-nt_ids[token])
                nt_seen = True
            elif not nt_seen:
                trgt_seq.append(int(token))
            else:
                self.n_discarded = self.n_discarded + 1
                return None
        return Rule(src_seq, trgt_seq, trgt_src_map, weight)
                
        
    def parse(self, line, feature_weights = None):
        """Parse a line in a rule file from ruleXtract and add the rule
        to the set.
        
        Args:
            line (string). Line in the rules file
            feature_weights (list). Feature weights to compute the rule
                                    score or ``None`` to use uniform
                                    weights
        """
        stripped = line.strip()
        if not stripped or stripped[0] == '#':
            return
        parts = stripped.split()
        if len(parts) < 4: # Do not complain.. maybe empty line
            logging.warn("Parsing error in rule file: less than four columns")
        try:
            weights = [float(feat) for feat in parts[3:]]
            if feature_weights:
                weights = [f*w for (f,w) in zip(weights, feature_weights)]
            weight = sum(weights)
        except ValueError:
            logging.warn("Parsing error in rule file: non-numeric weights")
            return
        rule = self.create_rule(
                        [] if parts[1] == "<dr>" else parts[1].split("_"),
                        [] if parts[2] == "<dr>" else parts[2].split("_"),
                        weight)
        if rule:
            self.n_rules = self.n_rules + 1
            self.tries[self._get_nt_id(parts[0])].add(rule.rhs_src, rule)
            self.span_len_range_updated = False
        

class RuleXtractPredictor(Predictor):
    """Predictor based on ruleXtract rules. Bins are organized 
    according the number of target words. We assume that no rule 
    produces the empty word on the source side (but possibly on the 
    target side). Hypotheses are produced iteratively s.t. the 
    following invariant holds: The bins contain a set of (partial) 
    hypotheses from which we can derive all full hypotheses which are 
    consistent with the current target prefix (i.e. the prefix of the
    target sentence which has already been translated). This set is
    updated when calling either consume_word or predict_next: consume\_
    word deletes all hypotheses which become inconsistent with the new 
    word. ``predict_next`` requires all hypotheses to have a target\_
    prefix length of at least one plus the number of consumed words. 
    Therefore, ``predict_next`` expands hypotheses as long as they are
    shorter. This fits nicely with grouping hypotheses in bins of same
    target prefix length: we expand until all low rank bins are empty. 
    We predict the next target word by using the cost of the best 
    hypothesis with the word at the right position. 
    
    Note that this predictor is similar to the decoding algorithm in
    
      Efficient Left-to-Right Hierarchical Phrase-based Translation with
      Improved Reordering. 
      Maryam Siahbani, Baskaran Sankaran and Anoop Sarkar. 
      EMNLP 2013. Oct 18-21, 2013. Seattle, USA.
    
    without cube pruning, but it is extended to an arbitrary number of
    non-terminals as produced with ruleXtract.
    """
    
    def __init__(self, ruleXtract_path, use_weights, feature_weights = None):
        """Creates a new hiero predictor.
        
        Args:
            ruleXtract_path (string): Path to the rules file
            use_weights (bool): If false, set all hypothesis scores 
                                uniformly to 0 (= log 1). If true,
                                use the rule weights to compute
                                hypothesis scores
            feature_weights (list): Rule feature weights to compute
                                    the rule scores. If this is none
                                    we use uniform weights
        """
        super(RuleXtractPredictor, self).__init__()
        self.use_weights = use_weights
        self.rules = RuleSet()
        with (gzip.open(ruleXtract_path) if ruleXtract_path[-3:] == '.gz' 
                                         else open(ruleXtract_path)) as f:
            for line in f:
                self.rules.parse(line, feature_weights)
        self.rules.update_span_len_range()
        logging.info("%d rules loaded (%d discarded because not in GNF)" %
            (self.rules.n_rules, self.rules.n_discarded))
        if not 'S' in self.rules.nt2id:
            logging.fatal("No rule with start symbol S found!") 
        self.start_nt = self.rules.nt2id['S']
        logging.debug("Grammar start symbol: S (ID: %d)" % self.start_nt)

    def get_unk_probability(self, posterior):
        """Returns negative infinity if the posterior is not empty as
        words outside the grammar are not possible according this
        predictor. If ``posterior`` is empty, return 0 (= log 1)
        """ 
        return utils.NEG_INF
    
    def predict_next(self):
        """For predicting the distribution of the next target tokens, 
        we need to empty the stack with the current history length
        by expanding all hypotheses on it. Then, all hypotheses are
        in larger bins, i.e. have a longer target prefix than the
        current history. Thus, we can look up the possible next words
        by iterating through all active hypotheses.
        """
        # If there are still partial hypotheses...
        if self.n_consumed < len(self.stacks):
            # empty stack with n_consumed trgt_prefix length
            while self.stacks[self.n_consumed]:
                hypo = self.stacks[self.n_consumed].pop()
                #print("EXPAND: %s" % hypo)
                new_hypos = self.rules.expand_hypo(hypo, self.src_seq)
                for new_hypo in new_hypos:
                    n_covered = len(new_hypo.trgt_prefix)
                    if new_hypo.is_final():
                        while len(self.finals) <= n_covered:
                            self.finals.append(Cell())
                        # Make sure that it ends with EOS
                        new_hypo.trgt_prefix[-1] = utils.EOS_ID
                        self.finals[n_covered].add(new_hypo)
                    else:
                        while len(self.stacks) <= n_covered:
                            self.stacks.append(Cell())
                        self.stacks[n_covered].add(new_hypo)
        logging.debug("Predict next (consumed: %d)" % self.n_consumed)
        for idx,c in enumerate(self.stacks):
            if c.hypos:
                logging.debug("Stack %d: %d" % (idx, len(c.hypos)))
        for idx,c in enumerate(self.finals):
            if c.hypos:
                logging.debug("Finals %d: %s" % (idx, c.hypos))
        return self.build_posterior()
    
    def build_posterior(self):
        """We need to scan all hypotheses in ``self.stacks`` and add up
        scores grouped by the symbol at the n_consumed+1-th position.
        Then, we add end-of-sentence probability by checking 
        ``self.finals[n_consumed]``
        """
        posterior = {}
        for stack_idx in xrange(self.n_consumed+1, len(self.stacks)):
            for hypo in self.stacks[stack_idx].hypos:
                symb = hypo.trgt_prefix[self.n_consumed]
                posterior[symb] = max(posterior.get(symb, 0), hypo.cost)
        if self.n_consumed+1 < len(self.finals) and self.finals[self.n_consumed+1]:
            posterior[utils.EOS_ID] = max([hypo.cost
                            for hypo in self.finals[self.n_consumed+1].hypos])
        return self.finalize_posterior(posterior, self.use_weights, False)
    
    def initialize(self, src_sentence):
        """Delete all bins and add the initial cell to the first bin """
        self.reset()
        self.src_seq = [utils.GO_ID] + src_sentence + [utils.EOS_ID]
        self.src_len = len(self.src_seq)
        span = Span([-self.start_nt], (0, self.src_len))
        init_hypo = LRHieroHypothesis([], [span], 0)
        self.stacks = [Cell(init_hypo)]
        self.finals = []
        self.predict_next()
        self.consume(utils.GO_ID)
    
    def consume(self, word):
        """Remove all hypotheses with translation prefixes which do not
        match ``word``
        """
        for stack_idx in xrange(self.n_consumed+1, len(self.stacks)):
            self.stacks[stack_idx].filter(self.n_consumed, word)
        for stack_idx in xrange(self.n_consumed+1, len(self.finals)):
            self.finals[stack_idx].filter(self.n_consumed, word)
        if self.n_consumed < len(self.finals):
            # Empty this entry, not needed anymore
            self.finals[self.n_consumed] = Cell() 
        self.n_consumed = self.n_consumed + 1
    
    def get_state(self):
        """Predictor state consists of the stacks, the completed 
        hypotheses, and the number of consumed words. """
        return self.stacks,self.finals,self.n_consumed
    
    def set_state(self, state):
        """Set the predictor state. """
        self.stacks,self.finals,self.n_consumed = state

    def reset(self):
        """Empty the stack and delete history. """
        self.stacks = []
        self.n_consumed = 0

