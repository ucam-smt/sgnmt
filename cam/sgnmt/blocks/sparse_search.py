"""This module is similar to the ``blocks.search`` module but supports
search with sparse features on the output side. This is significantly
different from plain word IDs because it requires a cascaded search:
First, we do beam search to get the n best output vectors. Then, we
scan the sparse feat map table for the 12 best words given the
output vectors. In vanilla NMT, the second step is not required because
the 12 best words directly correspond to the dimensions of the output
vector.
"""


from blocks.search import BeamSearch
from blocks.roles import OUTPUT
from theano import config, function
from blocks.filter import VariableFilter
from operator import itemgetter
import numpy


class SparseBeamSearch(BeamSearch):
    """This class modifies the original ``BeamSearch`` implementation
    in blocks to work with sparse features on the output side. Note 
    that this violates the Liskov substitution principle as we inherit
    only to leverage off the methods from ``BeamSearch``. However, we
    leave it to future work to agonize about that.. 
    """
    
    def __init__(self, samples, trg_sparse_feat_map):
        super(SparseBeamSearch, self).__init__(samples)
        self.trg_sparse_feat_map = trg_sparse_feat_map

    def _compile_logprobs_computer(self):
        # This filtering should return identical variables
        # (in terms of computations) variables, and we do not care
        # which to use.
        
        # fs439: use variable from emit
        logprobs = VariableFilter(
            applications=[self.generator.readout.emitter.emit],
            roles=[OUTPUT])(self.inner_cg)[0]
        # fs439: do not convert to negative log because we use squared error
        self.logprobs_computer = function(
            self.contexts + self.input_states, logprobs,
            on_unused_input='ignore')

    def search(self, input_values, eol_symbol, max_length,
               ignore_first_eol=False, as_arrays=False):
        """Performs beam search. For a full description see
        ``blocks.search.BeamSearch.search``.
        """
        if not self.compiled:
            self.compile()

        contexts, states, beam_size = self.compute_initial_states_and_contexts(
            input_values)
        shp = (states['outputs'][None, :].shape[0], states['outputs'][None, :].shape[1])
        all_masks = numpy.ones(shp, dtype=config.floatX)
        all_costs = numpy.inf * numpy.ones(shp, dtype=config.floatX)
        all_costs[0,0] = 0.0
        all_words = numpy.ones(shp, dtype=numpy.int64)
        for i in range(max_length):
            if all_masks[-1].sum() == 0:
                break

            logprobs = self.compute_logprobs(contexts, states)
            # Collect n best words
            best_words = [] # Format: (beam_idx, word, cost)
            for i,prob in enumerate(logprobs):
                base_cost = all_costs[-1,i]
                if not all_masks[-1,i]: # This one is already finished
                    best_words.append((i, eol_symbol, base_cost))
                    continue
                this_words = self.trg_sparse_feat_map.dense2nwords(prob, beam_size)
                best_words.extend([(i, w, base_cost + c) for w,c in this_words])
            chosen = sorted(best_words, key=itemgetter(2))[:beam_size]
            indexes = numpy.array([i for (i,w,c) in chosen])
            words = numpy.array([w for (i,w,c) in chosen])
            next_costs = numpy.array([c for (i,w,c) in chosen])
            outputs = numpy.array([self.trg_sparse_feat_map.word2dense(w) for (i,w,c) in chosen])
            mask = numpy.array([w != eol_symbol for (i,w,c) in chosen])
            if ignore_first_eol and i == 0:
                mask[:] = 1

            # Rearrange everything
            for name in states:
                states[name] = states[name][indexes]
            all_words = all_words[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]

            # Record chosen output and compute new states
            states.update(self.compute_next_states(contexts, states, outputs))
            all_costs = numpy.vstack([all_costs, next_costs[None, :]])
            all_masks = numpy.vstack([all_masks, mask[None, :]])
            all_words = numpy.vstack([all_words, words[None, :]])
        all_words = all_words[1:]
        all_masks = all_masks[:-1]
        all_costs = all_costs[1:] - all_costs[:-1]
        result = all_words, all_masks, all_costs
        if as_arrays:
            return result
        return self.result_to_lists(result)

