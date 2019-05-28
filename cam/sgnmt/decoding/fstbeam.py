"""Implementation of a beam search which uses an FST for synchronization."""

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.decoding.core import PartialHypothesis
import logging

from cam.sgnmt.utils import load_fst
import pywrapfst as fst


class FSTBeamDecoder(BeamDecoder):
    """This beam search variant synchronizes hypotheses if they share
    the same node in an FST. This is similar to the syncbeam strategy,
    but rather than using a dedicated synchronization symbol, we keep
    track of the state ID of each hypothesis in an FST. Hypotheses are
    expanded until all of them arrive at the same state id, and are
    then compared with each other to select the set of active 
    hypotheses in the next time step.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance with FST-based
        synchronization. In addition to the constructor of
        `BeamDecoder`, the following values are fetched from 
        `decoder_args`:
        
            max_word_len (int): Maximum length of a single word
            fst_path (string): Path to the FST.
        """
        super(FSTBeamDecoder, self).__init__(decoder_args)
        self.fst_path = decoder_args.fst_path
        self.max_word_len = decoder_args.max_word_len
    
    def _register_sub_score(self, score):
        """Updates best_scores and min_score. """
        if not self.maintain_best_scores:
            return
        self.sub_best_scores.append(score)
        self.sub_best_scores.sort(reverse=True)
        if len(self.sub_best_scores) >= self.beam_size:
            self.sub_best_scores = self.sub_best_scores[:self.beam_size]
            self.sub_min_score = self.sub_best_scores[-1] 

    def _get_label2node(self, root_node):
        return {arc.olabel: arc.nextstate 
                for arc in self.cur_fst.arcs(root_node)}

    def _find_start_node(self):
        for arc in self.cur_fst.arcs(self.cur_fst.start()):
            if arc.olabel == utils.GO_ID:
                return arc.nextstate
        logging.error("Start symbol %d not found in fstbeam FST!" % utils.GO_ID)

    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        self.cur_fst = load_fst(utils.get_path(self.fst_path,
                                               self.current_sen_id+1))
        init_hypo = PartialHypothesis(self.get_predictor_states())
        init_hypo.fst_node = self._find_start_node()
        return [init_hypo]
    
    def _expand_hypo(self, hypo):
        """Expand hypo until all of the beam size best hypotheses end 
        with ``sync_symb`` or EOS.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Return:
            list. List of expanded hypotheses.
        """
        # Get initial expansions
        l2n = self._get_label2node(hypo.fst_node)
        deepest_node = -1
        next_hypos = []
        next_scores = []
        for next_hypo in super(FSTBeamDecoder, self)._expand_hypo(hypo):
            node_id = l2n[next_hypo.trgt_sentence[-1]]
            deepest_node = max(node_id, deepest_node)
            next_hypo.fst_node = node_id
            next_hypos.append(next_hypo)
            next_scores.append(self._get_combined_score(next_hypo))
        # Expand until all hypos are at deepest_node.
        # This assumes that the FST is topologically sorted
        open_hypos = []
        open_hypos_scores = []
        closed_hypos = []
        for next_hypo, next_score in zip(next_hypos, next_scores):
            if next_hypo.fst_node == deepest_node:
                closed_hypos.append(next_hypo)
            else:
                open_hypos.append(next_hypo)
                open_hypos_scores.append(next_score)
        open_hypos = self._get_next_hypos(open_hypos, open_hypos_scores)
        it = 1
        while open_hypos:
            if it > self.max_word_len: # prevent infinite loops
                logging.debug("Maximum word length reached.")
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            self.sub_min_score = self.min_score
            self.sub_best_scores = []
            for h in open_hypos:
                if h.score > self.sub_min_score:
                    l2n = self._get_label2node(h.fst_node)
                    for next_hypo in super(FSTBeamDecoder, self)._expand_hypo(h):
                        next_score = self._get_combined_score(next_hypo)
                        if next_score > self.sub_min_score:
                            next_hypo.fst_node = l2n[next_hypo.trgt_sentence[-1]]
                            if next_hypo.fst_node < deepest_node: # Keep
                                next_hypos.append(next_hypo)
                                next_scores.append(next_score)
                                self._register_sub_score(next_score)
                            elif next_hypo.fst_node == deepest_node: # Add to closed
                                closed_hypos.append(next_hypo)
                            elif next_hypo.fst_node > deepest_node: # Log
                                logging.debug("FSTBeam: Deepest node exceeded")
            open_hypos = self._get_next_hypos(next_hypos, next_scores)
        logging.debug("Expand %f: %s (%d)" % (hypo.score,
                                              hypo.trgt_sentence, 
                                              hypo.fst_node))
        for h in closed_hypos:
            logging.debug("-> %f: %s (%d)" % (h.score,
                                              h.trgt_sentence, 
                                              h.fst_node))
        return closed_hypos

