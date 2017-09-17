"""Implementation of beam search with explicit synchronization symbol"""


from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder


class SyncBeamDecoder(BeamDecoder):
    """This beam search implementation is a two level approach.
    Hypotheses are not compared after each iteration, but after
    consuming an explicit synchronization symbol. This is useful
    when SGNMT runs on the character level, but it makes more sense
    to compare hypos with same lengths in terms of number of words 
    and not characters. The end-of-word symbol </w> can be used as
    synchronization symbol.
    """
    
    def __init__(self,
                 decoder_args,
                 hypo_recombination,
                 beam_size,
                 pure_heuristic_scores = False, 
                 diversity_factor = -1.0,
                 early_stopping = True,
                 sync_symb = -1,
                 max_word_len = 25):
        """Creates a new beam decoder instance with explicit
        synchronization symbol.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination 
            beam_size (int): Absolute beam size. A beam of 12 means
                             that we keep track of 12 active hypothesis
            pure_heuristic_scores (bool): Hypotheses to keep in the beam
                                          are normally selected 
                                          according the sum of partial
                                          hypo score and future cost
                                          estimates. If set to true, 
                                          partial hypo scores are 
                                          ignored.
            diversity_factor (float): If this is set to a positive 
                                      value we add diversity promoting
                                      penalization terms to the partial
                                      hypothesis scores following Li
                                      and Jurafsky, 2016
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable
            sync_symb (int): Synchronization symbol. If negative, fetch
                             '</w>' from ``utils.trg_cmap`` 
            max_word_len (int): Maximum length of a single word
        """
        super(SyncBeamDecoder, self).__init__(decoder_args,
                                              hypo_recombination,
                                              beam_size,
                                              pure_heuristic_scores, 
                                              diversity_factor,
                                              early_stopping)
        self.sync_symb = sync_symb
        self.max_word_len = max_word_len
    
    def _is_closed(self, hypo):
        """Returns true if hypo ends with </S> or </W>"""
        return hypo.get_last_word() in [utils.EOS_ID, self.sync_symb]
    
    def _all_eos_or_eow(self, hypos):
        """Returns true if the all hypotheses end with </S> or </W>"""
        for hypo in hypos:
            if not self._is_closed(hypo):
                return True
        return False
    
    def _expand_hypo(self, hypo):
        """Expand hypo until all of the beam size best hypotheses end 
        with ``sync_symb`` or EOS.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Return:
            list. List of expanded hypotheses.
        """
        # Get initial expansions
        next_hypos = []
        next_scores = []
        for next_hypo in super(SyncBeamDecoder, self)._expand_hypo(hypo):
            next_hypos.append(next_hypo)
            next_scores.append(self._get_combined_score(next_hypo))
        hypos = self._get_next_hypos(next_hypos, next_scores)
        # Expand until all hypos are closed
        it = 1
        while self._all_eos_or_eow(hypos):
            if it > self.max_word_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if self._is_closed(hypo):
                    next_hypos.append(hypo)
                    next_scores.append(self._get_combined_score(hypo))
                    continue 
                for next_hypo in super(SyncBeamDecoder, self)._expand_hypo(hypo):
                    next_hypos.append(next_hypo)
                    next_scores.append(self._get_combined_score(next_hypo))
            hypos = self._get_next_hypos(next_hypos, next_scores)
        return [h for h in hypos if self._is_closed(h)]

