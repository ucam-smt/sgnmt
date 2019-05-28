"""Implementation of beam search with explicit synchronization symbol"""

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
import logging


class SyncBeamDecoder(BeamDecoder):
    """This beam search implementation is a two level approach.
    Hypotheses are not compared after each iteration, but after
    consuming an explicit synchronization symbol. This is useful
    when SGNMT runs on the character level, but it makes more sense
    to compare hypos with same lengths in terms of number of words 
    and not characters. The end-of-word symbol </w> can be used as
    synchronization symbol.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance with explicit
        synchronization symbol. In addition to the constructor of
        `BeamDecoder`, the following values are fetched from 
        `decoder_args`:
        
            sync_symb (int): Synchronization symbol. If negative,
                             consider a hypothesis as closed when it
                             ends with a terminal symbol (see 
                             syntax_min_terminal_id and
                             syntax_max_terminal_id)
            max_word_len (int): Maximum length of a single word
        """
        super(SyncBeamDecoder, self).__init__(decoder_args)
        if decoder_args.sync_symbol >= 0:
            self.sync_min = decoder_args.sync_symbol
            self.sync_max = decoder_args.sync_symbol
        else:
            self.sync_min = decoder_args.syntax_min_terminal_id
            self.sync_max = decoder_args.syntax_max_terminal_id
        self.max_word_len = decoder_args.max_word_len
    
    def _is_closed(self, hypo):
        """Returns true if hypo ends with </S> or </w>"""
        w = hypo.get_last_word()
        return w == utils.EOS_ID or (w >= self.sync_min and w <= self.sync_max)
    
    def _all_eos_or_eow(self, hypos):
        """Returns true if the all hypotheses end with </S> or </W>"""
        for hypo in hypos:
            if not self._is_closed(hypo):
                return True
        return False

    def _register_sub_score(self, score):
        """Updates best_scores and min_score. """
        if not self.maintain_best_scores:
            return
        self.sub_best_scores.append(score)
        self.sub_best_scores.sort(reverse=True)
        if len(self.sub_best_scores) >= self.beam_size:
            self.sub_best_scores = self.sub_best_scores[:self.beam_size]
            self.sub_min_score = self.sub_best_scores[-1] 
    
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
                logging.debug("Maximum word length reached.")
                return []
            it = it + 1
            next_hypos = []
            next_scores = []
            self.sub_min_score = self.min_score
            self.sub_best_scores = []
            for h in hypos:
                if self._is_closed(h):
                    next_hypos.append(h)
                    next_scores.append(self._get_combined_score(h))
                    continue 
                if h.score > self.sub_min_score:
                    for next_hypo in super(SyncBeamDecoder, self)._expand_hypo(h):
                        next_score = self._get_combined_score(next_hypo)
                        if next_score > self.sub_min_score:
                            next_hypos.append(next_hypo)
                            next_scores.append(next_score)
                            self._register_sub_score(next_score)
            hypos = self._get_next_hypos(next_hypos, next_scores)
        ret =  [h for h in hypos if self._is_closed(h)]
        logging.debug("Expand %f: %s (%d)" % (hypo.score,
                                              hypo.trgt_sentence, 
                                              len(hypo.trgt_sentence)))
        for h in ret:
            logging.debug("-> %f: %s (%d)" % (h.score,
                                              h.trgt_sentence, 
                                              len(h.trgt_sentence)))
        return ret

