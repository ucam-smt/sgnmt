"""This beam search uses an MBR-based criterion at each time step."""


import logging
import numpy as np

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
from cam.sgnmt.misc.trie import SimpleTrie


def is_sublist(needle, haystack):
    """True if needle is a sublist of haystack, False otherwise.
    Could be more efficient with Boyer-Moore-Horspool but our needles
    are usually ngrams (ie. short), so this is a O(nm) implementation.
    """
    ln = len(needle)
    lh = len(haystack)
    for pos in xrange(lh - ln + 1):
        if needle == haystack[pos:pos+ln]:
            return True
    return False


class MBRBeamDecoder(BeamDecoder):
    """The MBR-based beam decoder does not select the n most likely
    hypotheses in each timestep. Instead, it tries to find the translation
    with the best expected BLEU. Two strategies control the behavior of 
    mbrbeam: the `evidence_strategy` and the `selection_strategy`.
    Available evidence strategies:
      'renorm': Only makes use of the n-best expansions of the hypos in the
                current beam. It renormalizes the scores, and count ngrams in
                the n^2 hypos.
      'maxent': Applies the MaxEnt rule to the evidence space. It makes use of
                all partial hypos seen so far and updates its belief about the
                probability of an ngram based on that. Following MaxEnt we 
                assume that translations outside the explored space are 
                uniformly distributed.
    Available selection strategies:
      'bleu': Select the n hypotheses with the best expected BLEU
      'oracle_bleu': Select'a subset with n elements which maximizes the 
                     expected maximum BLEU of one of the selected hypos. In 
                     other words, we optimize the oracle BLEU of the n-best 
                     list at each time step, where the n-best list consists 
                     of the active hypotheses in the beam.
    """
    
    def __init__(self, decoder_args):
        """Creates a new MBR beam decoder instance. We explicitly
        set early stopping to False since risk-free pruning is not 
        supported by the MBR-based beam decoder. The MBR-based
        decoder fetches the following fields from ``decoder_args``:

          min_ngram_order (int): Minimum n-gram order
          max_ngram_order (int): Maximum n-gram order
          mbrbeam_smooth_factor (float): Smoothing factor for evidence space
          mbrbeam_evidence_strategy (String): Evidence strategy
          mbrbeam_selection_strategy (String): Selection strategy

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        self.min_order = decoder_args.min_ngram_order
        self.max_order = decoder_args.max_ngram_order
        self.smooth_factor = decoder_args.mbrbeam_smooth_factor
        self.selection_strategy = decoder_args.mbrbeam_selection_strategy
        if not self.selection_strategy in ['bleu', 'oracle_bleu']:
            raise AttributeError("Unknown selection strategy '%s'"
                                 % self.selection_strategy)
        if decoder_args.mbrbeam_evidence_strategy == 'renorm':
            self._get_next_hypos_mbr = self._get_next_hypos_renorm
        elif decoder_args.mbrbeam_evidence_strategy == 'maxent':
            self._get_next_hypos_mbr = self._get_next_hypos_maxent
            if decoder_args.sub_beam <= 0:
                self.sub_beam_size = 0 # We need all expansions
        else:
            raise AttributeError("Unknown evidence strategy '%s'"
                                 % decoder_args.mbrbeam_evidence_strategy)
        super(MBRBeamDecoder, self).__init__(decoder_args)
        self.maintain_best_scores = False # Does not work with MBR

    def _compute_bleu(self, hyp_ngrams, ref_ngrams, hyp_length, ref_length):
        """Not the exact BLEU score, we do filter out multiple matches for
        the same ngram.
        """
        precisions = []
        for hyp_ngrams_order, ref_ngrams_order in zip(hyp_ngrams, ref_ngrams):
            if not hyp_ngrams_order:
                continue
            cnt = 0
            for hyp_ngram in hyp_ngrams_order:
                if hyp_ngram in ref_ngrams_order:
                    cnt += 1
            precisions.append(float(cnt) / float(len(hyp_ngrams_order)))
        weight = 1.0 / float(len(precisions))
        p = 1.0
        for precision in precisions:
            p *= precision ** weight
        bp = 1.0
        if hyp_length < ref_length:
            bp = np.exp(1.0 - float(ref_length) / float(hyp_length))
        return bp * p

    def _get_next_hypos_maxent(self, hypos, scores):
        """Get hypotheses of the next time step.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses.
        """
        # Update self.maxent_ngram_mass
        for hypo_score, hypo in zip(scores, hypos):
            s = hypo.trgt_sentence
            h = s[:-1]
            l = len(s)
            if l <= self.maxent_processed_length:
                continue
            # TODO: Could be more efficient by checking is_sublist for
            # all orders in one pass
            for order in xrange(min(len(s), self.max_order), 
                                self.min_order-1,
                                -1):
                ngram = s[-order:]
                # Do not use this ngram if it occurs before
                if is_sublist(ngram, h):
                    break # All lower order ngrams are too
                prev_mass = self.maxent_ngram_mass.get(ngram)
                if prev_mass is None:
                    updated_mass = hypo_score
                else:
                    updated_mass = max(prev_mass, hypo_score, 
                            np.log(np.exp(prev_mass)+np.exp(hypo_score)))
                self.maxent_ngram_mass.add(ngram, updated_mass)
        self.maxent_processed_length += 1
        exp_counts = []
        for hypo in hypos:
            s = hypo.trgt_sentence
            l = len(s)
            cnt = 0.0
            for order in xrange(self.min_order, self.max_order+1):
                for start in xrange(l-order+1):
                    logprob = self.maxent_ngram_mass.get(s[start:start+order])
                    # MaxEnt means that we estimate the probability of the 
                    # ngram as p + (1-p) * 0.5 ie.
                    if logprob:
                        cnt += 1.0 + np.exp(logprob)
                    else:
                        cnt += 1.0
            exp_counts.append(cnt * 0.5)
        next_hypos = []
        for idx in utils.argmax_n(exp_counts, self.beam_size):
            hypos[idx].bleu = exp_counts[idx]
            next_hypos.append(hypos[idx])
            logging.debug("Selected (score=%f expected_counts=%f): %s"
                % (scores[idx], hypos[idx].bleu, hypos[idx].trgt_sentence))
        return next_hypos

    def _get_next_hypos_renorm(self, hypos, scores):
        """Get hypotheses of the next time step.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses.
        """
        probs = (1.0 - self.smooth_factor) * np.exp(
            scores - utils.log_sum(scores)) \
            + self.smooth_factor / float(len(scores))
        lengths = [len(hypo.trgt_sentence) for hypo in hypos]
        logging.debug("%d candidates min_length=%d max_length=%d" % 
            (len(lengths), min(lengths), max(lengths)))
        ngrams = []
        for hypo in hypos:
            ngram_list = []
            for order in xrange(self.min_order, self.max_order+1):
                ngram_list.append(set([
                    " ".join(map(str, hypo.trgt_sentence[start:start+order]))
                    for start in xrange(len(hypo.trgt_sentence))]))
            ngrams.append(ngram_list)
        exp_bleus = []
        for hyp_ngrams, hyp_length in zip(ngrams, lengths):
            precisions = np.array([self._compute_bleu(
                    hyp_ngrams, ref_ngrams, hyp_length, ref_length)
                for ref_ngrams, ref_length in zip(ngrams, lengths)])
            exp_bleus.append(precisions * probs)
        next_hypos = []
        if self.selection_strategy == 'oracle_bleu': 
            for _ in xrange(min(self.beam_size, len(hypos))):
                idx = np.argmax(np.sum(exp_bleus, axis=1))
                bleu = np.sum(exp_bleus[idx])
                logging.debug("Selected (score=%f expected_bleu=%f): %s"
                        % (scores[idx], bleu, hypos[idx].trgt_sentence))
                hypos[idx].bleu = -bleu
                next_hypos.append(hypos[idx])
                gained_bleus = exp_bleus[idx]
                for update_idx in xrange(len(exp_bleus)):
                    exp_bleus[update_idx] = np.maximum(exp_bleus[update_idx], 
                                                       gained_bleus)
        else: # selection strategy 'bleu'
            total_exp_bleus = np.sum(exp_bleus, axis=1)
            for idx in utils.argmax_n(total_exp_bleus, self.beam_size):
                hypos[idx].bleu = total_exp_bleus[idx]
                next_hypos.append(hypos[idx])
                logging.debug("Selected (score=%f expected_bleu=%f): %s"
                    % (scores[idx], hypos[idx].bleu, hypos[idx].trgt_sentence))
        return next_hypos
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        self.min_score = utils.NEG_INF
        self.maxent_ngram_mass = SimpleTrie()
        self.maxent_processed_length = 0
        while self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self._get_combined_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo):
                    next_score = self._get_combined_score(next_hypo)
                    if next_score > utils.NEG_INF:
                        next_hypos.append(next_hypo)
                        next_scores.append(next_score)
            hypos = self._get_next_hypos_mbr(next_hypos, next_scores)
        for hypo in hypos:
            hypo.score = hypo.bleu
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()

