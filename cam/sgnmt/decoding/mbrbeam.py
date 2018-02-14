"""This beam search uses an MBR-based criterion at each time step."""


import logging
import numpy as np

from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder


class MBRBeamDecoder(BeamDecoder):
    """The MBR-based beam decoder does not select the n most likely
    hypotheses in each timestep. Instead, it tries to find a subset
    with n elements which maximizes the expected maximum BLEU of one
    of the selected hypos. In other words, we optimize the oracle
    BLEU of the n-best list at each time step, where the n-best list
    consists of the active hypotheses in the beam.
    """
    
    def __init__(self, decoder_args):
        """Creates a new MBR beam decoder instance. We explicitly
        set early stopping to False since risk-free pruning is not 
        supported by the MBR-based beam decoder. The MBR-based
        decoder fetches the following fields from ``decoder_args``:

          min_ngram_order (int): Minimum n-gram order
          max_ngram_order (int): Maximum n-gram order
          mbrbeam_smooth_factor (float): Smoothing factor for evidence space

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        if decoder_args.early_stopping:
            logging.warn("Setting early_stopping=False (not supported by MBR)")
            decoder_args.early_stopping = False
        self.min_order = decoder_args.min_ngram_order
        self.max_order = decoder_args.max_ngram_order
        self.smooth_factor = decoder_args.mbrbeam_smooth_factor
        super(MBRBeamDecoder, self).__init__(decoder_args)

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

    def _get_next_hypos_mbr(self, hypos, scores):
        """Get hypotheses of the next time step.
        
        Args:
            hypos (list): List of hypotheses
            scores (list): hypo scores with heuristic estimates
        
        Return:
            list. List with hypotheses.
        """
        #probs = np.exp(scores - utils.log_sum(scores))
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
        for _ in xrange(min(self.beam_size, len(hypos))):
            idx = np.argmax(np.sum(exp_bleus, axis=1))
            bleu = np.sum(exp_bleus[idx])
            logging.debug("Selected (score=%f prob=%f expected_bleu=%f): %s"
                    % (scores[idx], probs[idx], bleu, 
                       hypos[idx].trgt_sentence))
            hypos[idx].bleu = bleu
            next_hypos.append(hypos[idx])
            gained_bleus = exp_bleus[idx]
            for update_idx in xrange(len(exp_bleus)):
                exp_bleus[update_idx] = np.maximum(exp_bleus[update_idx], 
                                                   gained_bleus)
        return next_hypos
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        it = 0
        self.min_score = utils.NEG_INF
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
                    next_hypos.append(next_hypo)
                    next_scores.append(next_score)
            hypos = self._get_next_hypos_mbr(next_hypos, next_scores)
        for hypo in hypos:
            hypo.score = -hypo.bleu
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()

