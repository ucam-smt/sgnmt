"""This module integrates Nizza alignment models.

https://github.com/fstahlberg/nizza
"""

import logging
import os
import numpy as np
from scipy.misc import logsumexp

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor

try:
    import tensorflow as tf
    from tensorflow.python.training import saver
    from tensorflow.python.training import training
    # Requires nizza
    from nizza import registry
    from nizza.utils import common_utils
except ImportError:
    pass # Deal with it in decode.py



class BaseNizzaPredictor(Predictor):
    """Common functionality for Nizza based predictors. This includes 
    loading checkpoints, creating sessions, and creating computation 
    graphs.
    """

    def __init__(self, src_vocab_size, trg_vocab_size, model_name, 
                 hparams_set_name, checkpoint_dir, single_cpu_thread,
                 nizza_unk_id=None):
        """Initializes a nizza predictor.

        Args:
            src_vocab_size (int): Source vocabulary size (called inputs_vocab_size
                in nizza)
            trg_vocab_size (int): Target vocabulary size (called targets_vocab_size
                in nizza)
            model_name (string): Name of the nizza model
            hparams_set_name (string): Name of the nizza hyper-parameter set
            checkpoint_dir (string): Path to the Nizza checkpoint directory. The 
                                     predictor will load the top most checkpoint in 
                                     the `checkpoints` file.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            nizza_unk_id (int): If set, use this as UNK id. Otherwise, the
                nizza is assumed to have no UNKs

        Raises:
            IOError if checkpoint file not found.
        """
        super(BaseNizzaPredictor, self).__init__()
        if not os.path.isfile("%s/checkpoint" % checkpoint_dir):
            logging.fatal("Checkpoint file %s/checkpoint not found!" 
                          % checkpoint_dir)
            raise IOError
        self._single_cpu_thread = single_cpu_thread
        self._checkpoint_dir = checkpoint_dir
        self._nizza_unk_id = nizza_unk_id
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = registry.get_registered_hparams_set(hparams_set_name)
            hparams.add_hparam("inputs_vocab_size", src_vocab_size)
            hparams.add_hparam("targets_vocab_size", trg_vocab_size)
            run_config = tf.contrib.learn.RunConfig()
            run_config = run_config.replace(model_dir=checkpoint_dir)
            model = registry.get_registered_model(model_name, hparams, run_config)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                               name="sgnmt_targets")
            features = {"inputs": tf.expand_dims(self._inputs_var, 0), 
                        "targets": tf.expand_dims(self._targets_var, 0)}
            mode = tf.estimator.ModeKeys.PREDICT
            self.precomputed = model.precompute(features, mode, hparams)
            self.log_probs = tf.squeeze(
                model.predict_next_word(features, hparams, self.precomputed), 0)
            self.mon_sess = self.create_session(self._checkpoint_dir)

    def _session_config(self):
        """Creates the session config with t2t default parameters."""
        graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
            opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
        if self._single_cpu_thread:
            config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                graph_options=graph_options,
                log_device_placement=False)
        else:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=0.95)
            config = tf.ConfigProto(
                allow_soft_placement=True,
                graph_options=graph_options,
                gpu_options=gpu_options,
                log_device_placement=False)
        return config

    def create_session(self, checkpoint_dir):
        """Creates a MonitoredSession for this predictor."""
        checkpoint_path = saver.latest_checkpoint(checkpoint_dir)
        return training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                config=self._session_config()))

    def get_unk_probability(self, posterior):
        """Fetch posterior[t2t_unk_id] or return NEG_INF if None."""
        if self._nizza_unk_id is None:
            return utils.NEG_INF
        return posterior[self._nizza_unk_id]


class NizzaPredictor(BaseNizzaPredictor):
    """This predictor uses Nizza alignment models to derive a posterior over
    the target vocabulary for the next position. It mainly relies on the
    predict_next_word() implementation of Nizza models.
    """

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self.log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: self.consumed + [common_utils.PAD_ID]})
        log_probs[common_utils.PAD_ID] = utils.NEG_INF  # Mask padding
        return log_probs
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.consumed = []
        self.src_sentence = src_sentence + [utils.EOS_ID]
    
    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)
    
    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed = state

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1 == state2


class LexNizzaPredictor(BaseNizzaPredictor):
    """This predictor is only compatible to Model1-like Nizza models
    which return lexical translation probabilities in precompute(). The
    predictor keeps a list of the same length as the source sentence
    and initializes it with zeros. At each timestep it updates this list
    by the lexical scores Model1 assigned to the last consumed token.
    The predictor score aims to bring up all entries in the list, and 
    thus serves as a coverage mechanism over the source sentence.
    """

    def __init__(self, src_vocab_size, trg_vocab_size, model_name, 
                 hparams_set_name, checkpoint_dir, single_cpu_thread,
                 alpha, beta, shortlist_strategies,
                 trg2src_model_name="", trg2src_hparams_set_name="",
                 trg2src_checkpoint_dir="",
                 max_shortlist_length=0,
                 min_id=0,
                 nizza_unk_id=None):
        """Initializes a nizza predictor.

        Args:
            src_vocab_size (int): Source vocabulary size (called inputs_vocab_size
                in nizza)
            trg_vocab_size (int): Target vocabulary size (called targets_vocab_size
                in nizza)
            model_name (string): Name of the nizza model
            hparams_set_name (string): Name of the nizza hyper-parameter set
            checkpoint_dir (string): Path to the Nizza checkpoint directory. The 
                                     predictor will load the top most checkpoint in 
                                     the `checkpoints` file.
            single_cpu_thread (bool): If true, prevent tensorflow from
                                      doing multithreading.
            alpha (float): Score for each matching word
            beta (float): Penalty for each uncovered word at the end
            shortlist_strategies (string): Comma-separated list of shortlist
                strategies.
            trg2src_model_name (string): Name of the target2source nizza model
            trg2src_hparams_set_name (string): Name of the nizza hyper-parameter set
                                     for the target2source model
            trg2src_checkpoint_dir (string): Path to the Nizza checkpoint directory
                                     for the target2source model. The 
                                     predictor will load the top most checkpoint in 
                                     the `checkpoints` file.
            max_shortlist_length (int): If a shortlist exceeds this limit,
                initialize the initial coverage with 1 at this position. If
                zero, do not apply any limit
            min_id (int): Do not use IDs below this threshold (filters out most
                frequent words).
            nizza_unk_id (int): If set, use this as UNK id. Otherwise, the
                nizza is assumed to have no UNKs

        Raises:
            IOError if checkpoint file not found.
        """
        super(LexNizzaPredictor, self).__init__(
                src_vocab_size, trg_vocab_size, model_name, hparams_set_name, 
                checkpoint_dir, single_cpu_thread, nizza_unk_id=nizza_unk_id)
        self.alpha = alpha
        self.alpha_is_zero = alpha == 0.0
        self.beta = beta
        self.shortlist_strategies = utils.split_comma(shortlist_strategies)
        self.max_shortlist_length = max_shortlist_length
        self.min_id = min_id
        if trg2src_checkpoint_dir:
            self.use_trg2src = True
            predictor_graph = tf.Graph()
            with predictor_graph.as_default() as g:
                hparams = registry.get_registered_hparams_set(trg2src_hparams_set_name)
                hparams.add_hparam("inputs_vocab_size", trg_vocab_size)
                hparams.add_hparam("targets_vocab_size", src_vocab_size)
                run_config = tf.contrib.learn.RunConfig()
                run_config = run_config.replace(model_dir=trg2src_checkpoint_dir)
                model = registry.get_registered_model(trg2src_model_name, hparams, run_config)
                features = {"inputs": tf.expand_dims(tf.range(trg_vocab_size), 0)}
                mode = tf.estimator.ModeKeys.PREDICT
                trg2src_lex_logits = model.precompute(features, mode, hparams)
                # Precompute trg2src partitions
                partitions = tf.reduce_logsumexp(trg2src_lex_logits, axis=-1)
                self._trg2src_src_words_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                                  name="sgnmt_trg2src_src_words")
                # trg2src_lex_logits has shape [1, trg_vocab_size, src_vocab_size]
                self.trg2src_logits = tf.gather(tf.transpose(trg2src_lex_logits[0, :, :]), self._trg2src_src_words_var)
                # trg2src_logits has shape [len(src_words), trg_vocab_size]
                self.trg2src_mon_sess = self.create_session(trg2src_checkpoint_dir)
                logging.debug("Precomputing lexnizza trg2src partitions...")
                self.trg2src_partitions = self.trg2src_mon_sess.run(partitions)
        else:
            self.use_trg2src = False
            logging.warn("No target-to-source model specified for lexnizza.")

    def get_unk_probability(self, posterior):
        if self.alpha_is_zero:
            return 0.0 
        if self._nizza_unk_id is None:
            return utils.NEG_INF
        return posterior[self._nizza_unk_id]

    def predict_next(self):
        """Predict record scores."""
        if self.alpha_is_zero:
            n_uncovered = self.coverage.count("0")
            return {utils.EOS_ID: -float(n_uncovered) * self.beta}
        uncovered_scores = [self.short_list_scores[src_pos]
             for src_pos, is_covered in enumerate(self.coverage)
             if is_covered == "0"]
        if not uncovered_scores:
            return np.zeros(self.trg_vocab_size)
        scores = np.max(uncovered_scores, axis=0)
        scores[utils.EOS_ID] = -len(uncovered_scores) * self.beta
        return scores
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.filt_src_sentence = [w for w in src_sentence if w >= self.min_id]
        scores = self.mon_sess.run(self.precomputed,
            {self._inputs_var: self.filt_src_sentence})
        scores = scores[0, :, :]
        # scores has shape [src_sentence_len, trg_vocab_size]
        self.trg_vocab_size = scores.shape[1]
        if self.use_trg2src:
            trg2src_logits = self.trg2src_mon_sess.run(self.trg2src_logits,
                {self._trg2src_src_words_var: self.filt_src_sentence})
            src2trg_logits = scores
            src2trg_partitions = logsumexp(src2trg_logits, axis=1, keepdims=True)
            trg2src_logprobs = trg2src_logits - self.trg2src_partitions
            src2trg_logprobs = src2trg_logits - src2trg_partitions
            scores = src2trg_logprobs + trg2src_logprobs
        src_len = len(self.filt_src_sentence)
        is_covered = []
        self.short_lists = []
        self.short_list_scores = []
        for src_pos in xrange(src_len):
            shortlist = self._create_short_list(scores[src_pos, :])
            if (self.max_shortlist_length > 0 
                      and len(shortlist) > self.max_shortlist_length):
                is_covered.append("1")
                shortlist = set([])
            else:
                is_covered.append("0")
            self.short_lists.append(shortlist)
            if not self.alpha_is_zero:
                alpha_scores = np.zeros(self.trg_vocab_size)
                for w in shortlist:
                    alpha_scores[w] = self.alpha
                self.short_list_scores.append(alpha_scores)
        self.coverage = "".join(is_covered)
        logging.debug("Short list sizes: %s" % ", ".join([
                str(len(l)) for l in self.short_lists]))
        logging.debug("Initial coverage: %s" % self.coverage)
        #print("SHORT LISTS")
        #for w, l in zip(self.filt_src_sentence, self.short_lists):
        #    print("\n\n%d" % w)
        #    if len(l) < 40:
        #        print(" ".join(map(str, l)))
              
    def consume(self, word):
        """Update coverage."""
        new_coverage = []
        for src_pos, is_covered in enumerate(self.coverage):
            if is_covered == "0" and word in self.short_lists[src_pos]:
                is_covered = "1"
            new_coverage.append(is_covered)
        self.coverage = "".join(new_coverage)
        #logging.debug("Partial: %s" % " ".join([str(self.filt_src_sentence[idx])  for idx, c in enumerate(self.coverage) if c == "1"]))
        #logging.debug(self.coverage)

    def _create_short_list(self, logits):
        """Creates a set of tokens which are likely translations."""
        words = set()
        filt_logits = logits[self.min_id:]
        for strat in self.shortlist_strategies:
            if strat[:3] == "top":
                n = int(strat[3:])
                words.update(utils.argmax_n(filt_logits, n))
            elif strat[:4] == "prob":
                p = float(strat[4:])
                unnorm_probs = np.exp(filt_logits)
                threshold = np.sum(unnorm_probs) * p
                acc = 0.0
                for word in np.argsort(filt_logits)[::-1]:
                    acc += unnorm_probs[word]
                    words.add(word)
                    if acc >= threshold:
                        break
            else:
                raise AttributeError("Unknown shortlist strategy '%s'" % strat)
        if self.min_id:
            words = set(w+self.min_id for w in words)
        try:
            words.remove(utils.EOS_ID)
        except KeyError:
            pass
        return words

    def estimate_future_cost(self, hypo):
        """We use the number of uncovered words times beta as heuristic
        estimate.
        """
        if hypo.trgt_sentence[:-1] == [utils.EOS_ID]:
            return 0.0
        n_uncovered = 0
        for short_list in self.short_lists:
            if not any(w in hypo.trgt_sentence for w in short_list):
                n_uncovered += 1
        return -float(n_uncovered) * self.beta * 0.1
    
    def get_state(self):
        """The predictor state is the coverage vector."""
        return self.coverage
    
    def set_state(self, state):
        """The predictor state is the coverage vector."""
        self.coverage = state

