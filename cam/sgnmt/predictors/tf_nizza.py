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
            self.mon_sess = self.create_session()

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

    def create_session(self):
        """Creates a MonitoredSession for this predictor."""
        checkpoint_path = saver.latest_checkpoint(self._checkpoint_dir)
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
        lex_logits = self.mon_sess.run(self.precomputed,
            {self._inputs_var: src_sentence})
        self.trg_vocab_size = lex_logits.shape[2]
        src_len = len(src_sentence)
        self.coverage = "0" * src_len
        self.short_lists = []
        self.short_list_scores = []
        for src_pos in xrange(src_len):
            shortlist = self._create_short_list(lex_logits[0, src_pos, :])
            self.short_lists.append(shortlist)
            if not self.alpha_is_zero:
                scores = np.zeros(self.trg_vocab_size)
                for w in shortlist:
                    scores[w] = self.alpha
                self.short_list_scores.append(scores)
        logging.debug("Short list sizes: %s" % ", ".join([
                str(len(l)) for l in self.short_lists]))
    
    def consume(self, word):
        """Update coverage."""
        new_coverage = []
        for src_pos, is_covered in enumerate(self.coverage):
            if is_covered == "0" and word in self.short_lists[src_pos]:
                is_covered = "1"
            new_coverage.append(is_covered)
        self.coverage = "".join(new_coverage)

    def _create_short_list(self, logits):
        """Creates a set of tokens which are likely translations."""
        words = set()
        for strat in self.shortlist_strategies:
            if strat[:3] == "top":
                n = int(strat[3:])
                words.update(utils.argmax_n(logits, n))
            elif strat[:4] == "prob":
                p = float(strat[4:])
                unnorm_probs = np.exp(logits)
                threshold = np.sum(unnorm_probs) * p
                acc = 0.0
                for word in np.argsort(logits)[::-1]:
                    acc += unnorm_probs[word]
                    words.add(word)
                    if acc >= threshold:
                        break
            else:
                raise AttributeError("Unknown shortlist strategy '%s'" % strat)
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

