"""This is the interface to the tensor2tensor library.

https://github.com/tensorflow/tensor2tensor

Alternatively, you may use the following fork which has been tested in
combination with SGNMT:

https://github.com/fstahlberg/tensor2tensor

The t2t predictor can read any model trained with tensor2tensor which
includes the transformer model, convolutional models, and RNN-based
sequence models.
"""

import logging
import os

from cam.sgnmt import utils, tf_utils
from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt.misc.trie import SimpleTrie

POP = "##POP##"
"""Textual representation of the POP symbol."""

try:
    # Requires tensor2tensor
    from tensor2tensor import models  # pylint: disable=unused-import
    from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
    from tensor2tensor.utils import usr_dir
    from tensor2tensor.utils import registry
    from tensor2tensor.utils import devices
    from tensor2tensor.utils import trainer_lib
    from tensor2tensor.data_generators.text_encoder import TextEncoder
    from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
    from tensor2tensor.data_generators import text_encoder
    import tensorflow as tf
    import numpy as np

    class DummyTextEncoder(TextEncoder):
        """Dummy TextEncoder implementation. The TextEncoder 
        implementation in tensor2tensor reads the vocabulary file in
        the constructor, which is not available inside SGNMT. This
        class can be used to replace the standard TextEncoder 
        implementation with a fixed vocabulary size. Note that this
        encoder cannot be used to translate between raw text and
        integer sequences.
        """

        def __init__(self, vocab_size, pop_id=None):
            super(DummyTextEncoder, self).__init__(num_reserved_ids=None)
            self._vocab_size = vocab_size

        def encode(self, s):
            raise NotImplementedError("Dummy encoder cannot be used to encode.")

        def decode(self, ids):
            raise NotImplementedError("Dummy encoder cannot be used to decode.")

        @property
        def vocab_size(self):
            return self._vocab_size

    # Define flags from the t2t binaries
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("schedule", "train_and_evaluate",
                        "Method of tf.contrib.learn.Experiment to run.")
except ImportError:
    pass # Deal with it in decode.py


T2T_INITIALIZED = False
"""Set to true by _initialize_t2t() after first constructor call."""


def _initialize_t2t(t2t_usr_dir):
    global T2T_INITIALIZED
    if not T2T_INITIALIZED:
        logging.info("Setting up tensor2tensor library...")
        tf.logging.set_verbosity(tf.logging.INFO)
        usr_dir.import_usr_dir(t2t_usr_dir)
        T2T_INITIALIZED = True


def log_prob_from_logits(logits):
    """Softmax function."""
    return logits - tf.reduce_logsumexp(logits, keepdims=True, axis=-1)


def expand_input_dims_for_t2t(t, batched=False):
    """Expands a plain input tensor for using it in a T2T graph.

    Args:
        t: Tensor
        batched: Whether to expand on the left side

    Returns:
      Tensor `t` expanded by 1 dimension on the left and two dimensions
      on the right.
    """
    if not batched:
        t = tf.expand_dims(t, 0) # Because of batch_size
    t = tf.expand_dims(t, -1) # Because of modality
    t = tf.expand_dims(t, -1) # Because of random reason X
    return t


def gather_2d(params, indices):
  """This is a batched version of tf.gather(), ie. it applies tf.gather() to
  each batch separately.

  Example:
    params = [[10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24]]
    indices = [[0, 0, 1, 1, 1, 2],
               [1, 3, 0, 0, 2, 2]]
    result = [[10, 10, 11, 11, 11, 12],
              [21, 23, 20, 20, 22, 22]]

  Args:
    params: A [batch_size, n, ...] tensor with data
    indices: A [batch_size, num_indices] int32 tensor with indices into params.
             Entries must be smaller than n

  Returns:
    The result of tf.gather() on each entry of the batch.
  """
  # TODO(fstahlberg): Curse TF for making this so awkward.
  batch_size = tf.shape(params)[0]
  num_indices = tf.shape(indices)[1]
  batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1),
                          [1, num_indices])
  # batch_indices is [[0,0,0,0,...],[1,1,1,1,...],...]
  gather_nd_indices = tf.stack([batch_indices, indices], axis=2)
  return tf.gather_nd(params, gather_nd_indices)


class _BaseTensor2TensorPredictor(Predictor):
    """Base class for tensor2tensor based predictors."""

    def __init__(self,
                 t2t_usr_dir,
                 checkpoint_dir,
                 src_vocab_size,
                 trg_vocab_size,
                 t2t_unk_id,
                 n_cpu_threads,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Common initialization for tensor2tensor predictors.

        Args:
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            n_cpu_threads (int): Number of TensorFlow CPU threads.
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(_BaseTensor2TensorPredictor, self).__init__()
        self._n_cpu_threads = n_cpu_threads
        self._t2t_unk_id = utils.UNK_ID if t2t_unk_id < 0 else t2t_unk_id
        self._checkpoint_dir = checkpoint_dir
        try:
            self.pop_id = int(pop_id) 
        except ValueError:
            logging.warn("t2t predictor only supports single POP IDs. "
                         "Reset to -1")
            self.pop_id = -1
        self.max_terminal_id = max_terminal_id 
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        _initialize_t2t(t2t_usr_dir)

    def _add_problem_hparams(self, hparams, problem_name):
        """Add problem hparams for the problems. 

        This method corresponds to create_hparams() in tensor2tensor's
        trainer_lib module, but replaces the feature encoders with
        DummyFeatureEncoder's.

        Args:
            hparams (Hparams): Model hyper parameters.
            problem_name (string): T2T problem name.
        
        Returns:
            hparams object.

        Raises:
            LookupError if the problem name is not in the registry or
            uses the old style problem_hparams.
        """
        if self.pop_id >= 0:
            try:
                hparams.add_hparam("pop_id", self.pop_id)
            except:
                if hparams.pop_id != self.pop_id:
                    logging.warn("T2T pop_id does not match (%d!=%d)"
                                 % (hparams.pop_id, self.pop_id))
        try:
            hparams.add_hparam("max_terminal_id", self.max_terminal_id)
        except:
            if hparams.max_terminal_id != self.max_terminal_id:
                logging.warn("T2T max_terminal_id does not match (%d!=%d)"
                             % (hparams.max_terminal_id, self.max_terminal_id))
        try:
            hparams.add_hparam("closing_bracket_id", self.pop_id)
        except:
            if hparams.closing_bracket_id != self.pop_id:
                logging.warn("T2T closing_bracket_id does not match (%d!=%d)"
                             % (hparams.closing_bracket_id, self.pop_id))
        problem = registry.problem(problem_name)
        problem._encoders = {
            "inputs": DummyTextEncoder(vocab_size=self.src_vocab_size),
            "targets": DummyTextEncoder(vocab_size=self.trg_vocab_size)
        }
        p_hparams = problem.get_hparams(hparams)
        hparams.problem = problem
        hparams.problem_hparams = p_hparams
        return hparams

    def create_session(self):
        return tf_utils.create_session(self._checkpoint_dir,
                                       self._n_cpu_threads)

    def get_unk_probability(self, posterior):
        """Fetch posterior[t2t_unk_id]"""
        return utils.common_get(posterior, self._t2t_unk_id, utils.NEG_INF)


class T2TPredictor(_BaseTensor2TensorPredictor):
    """This predictor implements scoring with Tensor2Tensor models. We
    follow the decoder implementation in T2T and do not reuse network
    states in decoding. We rather compute the full forward pass along
    the current history. Therefore, the decoder state is simply the
    the full history of consumed words.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 n_cpu_threads=-1,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Creates a new T2T predictor. The constructor prepares the
        TensorFlow session for predict_next() calls. This includes:
        - Load hyper parameters from the given set (hparams)
        - Update registry, load T2T model
        - Create TF placeholders for source sequence and target prefix
        - Create computation graph for computing log probs.
        - Create a MonitoredSession object, which also handles 
          restoring checkpoints.
        
        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            n_cpu_threads (int): Number of TensorFlow CPU threads.
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(T2TPredictor, self).__init__(t2t_usr_dir, 
                                           checkpoint_dir, 
                                           src_vocab_size,
                                           trg_vocab_size,
                                           t2t_unk_id, 
                                           n_cpu_threads,
                                           max_terminal_id,
                                           pop_id)
        if not model_name or not problem_name or not hparams_set_name:
            logging.fatal(
                "Please specify t2t_model, t2t_problem, and t2t_hparams_set!")
            raise AttributeError
        self.consumed = []
        self.src_sentence = []
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = trainer_lib.create_hparams(hparams_set_name)
            self._add_problem_hparams(hparams, problem_name)
            translate_model = registry.model(model_name)(
                hparams, tf.estimator.ModeKeys.PREDICT)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                               name="sgnmt_targets")
            features = {"inputs": expand_input_dims_for_t2t(self._inputs_var), 
                        "targets": expand_input_dims_for_t2t(self._targets_var)}
            translate_model.prepare_features_for_infer(features)
            translate_model._fill_problem_hparams_features(features)
            logits, _ = translate_model(features)
            logits = tf.squeeze(logits, [0, 1, 2, 3])
            self._log_probs = log_prob_from_logits(logits)
            self.mon_sess = self.create_session()

                
    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: utils.oov_to_unk(
                 self.consumed + [text_encoder.PAD_ID],
                 self.trg_vocab_size,
                 self._t2t_unk_id)})
        log_probs[text_encoder.PAD_ID] = utils.NEG_INF
        return log_probs
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        self.consumed = []
        self.src_sentence = utils.oov_to_unk(
            src_sentence + [text_encoder.EOS_ID], 
            self.src_vocab_size, self._t2t_unk_id)
   
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


class EditT2TPredictor(_BaseTensor2TensorPredictor):
    """This predictor can be used for T2T models conditioning on the
    full target sentence. The predictor state is a full target sentence.
    The state can be changed by insertions, substitutions, and deletions
    of single tokens, whereas each operation is encoded as SGNMT token
    in the following way:

      1xxxyyyyy: Insert the token yyyyy at position xxx.
      2xxxyyyyy: Replace the xxx-th word with the token yyyyy.
      3xxx00000: Delete the xxx-th token.
    """

    INS_OFFSET = 100000000
    SUB_OFFSET = 200000000
    DEL_OFFSET = 300000000

    POS_FACTOR = 100000
    MAX_SEQ_LEN = 999

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 trg_test_file,
                 beam_size,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 n_cpu_threads=-1,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Creates a new edit T2T predictor. This constructor is
        similar to the constructor of T2TPredictor but creates a
        different computation graph which retrieves scores at each
        target position, not only the last one.
        
        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            trg_test_file (string): Path to a plain text file with
                initial target sentences. Can be empty.
            beam_size (int): Determines how many substitutions and
                insertions are considered at each position.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            n_cpu_threads (int): Number of TensorFlow CPU threads.
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(EditT2TPredictor, self).__init__(t2t_usr_dir, 
                                               checkpoint_dir, 
                                               src_vocab_size,
                                               trg_vocab_size,
                                               t2t_unk_id, 
                                               n_cpu_threads,
                                               max_terminal_id,
                                               pop_id)
        if not model_name or not problem_name or not hparams_set_name:
            logging.fatal(
                "Please specify t2t_model, t2t_problem, and t2t_hparams_set!")
            raise AttributeError
        if trg_vocab_size >= EditT2TPredictor.POS_FACTOR:
            logging.fatal("Target vocabulary size (%d) must be less than %d!"
                          % (trg_vocab_size, EditT2TPredictor.POS_FACTOR))
            raise AttributeError
        self.beam_size = max(1, beam_size // 10) + 1
        self.batch_size = 2048 # TODO(fstahlberg): Move to config
        self.initial_trg_sentences = None
        if trg_test_file: 
            self.initial_trg_sentences = []
            with open(trg_test_file) as f:
                for line in f:
                    self.initial_trg_sentences.append(utils.oov_to_unk(
                       [int(w) for w in line.strip().split()] + [utils.EOS_ID],
                       self.trg_vocab_size, self._t2t_unk_id))
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = trainer_lib.create_hparams(hparams_set_name)
            self._add_problem_hparams(hparams, problem_name)
            translate_model = registry.model(model_name)(
                hparams, tf.estimator.ModeKeys.EVAL)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None, None], 
                                               name="sgnmt_targets")
            shp = tf.shape(self._targets_var)
            bsz = shp[0]
            inputs = tf.tile(tf.expand_dims(self._inputs_var, 0), [bsz, 1])
            features = {"inputs": expand_input_dims_for_t2t(inputs,
                                                            batched=True), 
                        "targets": expand_input_dims_for_t2t(self._targets_var,
                                                             batched=True)}
            translate_model.prepare_features_for_infer(features)
            translate_model._fill_problem_hparams_features(features)
            logits, _ = translate_model(features)
            logits = tf.squeeze(logits, [2, 3])
            self._log_probs = log_prob_from_logits(logits)
            diag_logits = gather_2d(logits, tf.expand_dims(tf.range(bsz), 1))
            self._diag_log_probs = log_prob_from_logits(diag_logits)
            no_pad = tf.cast(tf.not_equal(
                self._targets_var, text_encoder.PAD_ID), tf.float32)
            flat_bsz = shp[0] * shp[1]
            word_scores = gather_2d(
                tf.reshape(self._log_probs, [flat_bsz, -1]),
                tf.reshape(self._targets_var, [flat_bsz, 1]))
            word_scores = tf.reshape(word_scores, (shp[0], shp[1])) * no_pad
            self._sentence_scores = tf.reduce_sum(word_scores, -1)
            self.mon_sess = self.create_session()

    def _ins_op(self, pos, token):
        """Returns a copy of trg sentence after an insertion."""
        return self.trg_sentence[:pos] + [token] + self.trg_sentence[pos:]

    def _sub_op(self, pos, token):
        """Returns a copy of trg sentence after a substitution."""
        ret = list(self.trg_sentence)
        ret[pos] = token
        return ret

    def _del_op(self, pos):
        """Returns a copy of trg sentence after a deletion."""
        return self.trg_sentence[:pos] + self.trg_sentence[pos+1:]

    def _top_n(self, scores, sort=False):
        """Sorted indices of beam_size best entries along axis 1"""
        costs = -scores
        costs[:, utils.EOS_ID] = utils.INF
        top_n_indices = np.argpartition(
            costs,
            self.beam_size, 
            axis=1)[:, :self.beam_size]
        if not sort:
            return top_n_indices
        b_indices = np.expand_dims(np.arange(top_n_indices.shape[0]), axis=1)
        sorted_indices = np.argsort(costs[b_indices, top_n_indices], axis=1)
        return top_n_indices[b_indices, sorted_indices]

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        next_sentences = {}
        logging.debug("EditT2T: Exploring score=%f sentence=%s" 
                      % (self.cur_score, " ".join(map(str, self.trg_sentence))))
        n_trg_words = len(self.trg_sentence)
        if n_trg_words > EditT2TPredictor.MAX_SEQ_LEN:
            logging.warn("EditT2T: Target sentence exceeds maximum length (%d)"
                         % EDITT2TPredictor.MAX_SEQ_LEN)
            return {utils.EOS_ID: 0.0}
        # Substitutions
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._targets_var: [self.trg_sentence]})
        top_n = self._top_n(np.squeeze(log_probs, axis=0))
        for pos, cur_token in enumerate(self.trg_sentence[:-1]):
            offset = EditT2TPredictor.SUB_OFFSET 
            offset += EditT2TPredictor.POS_FACTOR * pos
            for token in top_n[pos]:
                if token != cur_token:
                    next_sentences[offset + token] = self._sub_op(pos, token)

        # Insertions
        if n_trg_words < EditT2TPredictor.MAX_SEQ_LEN - 1:
            ins_trg_sentences = np.full((n_trg_words, n_trg_words+1), 999)
            for pos in range(n_trg_words):
                ins_trg_sentences[pos, :pos] = self.trg_sentence[:pos]
                ins_trg_sentences[pos, pos+1:] = self.trg_sentence[pos:]
            diag_log_probs = self.mon_sess.run(self._diag_log_probs,
                {self._inputs_var: self.src_sentence,
                 self._targets_var: ins_trg_sentences})
            top_n = self._top_n(np.squeeze(diag_log_probs, axis=1))
            for pos in range(n_trg_words):
                offset = EditT2TPredictor.INS_OFFSET 
                offset += EditT2TPredictor.POS_FACTOR * pos
                for token in top_n[pos]:
                    next_sentences[offset + token] = self._ins_op(pos, token)
        # Deletions
        idx = EditT2TPredictor.DEL_OFFSET
        for pos in range(n_trg_words - 1): # -1: Do not delete EOS
            next_sentences[idx] = self._del_op(pos)
            idx += EditT2TPredictor.POS_FACTOR
        abs_scores = self._score(next_sentences, n_trg_words + 1)
        rel_scores = {i: s - self.cur_score 
                      for i, s in abs_scores.items()}
        rel_scores[utils.EOS_ID] = 0.0
        return rel_scores

    def _score(self, sentences, n_trg_words=1):
        max_n_sens = max(1, self.batch_size // n_trg_words)
        scores = {}
        batch_ids = []
        batch_sens = []
        for idx, trg_sentence in sentences.items():
            score = self.cache.get(trg_sentence)
            if score is None:
                batch_ids.append(idx)
                np_sen = np.zeros(n_trg_words, dtype=np.int)
                np_sen[:len(trg_sentence)] = trg_sentence
                batch_sens.append(np_sen)
                if len(batch_ids) >= max_n_sens:
                    self._score_single_batch(scores, batch_ids, batch_sens)
                    batch_ids = []
                    batch_sens = []
            else:
                scores[idx] = score
        self._score_single_batch(scores, batch_ids, batch_sens)
        return scores 

    def _score_single_batch(self, scores, ids, trg_sentences):
        "Score sentences and add them to scores and the cache."""
        if not ids:
            return
        batch_scores = self.mon_sess.run(self._sentence_scores,
            {self._inputs_var: self.src_sentence,
             self._targets_var: np.stack(trg_sentences)})
        for idx, sen, score in zip(ids, trg_sentences, batch_scores):
            self.cache.add(sen, score)
            scores[idx] = score

    def _update_cur_score(self):
        self.cur_score = self.cache.get(self.trg_sentence)
        if self.cur_score is None:
            scores = self._score({1: self.trg_sentence}, len(self.trg_sentence))
            self.cur_score = scores[1]
            self.cache.add(self.trg_sentence, self.cur_score)
    
    def initialize(self, src_sentence):
        """Set src_sentence, reset consumed."""
        if self.initial_trg_sentences is None:
            self.trg_sentence = [text_encoder.EOS_ID]
        else:
            self.trg_sentence = self.initial_trg_sentences[self.current_sen_id]
        self.src_sentence = utils.oov_to_unk(
            src_sentence + [text_encoder.EOS_ID], 
            self.src_vocab_size, self._t2t_unk_id)
        self.cache = SimpleTrie()
        self._update_cur_score()
        logging.debug("Initial score: %f" % self.cur_score)
   
    def consume(self, word):
        """Append ``word`` to the current history."""
        if word == utils.EOS_ID:
            return
        pos = (word // EditT2TPredictor.POS_FACTOR) \
              % (EditT2TPredictor.MAX_SEQ_LEN + 1)
        token = word % EditT2TPredictor.POS_FACTOR
        # TODO(fstahlberg): Do not hard code the following section
        op = word // 100000000  
        if op == 1:  # Insertion
            self.trg_sentence = self._ins_op(pos, token)
        elif op == 2:  # Substitution
            self.trg_sentence = self._sub_op(pos, token)
        elif op == 3:  # Deletion
            self.trg_sentence = self._del_op(pos)
        else:
            logging.warn("Invalid edit descriptor %d. Ignoring..." % word)
        self._update_cur_score()
        self.cache.add(self.trg_sentence, utils.NEG_INF)
    
    def get_state(self):
        """The predictor state is the complete target sentence."""
        return self.trg_sentence, self.cur_score
    
    def set_state(self, state):
        """The predictor state is the complete target sentence."""
        self.trg_sentence, self.cur_score = state

    def is_equal(self, state1, state2):
        """Returns true if the target sentence is the same """
        return state1[0] == state2[0]


class FertilityT2TPredictor(T2TPredictor):
    """Use this predictor to integrate fertility models trained with 
    T2T. Fertility models output the fertility for each source word
    instead of target words. We define the fertility of the i-th
    source word in a hypothesis as the number of tokens between the 
    (i-1)-th and the i-th POP token.

    TODO: This is not SOLID (violates substitution principle)
    """

    def _update_scores(self):
        """Call the T2T model in self.mon_sess to update pop_scores
        and other_scores.
        """
        log_probs = self.mon_sess.run(self._log_probs,
           {self._inputs_var: self.src_sentence,
            self._targets_var: self.fertility_history + [text_encoder.PAD_ID]})
        fert_log_probs = [p for p in log_probs[4:]] + [log_probs[utils.UNK_ID]]
        fert_log_probs = fert_log_probs[:10]
        prev_max = utils.NEG_INF
        best_future = []
        for f in fert_log_probs[:0:-1]:
            prev_max = max(prev_max, f)
            best_future.append(prev_max)
        best_future.reverse()
        self.pop_scores = []
        self.other_scores = []
        acc = 0.0
        for score, best_future_score in zip(fert_log_probs[:-1], best_future):
            self.pop_scores.append(score - acc)
            self.other_scores.append(best_future_score - acc)
            acc = best_future_score
    
    def initialize(self, src_sentence):
        """Set src_sentence, compute fertilities for first src word."""
        self.fertility_history = []
        self.n_aligned_words = 0
        self.src_sentence = utils.oov_to_unk(
            src_sentence + [text_encoder.EOS_ID], 
            self.src_vocab_size)
        self._update_scores()

    def predict_next(self):
        """Returns self.pop_scores[n_aligned_words] for POP and EOS."""
        score = utils.common_get(self.pop_scores, self.n_aligned_words, 0.0)
        return {self.pop_id: score, utils.EOS_ID: score, 6: 0.0, 7: 0.0}
   
    def consume(self, word):
        if word == self.pop_id:
            target = 4 + self.n_aligned_words
            if target >= self.trg_vocab_size:
                target = utils.UNK_ID
            self.fertility_history.append(target)
            self.n_aligned_words = 0
            self._update_scores()
        elif word != 6 and word != 7: 
            self.n_aligned_words += 1
    
    def get_state(self):
        return (self.fertility_history, 
                self.n_aligned_words, 
                self.pop_scores, 
                self.other_scores)
    
    def set_state(self, state):
        (self.fertility_history, 
         self.n_aligned_words, 
         self.pop_scores, 
         self.other_scores) = state

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1 == state2

    def get_unk_probability(self, posterior):
        """Returns self.other_scores[n_aligned_words]."""
        return utils.common_get(self.other_scores, self.n_aligned_words, 0.0)


class SegT2TPredictor(_BaseTensor2TensorPredictor):
    """This predictor is designed for document-level T2T models. It 
    differs from the normal t2t predictor in the following ways:

    - In addition to `input` and `targets`, it generates the features
      `inputs_seg`. `targets_seg`, `inputs_pos`, `targets_pos` which
      are used in glue models and the contextual Transformer.
    - The history is pruned when it exceeds a maximum number of <s> 
      symbols. This can be used to reduce complexity for document-level
      models on very long documents. When the maximum number is reached,
      we start removing sentences from ``self.consumed``, starting with
      the sentence which is `begin_margin` away from the document start
      and `end_margin` sentences away from the current sentence.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 t2t_usr_dir,
                 checkpoint_dir,
                 t2t_unk_id=None,
                 n_cpu_threads=-1,
                 max_terminal_id=-1,
                 pop_id=-1):
        """Creates a new document-level T2T predictor. See
        T2TPredictor.__init__().
        
        Args:
            src_vocab_size (int): Source vocabulary size.
            trg_vocab_size (int): Target vocabulary size.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            checkpoint_dir (string): Path to the T2T checkpoint 
                                     directory. The predictor will load
                                     the top most checkpoint in the 
                                     `checkpoints` file.
            t2t_unk_id (int): If set, use this ID to get UNK scores. If
                              None, UNK is always scored with -inf.
            n_cpu_threads (int): Number of TensorFlow CPU threads.
            max_terminal_id (int): If positive, maximum terminal ID. Needs to
                be set for syntax-based T2T models.
            pop_id (int): If positive, ID of the POP or closing bracket symbol.
                Needs to be set for syntax-based T2T models.
        """
        super(SegT2TPredictor, self).__init__(t2t_usr_dir, 
                                             checkpoint_dir, 
                                             src_vocab_size,
                                             trg_vocab_size,
                                             t2t_unk_id, 
                                             n_cpu_threads,
                                             max_terminal_id,
                                             pop_id)
        if not model_name or not problem_name or not hparams_set_name:
            logging.fatal(
                "Please specify t2t_model, t2t_problem, and t2t_hparams_set!")
            raise AttributeError
        self.begin_margin = 3
        self.end_margin = 3
        self.max_sentences = self.begin_margin + self.end_margin
        self.max_sentences = 10000  # TODO: Make configurable. Default disabled
        #self.max_sentences = 20
        #self.max_sentences = 25
        predictor_graph = tf.Graph()
        with predictor_graph.as_default() as g:
            hparams = trainer_lib.create_hparams(hparams_set_name)
            self._add_problem_hparams(hparams, problem_name)
            translate_model = registry.model(model_name)(
                hparams, tf.estimator.ModeKeys.PREDICT)
            self._inputs_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                              name="sgnmt_inputs")
            self._targets_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                               name="sgnmt_targets")
            self._inputs_seg_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                                  name="sgnmt_inputs_seg")
            self._targets_seg_var = tf.placeholder(dtype=tf.int32, shape=[None], 
                                                   name="sgnmt_targets_seg")
            self._inputs_pos_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                                  name="sgnmt_inputs_pos")
            self._targets_pos_var = tf.placeholder(dtype=tf.int32, shape=[None],
                                                   name="sgnmt_targets_pos")
            features = {
                "inputs": expand_input_dims_for_t2t(self._inputs_var), 
                "targets": expand_input_dims_for_t2t(self._targets_var),
                "inputs_seg": tf.expand_dims(self._inputs_seg_var, 0),
                "targets_seg": tf.expand_dims(self._targets_seg_var, 0),
                "inputs_pos": tf.expand_dims(self._inputs_pos_var, 0), 
                "targets_pos": tf.expand_dims(self._targets_pos_var, 0)
            }
            translate_model.prepare_features_for_infer(features)
            translate_model._fill_problem_hparams_features(features)
            logits, _ = translate_model(features)
            logits = tf.squeeze(logits, [0, 1, 2, 3])
            self._log_probs = log_prob_from_logits(logits)
            self.mon_sess = self.create_session()

    def initialize(self, src_sentence):
        self.consumed = []
        self.src_sentence = utils.oov_to_unk(
            src_sentence + [text_encoder.EOS_ID], 
            self.src_vocab_size, self._t2t_unk_id)
        self.src_seg, self.src_pos = self._gen_seg_and_pos(self.src_sentence)
        self.history_sentences = [[]]

    def _gen_seg_and_pos(self, glued, trg=False):
        seg = []
        pos = []
        cur_seg = 1
        cur_pos = 0
        for w in glued:
            seg.append(cur_seg)
            pos.append(cur_pos)
            if w == utils.GO_ID:
                cur_seg += 1
                cur_pos = 0
            else:
                cur_pos += 1
        if trg:
            seg.append(cur_seg)
            pos.append(cur_pos)
        return seg, pos
   
    def consume(self, word):
        self.history_sentences[-1].append(
            word if word < self.trg_vocab_size else self._t2t_unk_id)
        if word == utils.GO_ID:
            if False and len(self.history_sentences) > self.max_sentences:
                logging.debug("Pruning document level history...")
                self.history_sentences = (
                    self.history_sentences[:self.begin_margin] 
                    + self.history_sentences[-self.end_margin:])
            self.history_sentences.append([])
    
    def get_state(self):
        return self.history_sentences
    
    def set_state(self, state):
        self.history_sentences = state

    def predict_next(self):
        """Call the T2T model in self.mon_sess."""
        if len(self.history_sentences) > self.max_sentences:
            return {}
        consumed = [w for s in self.history_sentences for w in s]
        trg_seg, trg_pos = self._gen_seg_and_pos(consumed, trg=True)
        log_probs = self.mon_sess.run(self._log_probs,
            {self._inputs_var: self.src_sentence,
             self._inputs_seg_var: self.src_seg,
             self._inputs_pos_var: self.src_pos,
             self._targets_var: consumed + [text_encoder.PAD_ID],
             self._targets_seg_var: trg_seg,
             self._targets_pos_var: trg_pos})
        log_probs[text_encoder.PAD_ID] = utils.NEG_INF
        return log_probs
    
    def is_equal(self, state1, state2):
        """Returns true if the (pruned) history is the same """
        return state1 == state2

    def get_unk_probability(self, posterior):
        """Fetch posterior[t2t_unk_id]"""
        if len(self.history_sentences) > self.max_sentences:
            return 0.0
        return utils.common_get(posterior, self._t2t_unk_id, utils.NEG_INF)
