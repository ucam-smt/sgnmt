"""This is the interface to the fairseq library.

https://github.com/pytorch/fairseq

The fairseq predictor can read any model trained with fairseq.
"""

import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor


try:
    # Requires fairseq
    from fairseq import checkpoint_utils, options, tasks
    from fairseq import utils as fairseq_utils
    from fairseq.sequence_generator import EnsembleModel
    import torch
    import numpy as np
except ImportError:
    pass # Deal with it in decode.py


FAIRSEQ_INITIALIZED = False
"""Set to true by _initialize_fairseq() after first constructor call."""


def _initialize_fairseq(user_dir):
    global FAIRSEQ_INITIALIZED
    if not FAIRSEQ_INITIALIZED:
        logging.info("Setting up fairseq library...")
        if user_dir:
            args = type("", (), {"user_dir": user_dir})()
            fairseq_utils.import_user_module(args)
        FAIRSEQ_INITIALIZED = True


class FairseqPredictor(Predictor):
    """Predictor for using fairseq models."""

    def __init__(self, model_path, user_dir, lang_pair, n_cpu_threads=-1):
        """Initializes a fairseq predictor.

        Args:
            model_path (string): Path to the fairseq model (*.pt). Like
                                 --path in fairseq-interactive.
            lang_pair (string): Language pair string (e.g. 'en-fr').
            user_dir (string): Path to fairseq user directory.
            n_cpu_threads (int): Number of CPU threads. If negative,
                                 use GPU.
        """
        super(FairseqPredictor, self).__init__()
        _initialize_fairseq(user_dir)
        self.use_cuda = torch.cuda.is_available() and n_cpu_threads < 0

        parser = options.get_generation_parser()
        input_args = ["--path", model_path, os.path.dirname(model_path)]
        if lang_pair:
            src, trg = lang_pair.split("-")
            input_args.extend(["--source-lang", src, "--target-lang", trg])
        args = options.parse_args_and_arch(parser, input_args)

        # Setup task, e.g., translation
        task = tasks.setup_task(args)
        self.src_vocab_size = len(task.source_dictionary)
        self.trg_vocab_size = len(task.target_dictionary)
        self.pad_id = task.source_dictionary.pad()

        # Load ensemble
        logging.info('Loading fairseq model(s) from {}'.format(model_path))
        self.models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'),
            task=task,
        )

        # Optimize ensemble for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1,
                need_attn=False,
            )
            if self.use_cuda:
                model.cuda()
        self.model = EnsembleModel(self.models)
        self.model.eval()

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
                
    def predict_next(self):
        """Call the fairseq model."""
        lprobs, _ = self.model.forward_decoder(
            torch.LongTensor([self.consumed]), self.encoder_outs
        )
        lprobs[0, self.pad_id] = utils.NEG_INF
        return np.array(lprobs[0])
    
    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""
        self.consumed = []
        src_tokens = torch.LongTensor([
            utils.oov_to_unk(src_sentence + [utils.EOS_ID],
                             self.src_vocab_size)])
        src_lengths = torch.LongTensor([len(src_sentence) + 1])
        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        self.encoder_outs = self.model.forward_encoder({
            'src_tokens': src_tokens,
            'src_lengths': src_lengths})
        self.consumed = [utils.GO_ID or utils.EOS_ID]
        # Reset incremental states
        for model in self.models:
            self.model.incremental_states[model] = {}
   
    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)
    
    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [self.model.incremental_states[m] 
                               for m in self.models]
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed, inc_states = state
        for model, inc_state in zip(self.models, inc_states):
            self.model.incremental_states[model] = inc_state

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]

