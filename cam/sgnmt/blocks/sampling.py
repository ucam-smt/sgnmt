"""This module is derived from the ``sampling`` module in the Blocks
NMT example, but reduced to providing functionality for model selection
according the BLEU score on the dev set.
"""

from __future__ import print_function

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch
import logging
import numpy
import operator
import os
import re
import signal
from subprocess import Popen, PIPE
import time

from cam.sgnmt import utils
from cam.sgnmt.blocks.sparse_search import SparseBeamSearch
from cam.sgnmt.misc.sparse import FlatSparseFeatMap


logger = logging.getLogger(__name__)


class BleuValidator(SimpleExtension):
    """Implements early stopping based on BLEU score. This class is 
    still very similar to the ``BleuValidator`` in the NMT Blocks
    example.
    
    TODO: Refactor, make this more similar to the rest of SGNMT, use
    vanilla_decoder.py
    """

    def __init__(self, 
                 source_sentence, 
                 samples, 
                 model, 
                 data_stream,
                 config, 
                 n_best=1, 
                 track_n_models=1,
                 normalize=True, 
                 store_full_main_loop=False, 
                 **kwargs):
        """Creates a new extension which adds model selection based on
        the BLEU score to the training main loop.
        
        Args:
            source_sentence (Variable): Input variable to the sampling
                                        computation graph
            samples (Variable): Samples variable of the CG
            model (NMTModel): See the model module
            data_stream (DataStream): Data stream to the development 
                                      set
            config (dict): NMT configuration
            n_best (int): beam size
            track_n_models (int): Number of n-best models for which to 
                                  create checkpoints.
            normalize (boolean): Enables length normalization
            store_full_main_loop (boolean): Stores the iteration state
                                            in the old style of
                                            Blocks 0.1. Not recommended
        """
        super(BleuValidator, self).__init__(**kwargs)
        self.store_full_main_loop = store_full_main_loop
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.best_models = []
        self.val_bleu_curve = []
        self.multibleu_cmd = (self.config['bleu_script'] % self.config['val_set_grndtruth']).split()
        logging.debug("BLEU command: %s" % self.multibleu_cmd)

        self.src_sparse_feat_map = config['src_sparse_feat_map'] if config['src_sparse_feat_map'] \
                                                                 else FlatSparseFeatMap()
        if config['trg_sparse_feat_map']:
            self.trg_sparse_feat_map = config['trg_sparse_feat_map']
            self.beam_search = SparseBeamSearch(
                                 samples=samples, 
                                 trg_sparse_feat_map=self.trg_sparse_feat_map) 
        else:
            self.trg_sparse_feat_map = FlatSparseFeatMap()
            self.beam_search = BeamSearch(samples=samples)
        
        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                        'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()
                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logging.info("BleuScores Reloaded")
            except:
                logging.info("BleuScores not Found")

    def do(self, which_callback, *args):
        """Decodes the dev set and stores checkpoints in case the BLEU
        score has improved.
        """
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):
        """Evaluate model and store checkpoints. """
        logging.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0
        ftrans = open(self.config['saveto'] + '/validation_out.txt', 'w')
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            seq = self.src_sparse_feat_map.words2dense(utils.oov_to_unk(
                line[0], self.config['src_vocab_size']))
            if self.src_sparse_feat_map.dim > 1: # sparse src feats
                input_ = numpy.transpose(
                             numpy.tile(seq, (self.config['beam_size'], 1, 1)),
                             (2,0,1))
            else: # word ids on the source side
                input_ = numpy.tile(seq, (self.config['beam_size'], 1))
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=utils.EOS_ID,
                    ignore_first_eol=True)
            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans = trans[best]
                    if trans and trans[-1] == utils.EOS_ID:
                        trans = trans[:-1]
                    trans_out = ' '.join([str(w) for w in trans])
                except ValueError:
                    logging.info(
                        "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'
                if j == 0:
                    # Write to subprocess and file if it exists
                    print(trans_out, file=mb_subprocess.stdin)
                    print(trans_out, file=ftrans)
            if i != 0 and i % 100 == 0:
                logging.info(
                    "Translated {} lines of validation set...".format(i))

            mb_subprocess.stdin.flush()
        logging.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        ftrans.close()
        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        logging.info(stdout)
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logging.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None
        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        logging.info(bleu_score)
        mb_subprocess.terminate()
        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def save_parameter_values(self, param_values, path):
        ''' This method is copied from blocks.machine_translation.checkpoint '''
        param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
        numpy.savez(path, **param_values)

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])
            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logging.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)
            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))
            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            # fs439: introduce store_full_main_loop and 
            # storing best_bleu_params_* files
            if self.store_full_main_loop:
                logging.info("Saving full main loop model {}".format(model.path))
                numpy.savez(model.path, 
                            **self.main_loop.model.get_parameter_dict())
            else:
                logging.info("Saving model parameters {}".format(model.path))
                params_to_save = self.main_loop.model.get_parameter_values()
                self.save_parameter_values(params_to_save, model.path)
            numpy.savez(
                os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    """Utility class for keeping track of evaluated models."""

    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        gen_path = os.path.join(
            path, 'best_bleu_params_%d_BLEU%.2f.npz' %
            (int(time.time()), self.bleu_score) if path else None)
        return gen_path
