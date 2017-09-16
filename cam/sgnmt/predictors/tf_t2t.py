"""This is the interface to the tensor2tensor library.

https://github.com/tensorflow/tensor2tensor

Alternatively, you may use the following fork which has been tested in
combination with SGNMT.

https://github.com/fstahlberg/tensor2tensor

This predictor can read any model trained with tensor2tensor which
includes the transformer model, convolutional models, and RNN-based
sequence models.
"""

from cam.sgnmt.predictors.core import Predictor
from cam.sgnmt import utils

try:
    # Requires tensor2tensor
    from tensor2tensor.utils import decoding
    from tensor2tensor.utils import trainer_utils
    from tensor2tensor.utils import usr_dir    
    import tensorflow as tf
except ImportError:
    pass # Deal with it in decode.py


class T2TPredictor(Predictor):
    """This predictor implements scoring with Tensor2Tensor models. We
    follow the decoder implementation in T2T and do not reuse network
    states in decoding. We rather compute the full forward pass along
    the current history. Therefore, the decoder state is simply the
    the full history of consumed words.
    """
    
    def __init__(self,
                 t2t_usr_dir,
                 model_name,
                 problem_name,
                 hparams_set_name,
                 checkpoint_dir):
        """Creates a new T2T predictor.
        
        Args:
            t2t_usr_dir (string): See --t2t_usr_dir in tensor2tensor.
            model_name (string): T2T model name.
            problem_name (string): T2T problem name.
            hparams_set_name (string): T2T hparams set name.
            checkpoint_dir (string): Path to the T2T checkpoint directory. The
                                     predictor will load the top most 
                                     checkpoint in the `checkpoints` file.
        """
        super(T2TPredictor, self).__init__()
        self.consumed = []
        self.src_sentence = []
        tf.logging.set_verbosity(tf.logging.INFO)
        usr_dir.import_usr_dir(t2t_usr_dir)
        trainer_utils.log_registry()
        hparams = trainer_utils.create_hparams(hparams_set_name,
                                               problem_name,
                                               "dummy_data_dir")
        self.estimator, _ = trainer_utils.create_experiment_components(
            hparams=hparams,
            output_dir=checkpoint_dir,
            data_dir="dummy_data_dir",
            model_name=model_name)
                
    def get_unk_probability(self, posterior):
        pass
    
    def predict_next(self):
        pass
    
    def initialize(self, src_sentence):
        self.consumed = []
        self.src_sentence = src_sentence
    
    def consume(self, word):
        self.consumed.append(word)
    
    def get_state(self):
        return self.consumed
    
    def set_state(self, state):
        """Set the predictor state. """
        self.consumed = state

    def reset(self):
        """Empty method. """
        pass

    def is_equal(self, state1, state2):
        """Returns true if the state is the same """
        return state1 == state2
