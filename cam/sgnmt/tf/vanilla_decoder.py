"""This module bypasses the normal predictor framework and decodes 
directly with TensorFlow...
"""
from cam.sgnmt.decoding.core import Decoder


class TensorFlowNMTVanillaDecoder(Decoder):
    """TODO: Implement
    """
    
    def __init__(self, path, config):
        """Set up the NMT model used by the decoder.
        
        Args:
            path (string):  Path to the NMT model file (.npz)
            config (dict): NMT configuration
        """
        super(TensorFlowNMTVanillaDecoder, self).__init__()
        self.config = config
        raise NotImplementedError
    
    def decode(self, src_sentence):
        """TODO: Implement
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S>
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        """
        raise NotImplementedError
    
    def has_predictors(self):
        """Always returns true. """
        return True
