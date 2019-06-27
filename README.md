# SGNMT


SGNMT is an open-source framework for neural machine translation (NMT) and other sequence prediction
tasks. The tool provides a flexible platform which allows pairing NMT with various other models such 
as language models, length models, or bag2seq models. It supports rescoring both n-best lists and lattices.
A wide variety of search strategies is available for complex decoding problems. 

SGNMT is compatible with the following NMT toolkits:

-  [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) ([TensorFlow](https://www.tensorflow.org/))
-  [fairseq](https://github.com/pytorch/fairseq) ([PyTorch](https://pytorch.org/))

Old SGNMT versions (0.x) are compatible with:

- [(extended) TF seq2seq tutorial](https://github.com/ehasler/tensorflow) ([TensorFlow](https://www.tensorflow.org/))
- [Blocks](http://blocks.readthedocs.io/en/latest/) ([Theano](http://deeplearning.net/software/theano/))


Features:

- Syntactically guided neural machine translation (NMT lattice rescoring)
- n-best list rescoring with NMT
- Integrating external n-gram posterior probabilities used in MBR
- Ensemble NMT decoding
- Forced NMT decoding
- Integrating language models
- Different search algorithms (beam, A\*, depth first search, greedy...)
- Target sentence length modelling
- Bag2Sequence models and decoding algorithms
- Joint decoding with word- and subword/character-level models
- Hypothesis recombination
- Heuristic search
- ...

### Documentation

Please see the [full SGNMT documentation](http://ucam-smt.github.io/sgnmt/html/) for more information.

### Contributors

- Felix Stahlberg, University of Cambridge
- Eva Hasler, SDL Research
- Danielle Saunders, University of Cambridge

### Citing

If you use SGNMT in your work, please cite the following paper:

Felix Stahlberg, Eva Hasler, Danielle Saunders, and Bill Byrne.
SGNMT - A Flexible NMT Decoding Platform for Quick Prototyping of New Models and Search Strategies.
In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (EMNLP 17 Demo Session)*, September 2017. Copenhagen, Denmark.
[arXiv](https://arxiv.org/abs/1707.06885)
 
