# SGNM

SGNMT is an open-source framework for neural machine translation (NMT). The tool provides
a flexible platform which allows pairing NMT with various other models such as language models,
length models, or bag2seq models. It supports rescoring both n-best lists and lattice rescoring.
A wide variety of search strategies is available for complex decoding problems. 
SGNMT is compatible with Blocks/Theano and TensorFlow. Features:

- Syntactically guided neural machine translation (NMT lattice rescoring)
- NMT support for Blocks/Theano and TensorFlow
- n-best list rescoring with NMT
- Ensemble NMT decoding
- Forced NMT decoding
- Integrating language models (Kneser-Ney, NPLM, RNNLM)
- Different search algorithms (beam, A\*, depth first search, greedy...)
- Target sentence length modelling
- NMT training with options for reshuffling and fixing word embeddings
- Bag2Sequence models and decoding algorithms
- Custom distributed word representations
- Joint decoding with word- and subword/character-level models
- Hypothesis recombination
- Heuristic search
- Neural word alignment
- ...

### Documentation

Please see the [full SGNMT documentation](http://ucam-smt.github.io/sgnmt/html/) for more information.

### Citing

If you use SGNMT in your work, please cite the following paper:

Felix Stahlberg, Eva Hasler, Aurelien Waite, and Bill Byrne. Syntactically guided neural machine translation. 
In *Proceedings of the 54th annual meeting of the Association for Computational Linguistics (ACL 16)*, 
August 2016. Berlin, Germany. [arXiv](http://arxiv.org/abs/1605.04569)
 
