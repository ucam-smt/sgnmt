# SGNMT

SGNMT is a tool for neural machine translation (NMT). It stands for Syntactically
Guided Neural Machine Translation as it is designed to work well with the
ucam-smt syntactical SMT system or other non-neural SMT systems. It builds up on the 
Blocks NMT example and adds support for n-best and lattice recoring, language models 
and much more. Currently, it supports

- Syntactically guided neural machine translation (NMT lattice rescoring)
- n-best list rescoring with NMT
- Ensemble NMT decoding
- Forced NMT decoding
- Integrating n-gram language models (srilm and nplm)
- Different search algorithms (beam, A\*, depth first search, greedy...)
- Target sentence length modelling
- NMT training with options for reshuffling and fixing word embeddings
- ...

### Documentation

Please see the [full SGNMT documentation](http://ucam-smt.github.io/sgnmt/html/) for more information.

### Citing

If you use SGNMT in your work, please cite the following paper:

Felix Stahlberg, Eva Hasler, Aurelien Waite, and Bill Byrne. Syntactically guided neural machine translation. 
In *Proceedings of the 54th annual meeting of the Association for Computational Linguistics (ACL 16)*, 
August 2016. Berlin, Germany. [arXiv](http://arxiv.org/abs/1605.04569)
 
