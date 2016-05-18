.. SGNMT documentation master file, created by
   sphinx-quickstart on Tue May 17 17:32:32 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SGNMT
========

SGNMT is a tool for neural machine translation (NMT). It stands for Syntactically
Guided Neural Machnie Translation as it is designed to work well with the
ucam-smt syntactical SMT system or other non-neural SMT systems. It builds up on the 
Blocks NMT example and adds support for n-best and lattice recoring, language models 
and much more. 


Contents
-------------
.. toctree::
   :maxdepth: 1

   setup
   tutorial
   command_line
   Predictors <cam.sgnmt.predictors>
   Decoders <cam.sgnmt.decoding>
   All modules <cam.sgnmt>
   publications

Quickstart
------------

For example, NMT decoding can be started with this command::

    $ python decode.py --predictors nmt --src_test sentences.txt

where sentences.txt is a plain (indexed) text file with sentences. Rescoring OpenFST
lattices with NMT is also straight-forward::

    $ python decode.py --predictors nmt,fst --fst_path lattices/%d.fst --src_test sentences.txt

See the documentation for more information.

Features
------------

- Syntactically guided neural machine translation (NMT lattice rescoring)
- n-best list rescoring with NMT
- Ensemble NMT decoding
- Forced NMT decoding
- Integrating n-gram language models (srilm and nplm)
- Different search algorithms (beam, A*, depth first search, greedy...)
- Target sentence length modelling
- NMT training with options for reshuffling and fixing word embeddings
- ...

Project links
--------------

- Issue Tracker: http://github.com/ucam-smt/sgnmt/issues
- Source Code: http://github.com/ucam-smt/sgnmt

License
---------

The project is licensed under the Apache 2 license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

