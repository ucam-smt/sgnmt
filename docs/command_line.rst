
Command-line reference
=======================

SGNMT provides ``decode.py`` for decoding and ``train.py`` for NMT training. Both
scripts can be configured via command line or configuration file. For a quick
overview of available parameters use ``--help``::

    python decode.py --help
    python train.py --help

The complete and detailed list of parameters is provided below.

Decoding
---------

.. argparse::
   :module: cam.sgnmt.blocks.ui
   :func: get_parser
   :prog: decode.py

Training
---------

The training script follows the NMT training example in blocks, but it adds an
option for enabling reshuffling the training data between epochs, and fixing
word embedding which might be used in later training stages.

.. argparse::
   :module: cam.sgnmt.blocks.ui
   :func: get_train_parser
   :prog: train.py

