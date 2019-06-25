Subword Neural Machine Translation
==================================

This subdirectory is cloned from

  https://github.com/rsennrich/subword-nmt

and slightly modified to be compatible with SGNMT. It contains preprocessing scripts 
to segment text into subword units. The main difference to the original repository
is that we use explicit end-of-word markers rather than '@@' separator symbols.

USAGE INSTRUCTIONS
------------------

See http://ucam-smt.github.io/sgnmt/html/tutorial.html

PUBLICATIONS
------------

The segmentation methods are described in:

Rico Sennrich, Barry Haddow and Alexandra Birch (2016):
    Neural Machine Translation of Rare Words with Subword Units
    Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
