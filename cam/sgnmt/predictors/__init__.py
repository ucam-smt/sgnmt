"""Predictors are the scoring modules used in SGNMT. They can be used 
together to form a combined search space and scores. Note that the
configuration of predictors is not decoupled with the central
configuration (yet). Therefore, new predictors need to be referenced to
in ``blocks.decode``, and their configuration parameters need to be 
added to ``blocks.ui``. 
"""