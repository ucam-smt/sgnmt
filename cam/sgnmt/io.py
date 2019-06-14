"""This module is responsible for converting input text to integer
representations (encode()), and integer translation hypotheses back to 
readable text (decode()). In the default configuration, this conversion
is an identity mapping: Source sentences are provided in integer
representations, and output files also contain indexed sentences. 
"""

import logging


def encode(src_sentence):
    """Converts the source sentence in string representation to a
    sequence of token IDs. Depending on the configuration of this
    module, it applies word maps and/or subword/character segmentation
    on the input.

    Args:
        src_sentence (string): A single input sentence

    Returns:
        List of integers.
    """
    return [int(w) for w in src_sentence.split()]


def decode(trg_sentence):
    """Converts the target sentence represented as sequence of token
    IDs to a string representation.

    Args:
        trg_sentence (list): A sequence of integers (token IDs)

    Returns:
        string.
    """
    return " ".join(map(str, trg_sentence))


# Word maps


src_wmap = {}
"""Source language word map (word -> id)"""


trg_wmap = {}
"""Target language word map (id -> word)"""


def load_src_wmap(path):
    """Loads a source side word map from the file system.
    
    Args:
        path (string): Path to the word map (Format: word id)
    
    Returns:
        dict. Source word map (key: word, value: id)
    """
    global src_wmap
    if not path:
        src_wmap = {}
        return src_wmap
    with open(path) as f:
        src_wmap = dict(map(lambda e: (e[0], int(e[-1])),
                        [line.strip().split() for line in f]))
    return src_wmap


def load_trg_wmap(path):
    """Loads a target side word map from the file system.
    
    Args:
        path (string): Path to the word map (Format: word id)
    
    Returns:
        dict. Source word map (key: id, value: word)
    """
    global trg_wmap
    if not path:
        trg_wmap = {}
        return trg_wmap
    with open(path) as f:
        trg_wmap = dict(map(lambda e: (int(e[-1]), e[0]),
                        [line.strip().split() for line in f]))
    return trg_wmap


