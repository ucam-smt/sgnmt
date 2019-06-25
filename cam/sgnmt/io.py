"""This module is responsible for converting input text to integer
representations (encode()), and integer translation hypotheses back to 
readable text (decode()). In the default configuration, this conversion
is an identity mapping: Source sentences are provided in integer
representations, and output files also contain indexed sentences. 
"""

import logging
from cam.sgnmt import utils
import codecs


def encode(src_sentence):
    """Converts the source sentence in string representation to a
    sequence of token IDs. Depending on the configuration of this
    module, it applies word maps and/or subword/character segmentation
    on the input. This method calls ``encoder.encode()``.

    Args:
        src_sentence (string): A single input sentence

    Returns:
        List of integers.
    """
    return encoder.encode(src_sentence)


def decode(trg_sentence):
    """Converts the target sentence represented as sequence of token
    IDs to a string representation. This method calls
    ``decoder.decode()``.

    Args:
        trg_sentence (list): A sequence of integers (token IDs)

    Returns:
        string.
    """
    return decoder.decode(trg_sentence)


def initialize(args):
    """Initializes the ``io`` module, including loading word maps and
    other resources needed for encoding and decoding. Subsequent calls
    of ``encode()`` and ``decode()`` will process input as specified in
    ``args``.

    Args:
        args (object): SGNMT config
    """
    global encoder, decoder
    if args.wmap:
        load_src_wmap(args.wmap)
        load_trg_wmap(args.wmap)
    if args.src_wmap:
        load_src_wmap(args.src_wmap)
    if args.trg_wmap:
        load_trg_wmap(args.trg_wmap)
    if args.preprocessing == "id":
        encoder = IDEncoder()
    elif args.preprocessing == "word":
        encoder = WordEncoder()
    elif args.preprocessing == "char":
        encoder = CharEncoder()
    elif args.preprocessing == "bpe":
        encoder = BPEEncoder(args.bpe_codes)
    else:
        raise NotImplementedError("Unknown preprocessing")
    if args.postprocessing == "id":
        decoder = IDDecoder()
    elif args.postprocessing == "word":
        decoder = WordDecoder()
    elif args.postprocessing == "char":
        decoder = CharDecoder()
    elif args.postprocessing == "bpe":
        decoder = BPEDecoder()
    else:
        raise NotImplementedError("Unknown postprocessing")


# Encoders and decoders


encoder = None
"""Encoder called in encode(). Initialized in initialize()."""


decoder = None
"""Decoder called in decode(). Initialized in initialize()."""


class Encoder(object):
    """Super class for IO encoders."""

    def encode(self, src_sentence):
        """Converts the source sentence in string representation to a
        sequence of token IDs. Depending on the configuration of this
        module, it applies word maps and/or subword/character segmentation
        on the input.

        Args:
            src_sentence (string): A single input sentence

        Returns:
            List of integers.
        """
        raise NotImplementedError


class Decoder(object):
    """"Super class for IO decoders."""

    def decode(self, trg_sentence):
        """Converts the target sentence represented as sequence of token
        IDs to a string representation.

        Args:
            trg_sentence (list): A sequence of integers (token IDs)

        Returns:
            string.
        """
        raise NotImplementedError


class IDEncoder(Encoder):
    """Encoder for ID mapping."""

    def encode(self, src_sentence):
        return [int(w) for w in src_sentence.split()]


class IDDecoder(Decoder):
    """"Decoder for ID mapping."""

    def decode(self, trg_sentence):
        return " ".join(map(str, trg_sentence))


class WordEncoder(Encoder):
    """Encoder for word based mapping."""

    def encode(self, src_sentence):
        return [src_wmap.get(w, utils.UNK_ID) 
                for w in src_sentence.split()]


class WordDecoder(Decoder):
    """"Decoder for word based mapping."""

    def decode(self, trg_sentence):
        return " ".join(trg_wmap.get(w, "<UNK>") for w in trg_sentence)


class CharEncoder(Encoder):
    """Encoder for char mapping."""

    def encode(self, src_sentence):
        return [src_wmap.get(c, utils.UNK_ID) 
                for c in src_sentence.replace(" ", "_")]


class CharDecoder(Decoder):
    """"Decoder for char mapping."""

    def decode(self, trg_sentence):
        return "".join(
              trg_wmap.get(c, "<UNK>") for c in trg_sentence).replace("_", " ")


# The BPE implementation is adapted from Rico Sennrich's subword_nmt 
# repository:
# https://github.com/rsennrich/subword-nmt
# For learning BPE encodings we recommend using the version of 
# subword_nmt in SGNMT's scripts/ directory to avoid version mismatch.


class BPE(object):
    """Adapted from the BPE class in subword-nmt. An important
    difference is that we use an empty separator symbol by default
    since word endings are marked with </w>.
    """

    def __init__(self, codes_path, separator=''):
        
        with codecs.open(codes_path, encoding='utf-8') as codes:
            self.bpe_codes = [tuple(item.split()) for item in codes]
         
        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i)
                for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.separator = separator

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = []
        for word in sentence.split():
            new_word = self.encode(word)

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return ' '.join(output)

    def get_pairs(self, word):
        """Return set of symbol pairs in a word.

        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def encode(self, orig, cache={}):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """

        if orig in ["<s>", "</s>"]:
            return [orig]
        if orig in cache:
            return cache[orig]

        word = tuple(orig) + ('</w>',)
        pairs = self.get_pairs(word)

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_codes.get(pair, float('inf')))
            if bigram not in self.bpe_codes:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        # don't print end-of-word symbols
        #if word[-1] == '</w>':
        #    word = word[:-1]
        #elif word[-1].endswith('</w>'):
        #    word = word[:-1] + (word[-1].replace('</w>',''),)

        cache[orig] = word
        return word


class BPEEncoder(Encoder):
    """Encoder for BPE mapping."""

    def __init__(self, codes_path):
        self.bpe = BPE(codes_path)

    def encode(self, src_sentence):
        bpe_str = self.bpe.segment(src_sentence)
        bpe_int = []
        for w in bpe_str.split():
            if w not in src_wmap:
                logging.warning("src_wmap does not fully cover bpe_codes ('%s'"
                    " not found in wmap, skipping)" % w)
            else:
                bpe_int.append(src_wmap[w])
        logging.debug("BPE segmentation: '%s' => '%s' (%s)" 
                      % (src_sentence, bpe_str, " ".join(map(str, bpe_int))))
        return bpe_int


class BPEDecoder(Decoder):
    """"Decoder for BPE mapping."""

    def decode(self, trg_sentence):
        return "".join(
           trg_wmap.get(w, "<UNK>") for w in trg_sentence).replace("</w>", " ")


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


