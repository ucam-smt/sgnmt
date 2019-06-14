"""This script applies a word map to sentences in stdin. If --dir is
set to s2i, the word strings in stdin are converted to their ids. If
--dir is i2s, we convert word IDs to their readable representations.
"""

import logging
import argparse
import sys

def load_wmap(path, inverse=False):
    with open(path) as f:
        d = dict(line.strip().split(None, 1) for line in f)
        if inverse:
            d = dict(zip(d.values(), d.keys()))
        return d

def detok_id(line):
    return line

def detok_eow(line):
    return line.replace(" ", "").replace("</w>", " ").strip()

parser = argparse.ArgumentParser(description='Convert between written and ID representation of words. '
                                 'The index 0 is always used as UNK token, wmap entry for 0 is ignored. '
                                 'Usage: python apply_wmap.py < in_sens > out_sens')
parser.add_argument('-d','--dir', help='s2i: convert to IDs (default), i2s: convert from IDs',
                    required=False)
parser.add_argument('-m','--wmap', help='Word map to apply (format: see -i parameter)',
                    required=True)
parser.add_argument('-u','--unk_id', default=3, help='UNK id')
parser.add_argument('-i','--inverse_wmap', help='Use this argument to use word maps with format "id word".'
                    ' Otherwise the format "word id" is assumed', action='store_true')
parser.add_argument('-t', '--tokenization', default='id', choices=['id', 'eow', 'mixed'],
                    help='This parameter adds support for tokenizations below word level. Choose '
                    '"id" if no further postprocessing should be applied after the mapping. "eow"'
                    'removes all blanks and replaces </w> tokens with new blanks. This can be used '
                    'for subword units with explicit end-of-word markers. Use "eow" for pure character-level '
                    'tokenizations')

args = parser.parse_args()

wmap = load_wmap(args.wmap, args.inverse_wmap)
unk = str(args.unk_id)
if args.dir and args.dir == 'i2s': # inverse wmap
    wmap = dict(zip(wmap.values(), wmap.keys()))
    unk = "NOTINWMAP"

detok = detok_id
if args.tokenization == 'eow':
    detok = detok_eow

# do not use for line in sys.stdin because incompatible with -u option
# required for lattice mert
while True:
    line = sys.stdin.readline()
    if not line: break # EOF
    print(detok(' '.join([wmap[w] if (w in wmap) else unk for w in line.strip().split()])))

