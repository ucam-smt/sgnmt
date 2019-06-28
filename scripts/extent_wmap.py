'''
This script reads words from stdin and checks whether they are in
the given wmap. If not, extent the wmap without changing existing ids
and output the updated wmap
'''

import argparse
import sys
import operator

def load_wmap(path, inverse_wmap):
    with open(path) as f:
        if inverse_wmap:
            d = dict((word, int(word_id)) for (word_id, word) in (line.strip().split(None, 1) for line in f))
        else:
            d = dict((word, int(word_id)) for (word, word_id) in (line.strip().split(None, 1) for line in f))
        return d

parser = argparse.ArgumentParser(description='Outputs a wmap which contains all words in stdin '
                                 'but uses consistent word IDs with -m')
parser.add_argument('-m','--wmap', help='Word map to apply (format: word id)',
                    required=True)
parser.add_argument('-i','--inverse_wmap', help='Use this argument to use word maps with format "id word".'
                    ' Otherwise the format "word id" is assumed', action='store_true')
args = parser.parse_args()

wmap = load_wmap(args.wmap, args.inverse_wmap)

next_id = max(wmap.values()) + 1
for line in sys.stdin:
    for w in line.strip().split():
        if w and (not w in wmap):
            wmap[w] = next_id
            next_id = next_id + 1

# Print wmap
if args.inverse_wmap:
    for (word, word_id) in sorted(wmap.items(), key=operator.itemgetter(1)):
        print("%d\t%s" % (word_id, word))
else:
    for (word, word_id) in sorted(wmap.items(), key=operator.itemgetter(1)):
        print("%s %d" % (word, word_id))
