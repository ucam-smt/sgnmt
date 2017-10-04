""""This script implements neural word alignment. The current alignment
procedure relies on a NMT model trained with ``train.py``. We support 
two different alignment models. ``nmt`` uses the alignment weights in
NMT forced decoding as alignment scores. ``nam`` discards the attention 
model, and optimizes the alignment weights given the source and target 
sentence.
"""

import logging
import pprint

from cam.sgnmt.blocks.alignment.nam import align_with_nam
from cam.sgnmt.blocks.alignment.nmt import align_with_nmt
from cam.sgnmt.blocks.nmt import blocks_get_default_nmt_config
from cam.sgnmt.misc.sparse import FileBasedFeatMap
from cam.sgnmt.output import CSVAlignmentOutputHandler, \
    NPYAlignmentOutputHandler, TextAlignmentOutputHandler
from cam.sgnmt.ui import get_blocks_align_parser


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

parser = get_blocks_align_parser()
args = parser.parse_args()

# Get configuration
configuration = blocks_get_default_nmt_config()
for k in dir(args):
    if k in configuration:
        configuration[k] = getattr(args, k)
if configuration['src_sparse_feat_map']:
    configuration['src_sparse_feat_map'] = FileBasedFeatMap(
                                        configuration['enc_embed'],
                                        configuration['src_sparse_feat_map'])
if configuration['trg_sparse_feat_map']:
    configuration['trg_sparse_feat_map'] = FileBasedFeatMap(
                                        configuration['dec_embed'],
                                        configuration['trg_sparse_feat_map'])
logging.info("Model options:\n{}".format(pprint.pformat(configuration)))
    
# Align
if args.alignment_model == "nam":
    alignments = align_with_nam(configuration, args)
elif args.alignment_model == "nmt":
    alignments = align_with_nmt(configuration, args)
else:
    logging.fatal("Unknown alignment model %s" % args.alignment_model)

# Create output file
output_handlers = []
if args.outputs:
    for name in args.outputs.split(","):
        path = args.output_path
        if '%s' in args.output_path:
            path = args.output_path % name
        if name == "csv":
            output_handlers.append(CSVAlignmentOutputHandler(path))
        elif name == "npy":
            output_handlers.append(NPYAlignmentOutputHandler(path))
        elif name == "align":
            output_handlers.append(TextAlignmentOutputHandler(path))
        else:
            logging.fatal("Unknown output format '%s'" % name)
for h in output_handlers:
    h.write_alignments(alignments)
