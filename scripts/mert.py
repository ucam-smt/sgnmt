"""Simple Powell search on SGNMT predictor weights."""

import sys, os, time, re
import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
import argparse
import shutil
from operator import itemgetter
from random import shuffle


def get_script_path():
    """ https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory-with-python """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


parser = argparse.ArgumentParser(description='Uses simple grid search for MERT '
                                 'tuning of SGNMT predictor weights.')
parser.add_argument('-d','--work_dir', help='Working directory',
                    required=True)
parser.add_argument('-b','--bleu_cmd', help='BLEU command. Must read indexed sentences from stdin',
                    required=True)
parser.add_argument('-c','--config_file', help='SGNMT configuration file. Must not include predictor_weights or outputs',
                    required=True)
parser.add_argument('-g','--groups', help='Use this to share predictor weights. Comma separated list of ints, same numbers'
                    'for predictors which share weights. Group 0 is not optimized and remains at the initial value.',
                    required=True)
parser.add_argument('-s','--decode_script', help='path to one of the decode_on_grid_cpu*.sh scripts which should be used '
                    'for distributed decoding',
                    default="%s/sge/decode_cpu.sh" % get_script_path(),
                    required=False)
parser.add_argument('-w','--initial_weights', help='Starting point (Default: 0.5,0.5,0.5,...)',
                    default='',
                    required=False)
parser.add_argument('-m','--memory', help='Memory requirements',
                    default='1.5G',
                    required=False)
parser.add_argument('-i','--iter', help='Number of iteration spent to optimize a single dimension',
                    default=2, type=int,
                    required=False)
parser.add_argument('-e','--epochs', help='Number of swipes through the dimensions',
                    default=100, type=int,
                    required=False)
parser.add_argument('-j','--max_jump', help='If set, lower and upper bounds are not more than this away from the current best point.',
                    default=-1.0, type=float,
                    required=False)
args = parser.parse_args()


def objective(point):
    """Definition of the objective function for hyperopt """
    params = {n: point[i] for i,n in enumerate(group_names)}
    pred_weights = []
    for g_idx, g in enumerate(groups):
        if g > 0:
            w = params['group%d' % g]
        elif g < 0: 
            w = 1.0 - params['group%d' % (-g)]
        else: # g == 0
            w = initial_weights[g_idx]
        pred_weights.append(str(w))
    pred_weights = ','.join(pred_weights)
    logging.info("Evaluate data point: %s (predictor weights: %s)" % (point, pred_weights))
    sgnmt_dir = "%s/sgnmt" % args.work_dir
    config_file = "%s/config.ini" % sgnmt_dir
    done_file = "%s/DONE" % sgnmt_dir
    try:
        shutil.rmtree(sgnmt_dir)
    except OSError:
        pass
    os.mkdir(sgnmt_dir)
    shutil.copyfile(args.config_file, config_file)
    with open(config_file, "a") as f:
        f.write("outputs: text,nbest\n")
        f.write("predictor_weights: %s\n" % pred_weights)
    cmd = "%s %s %s %s" % (args.decode_script, config_file, sgnmt_dir, args.memory)
    logging.info("Start decoding: %s" % cmd)
    os.system(cmd)
    logging.info("Waiting for %s..." % done_file)
    while not os.path.exists(done_file):
        time.sleep(10)
    eval_cmd = "cat %s/out.text | %s > %s/bleu" % (sgnmt_dir, args.bleu_cmd, sgnmt_dir)
    logging.info("Start evaluation: %s" % eval_cmd)
    os.system(eval_cmd)
    bleu_score = 0.0
    with open("%s/bleu" % sgnmt_dir) as f:
        for line in f:
            logging.info("bleu script stdout: %s" % line.strip())
            if bleu_score <= 0.0:
                match = re.match( r'^BLEU = ([.0-9]+),', line)
                if match:
                    bleu_score = float(match.group(1))
    logging.info("pred_weights: %s bleu_score: %f" % (pred_weights, bleu_score))
    return -bleu_score

n_predictors = -1
with open(args.config_file) as f:
    for line in f:
        parts = line.split(':', 1)
        if parts[0].strip() == 'predictors':
            n_predictors = len(parts[1].split(','))
            break
if n_predictors < 0:
    logging.fatal("Could not find 'predictors' key in config file")
    sys.exit()

groups = [int(i) for i in args.groups.split(',')] if args.groups else range(n_predictors)

if len(groups) != n_predictors:
    logging.fatal("%d predictors found, but %d group entries!" % (n_predictors, len(groups)))
    sys.exit()

group_names = ['group%d' % i for i in set([abs(g) for g in groups]) if i != 0]
try:
    os.mkdir(args.work_dir)
except:
    logging.info("Working directory already exists...")

# minimize the objective over the space of group interpolations
n_groups = len(group_names)
cur_point = [0.5] * n_groups
if args.initial_weights:
    initial_weights = map(float, args.initial_weights.split(','))
    for g_idx, g in enumerate(groups):
        if g > 0:
            cur_point[group_names.index('group%d' % g)] = initial_weights[g_idx]
logging.info("Starting point: %s" % map(str, cur_point))
allow_greater_1 = all(g >= 0 for g in groups) # Allow weights greater 1 if we don't have negative groups
if allow_greater_1:
    logging.info("Allowing weights > 1")
else:
    logging.info("Searching for weights in [0,1]")

last_dir = [0] * n_groups
vals = [(cur_point, objective(cur_point))]

for _ in xrange(args.epochs):
    for dim in xrange(n_groups):
        logging.info("Optimizing %s" % group_names[dim])
        for _ in xrange(args.iter):
            best_point, best_val = vals[0]
            best_pos = best_point[dim]
            lower_pos = 0.0
            upper_pos = 1.0
            if allow_greater_1:
                upper_pos += best_pos
            for point, val in vals: # Find values close to best_pos
                pos = point[dim]
                slack = sum([abs(p1-p2) for (p1, p2) in zip(best_point, point)]) - abs(pos-best_pos)
                if pos < best_pos and pos - slack > lower_pos:
                    lower_pos = pos - slack
                if pos > best_pos and pos + slack < upper_pos:
                    upper_pos = pos + slack
            if args.max_jump > 0.0:
                lower_pos = max(lower_pos, best_pos - args.max_jump)
                upper_pos = min(upper_pos, best_pos + args.max_jump)
            test_pos = [lower_pos, upper_pos]
            order = [0, 1]
            if last_dir[dim] > 0:
                order = [1, 0] # Check upper first
            elif last_dir[dim] == 0:
                shuffle(order)
            last_dir[dim] = 0
            for idx in order:
                next_pos = (best_pos + test_pos[idx]) / 2.0
                next_point = list(best_point) 
                next_point[dim] = next_pos
                next_val = objective(next_point)
                vals.append((next_point, next_val))
                if next_val < best_val:
                    last_dir[dim] = -1 if idx == 0 else 1
                    break
            vals.sort(key=itemgetter(1))
        
print(vals[0])
