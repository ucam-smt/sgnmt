"""This is a highly optimized version of single NMT beam search
decoding for blocks. It runs completely separated from the rest of
SGNMT and does not use the predictor frameworks or the ``Decoder``
search strategy abstraction. The central concepts of this decoder can
be summarized as follows:

  * Use a CPU scheduler to construct large batches of computation for
    the GPU
  * Run encoder first for all sentences, and store the result as shared
    variable in the GPU memory to minimize CPU/GPU communication costs
  * Use buckets to cluster sentences with similar lengths which can be
    grouped in a single batch for the GPU
"""

import logging
import time
import pprint
import threading
import os
import theano
import theano.tensor as T

from blocks.roles import INPUT, OUTPUT
from theano import function
from blocks.filter import VariableFilter

from cam.sgnmt.blocks.model import NMTModel, LoadNMTUtils
from cam.sgnmt.blocks.nmt import blocks_get_default_nmt_config, \
                                 get_nmt_model_path_best_bleu
from cam.sgnmt.ui import get_blocks_batch_decode_parser
from cam.sgnmt import utils
from blocks.search import BeamSearch
import Queue
import numpy as np
from collections import OrderedDict
from blocks.utils import shared_floatx_zeros
from picklable_itertools.extras import equizip

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)

state_names = ['outputs', 'states', 'weights', 'weighted_averages']

parser = get_blocks_batch_decode_parser()
args = parser.parse_args()

PARAM_ENC_MAX_WORDS = args.enc_max_words
"""Global variable storing ``--enc_max_words``. """

PARAM_MIN_JOBS = args.min_jobs
"""Global variable storing ``--min_jobs``. """

PARAM_MAX_TASKS_PER_JOB = args.max_tasks_per_job
"""Global variable storing ``--max_tasks_per_job``. """

PARAM_MAX_TASKS_PER_STATE_UPDATE_JOB = args.max_tasks_per_state_update_job
"""Global variable storing ``--max_tasks_per_state_update_job``. """

PARAM_MAX_DEPTH_PER_JOB = args.beam
"""Global variable storing ``--beam``. This is used to determine the 
maximum depth of a single task in a job.
"""

PARAM_MAX_ROWS_PER_JOB = args.max_rows_per_job
"""Global variable storing ``--enc_max_rows_per_job``. """

PARAM_BEAM_SIZE = args.beam
"""Global variable storing ``--beam``. """

PARAM_MIN_TASKS_PER_BUCKET = args.min_tasks_per_bucket
"""Global variable storing ``--min_tasks_per_bucket``. """

PARAM_MIN_BUCKET_TOLERANCE = args.min_bucket_tolerance
"""Global variable storing ``--min_bucket_tolerance``. """


def load_sentences(path, _range, src_vocab_size):
    """Loads the source sentences to decode from the file system.
    
    Args:
        path (string): path to the plain text file with indexed
                       source sentences
        _range (string): Range argument
        src_vocab_size (int): Source language vocabulary size
    
    Returns:
        list. List of tuples, the first element is the sentence ID and
        the second element is a list of integers representing the
        sentence ending with EOS.
    """
    seqs = []
    seq_id = 1
    with open(path) as f:
        for line in f:
            seq = [int(w) for w in line.strip().split()]
            seqs.append((
                    seq_id,
                    utils.oov_to_unk(seq, src_vocab_size) + [utils.EOS_ID]))
            seq_id += 1
    if _range:
        try:
            if ":" in args.range:
                from_idx,to_idx = args.range.split(":")
            else:
                from_idx = int(args.range)
                to_idx = from_idx
            return seqs[int(from_idx)-1:int(to_idx)]
        except Exception as e:
            logging.fatal("Invalid value for --range: %s" % e)
    return seqs


def compute_encoder(tasks):
    """Computes the contexts and initial states for the given sentences.

    Returns:
        tuple. A tuple of numpy arrays with the source annotations 
        (attended) and the initial decoder states
    """
    sen = np.array([t.src_sentence for t in tasks], dtype=int)
    logging.debug("Run encoder on shape %s" % (sen.shape,))
    contexts, states, _ = beam_search.compute_initial_states_and_contexts(
                                        {nmt_model.sampling_input: sen})
    return contexts['attended'], states


class DecodingTask(object):
    """A decoding task represents the state of decoding for a single
    sentence.
    """

    def __init__(self, sen_id, src_sentence):
        """Creates a new decoding task. This task needs to be
        initialized before with the initial states before it can be
        put into the queues.
        
        Args:
            sen_id (int): Sentence ID
            src_sentence (list): Source sentence to decode
        """
        self.sen_id = sen_id
        self.src_sentence = src_sentence
        self.index = -1
        self.bucket = None

    def initialize(self, initial_states):
        """Initializes this task by initializing the current stack
        with a single empty hypothesis with ``initial_states``.
        
        Args:
            initial_states (dict): Initial decoder states for this
                                   task
        """
        self.states = [initial_states]
        self.outputs = [[-1]]
        self.continuations= [] # List of next possible continuations
        self.costs = [0.0]
        self.cur_depth = 0
        self.needs_state_update = False
        self.n_expansions = 0

    def get_source_length(self):
        """Returns the length of the source sentence. """
        return len(self.src_sentence)

    def switch_to_continuations(self):
        """This is called when the current stack is expanded
        sufficiently. Updates the internal states, costs, and
        outputs variables and triggers a state update on this task.
        """
        self.n_expansions += len(self.continuations)
        self.states = [c.states for c in self.continuations]
        self.costs = [c.cost for c in self.continuations]
        self.outputs = [c.outputs for c in self.continuations]
        self.continuations = []
        self.cur_depth = 0
        self.needs_state_update = True

    def is_finished(self):
        """Returns true if the best hypothesis ends with EOS or the 
        maximum target sentence length is exceeded.
        """
        return self.outputs[0][-1] == utils.EOS_ID \
               or len(self.outputs[0]) > 2*self.get_source_length()

    def get_best_translation(self):
        """Get the best (partial) translation. """
        return self.outputs[0][1:-1]

    def get_stats_string(self):
        """Returns a string with information about this task. """
        return "score=%f num_expansions=%d" % (self.costs[0], self.n_expansions)

    def __hash__(self):
        return self.sen_id

    def __eq__(self, other):
        return self.sen_id == other.sen_id


class Continuation(object):
    """A continuation lives inside a decoding task and represents a
    possible candidate for the next time step.
    """

    def __init__(self, states, outputs, cost):
        """Create a new hypothesis continuation.
        
        Args:
            states (dict): decoder states before consuming the last 
                           word in ``outputs``
            outputs (list): Complete translation associated with this
                            continuation.
            cost (float): Total cost of this (partial) translation.
        """
        self.states = states
        self.outputs = outputs
        self.cost = cost

    def __cmp__(self, other):
        return cmp(self.cost, other.cost)


class ComputationJob(object):
    """This represents a single batch which can be send to the GPU for 
    computation. It consists of multiple ``DecodingTasks``. Depending
    on the queue this job is scheduled, it represents either a logprobs
    job or a state update job.
    """
    
    def __init__(self, bucket, tasks, src_indices, states, outputs = None):
        """Sole constructor.
        
        Args:
            bucket (Bucket): A reference to the bucket which all 
                             tasks of this job are in. Used to access
                             the theano function.
            tasks (list): List of tasks which are computed with this
                          batch
            src_indices (OrderedDict): passed through to Theano function
            states (OrderedDict): passed through to Theano function
            outputs (OrderedDict): passed through to Theano function
        """
        self.bucket = bucket
        self.tasks = tasks
        self.src_indices = src_indices
        self.states = states 
        self.outputs = outputs
        self.result = None
                                   

class Pipeline(object):
    """Global place to reference all the queues. This includes the
    following special purpose queues:
    
    * unscheduled_tasks: Contains tasks whichh have not been assigned
                         to a job
    * logprobs_jobs_queues: Logprobs computation jobs, one queue for
                            each bucket
    * state_update_jobs_queues: State update computation jobs, one 
                                queue for each bucket
    * logprobs_result_queue: The computation worker puts logprobs jobs
                             in here after processing them
    * state_update_result_queue: The computation worker puts state update
                                 jobs in here after processing them
    * finished_tasks_queue: All finished tasks end up here.
    """
    
    def __init__(self, buckets):
        """Initializes all the queues with empty lists.
        
        Args:
            buckets (list): List of all the buckets
        """
        self.buckets = buckets
        n_buckets = len(buckets)
        self.unscheduled_tasks = Queue.Queue() # Lists of tasks 

        self.logprobs_jobs_queues = [Queue.Queue() for _ in xrange(n_buckets)]
        self.state_update_jobs_queues = [Queue.Queue() for _ in xrange(n_buckets)]
        self.logprobs_result_queue = Queue.Queue() # Jobs
        self.state_update_result_queue = Queue.Queue() # Jobs
        self.finished_tasks_queue = Queue.Queue() # Tasks
        self.is_finished = False
        self.update_bucket_order()

    def update_bucket_order(self):
        """Buckets are processed according their priority. We aim to
        finish all buckets in about the same time so that the CPU 
        scheduler has the flexibility of switching between buckets as
        long as possible. This implementation is thread safe.
        """
        new_bucket_order = [bucket_id for bucket_id in range(len(self.buckets))
                                if not self.buckets[bucket_id].is_finished()]
        for bucket_id in new_bucket_order:
            self.buckets[bucket_id].update_priority()
        new_bucket_order.sort(
                        key=lambda bucket_id: self.buckets[bucket_id].priority)
        self.bucket_order = new_bucket_order


class Bucket(object):
    """A bucket is a set of decoding tasks which correspond to source
    sentences with similar lengths. Batches can be constructed within
    buckets, but not across buckets. Slight differences in source
    sentence lengths are addressed using padding. We create one shared
    variable per bucket which stores all the source annotations of
    the tasks in the bucket in the GPU memory.
    """
    
    def __init__(self, bucket_id):
        """Creates a new bucket.
        
        Args:
            bucket_id (int): Index of this bucket. Low indices should
                             refer to long sentences.
        """
        self.tasks = []
        self.bucket_id = bucket_id
        self.max_size = 0
        self.min_size = 10000
        self.priority = 0.0

    def update_priority(self):
        """Set the priority to the average length of partial hypotheses
        in the bucket divided by the source sentence lengths.
        """
        self.priority = (0.0 + sum([len(t.outputs[0]) for t in self.tasks])) \
                        / len(self.tasks) / self.max_size

    def can_add(self, size):
        """Returns true if the given size does not clash with the 
        minimum bucket tolerance restriction or if this bucket is not
        large enough yet.
        """
        return len(self.tasks) < PARAM_MIN_TASKS_PER_BUCKET \
               or (self.max_size - size <=  PARAM_MIN_BUCKET_TOLERANCE)

    def is_finished(self):
        """Returns true if all tasks in this bucket are finished. """
        return self.n_tasks == self.n_finished

    def count_unfinished(self):
        """Returns the number of unfinished tasks in this bucket. """
        return self.n_tasks - self.n_finished

    def add_task(self, task):
        """Add a new task to the bucket, and update its index and
        bucket reference. Note that we add the task even if ``can_add``
        would return False.
        
        Args:
            task (DecodingTask): task to add to the bucket
        """
        src_len = task.get_source_length()
        self.max_size = max(src_len, self.max_size)
        self.min_size = min(src_len, self.min_size)
        task.bucket = self
        task.index = len(self.tasks)
        self.tasks.append(task)

    def compile(self):
        """Do not add any more tasks after this function is called.
        Compiles the state update and logprobs theano functions for
        this bucket.
        """
        self.n_tasks = len(self.tasks)
        self.n_finished = 0
        self.all_attended = shared_floatx_zeros((1, 1, 1))
        self.all_masks = shared_floatx_zeros((1, 1))
        self.src_indices = T.ivector()
        givens = self._construct_givens()
        self._compile_next_state_computer(givens)
        self._compile_logprobs_computer(givens)

    def compute_context(self):
        """Compute all source annotations for this bucket, and store
        them as shared variable in the GPU memory.
        """
        all_mask_lst = []
        all_attended_lst = []
        start_pos = 0
        cur_len = self.tasks[0].get_source_length()
        cur_n_words = cur_len
        for pos in xrange(1, len(self.tasks)):
            this_len = self.tasks[pos].get_source_length()
            if this_len != cur_len or cur_len + cur_n_words > PARAM_ENC_MAX_WORDS:
                batch_mask_lst, batch_attended_lst = self._compute_context_range(
                                                                    start_pos,
                                                                    pos)
                all_mask_lst.extend(batch_mask_lst)
                all_attended_lst.extend(batch_attended_lst)
                cur_len = this_len
                start_pos = pos
                cur_n_words = 0
            cur_n_words += cur_len
        batch_mask_lst, batch_attended_lst = self._compute_context_range(
                                                               start_pos, 
                                                               len(self.tasks))
        all_mask_lst.extend(batch_mask_lst)
        all_attended_lst.extend(batch_attended_lst)
        all_mask_val = np.stack(all_mask_lst, axis=1) 
        all_attended_val = np.stack(all_attended_lst, axis=1) 
        self.all_masks.set_value(all_mask_val)
        self.all_attended.set_value(all_attended_val)

    def _compute_context_range(self, start_pos, end_pos):
        """Compute source annotations in a specific range. All tasks in
        this range should have the same source sentence length.
        """
        batch_tasks = self.tasks[start_pos:end_pos]
        attendeds,init_states = compute_encoder(batch_tasks)
        n_pad = self.max_size-batch_tasks[0].get_source_length()
        pad_attendeds = np.lib.pad(attendeds, 
                                   ((0, n_pad), 
                                    (0, 0),
                                    (0, 0)),
                                   'constant', 
                                   constant_values=0.0)
        for idx,task in enumerate(batch_tasks):
            task.initialize({n: init_states[n][idx] for n in state_names})
        mask = np.ones((self.max_size,), dtype=theano.config.floatX)
        if n_pad > 0:
            mask[-n_pad:] = 0.0
        return len(batch_tasks) * [mask], np.transpose(pad_attendeds, (1, 0, 2))

    def _construct_givens(self):
        """Constructs the givens dictionary for the theano functions.
        """
        return {beam_search.contexts[0]: self.all_attended[:,self.src_indices,:],
                beam_search.contexts[1]: self.all_masks[:,self.src_indices]}

    def _compile_next_state_computer(self, givens):
        """Modified version of ``BeamSearch._compile_next_state_computer``
        with ``givens``.
        """
        next_states = [VariableFilter(bricks=[beam_search.generator],
                                      name=name,
                                      roles=[OUTPUT])(beam_search.inner_cg)[-1]
                       for name in beam_search.state_names]
        next_outputs = VariableFilter(
            applications=[beam_search.generator.readout.emit], roles=[OUTPUT])(
                beam_search.inner_cg.variables)
        self.next_state_computer = function(
            [self.src_indices] + beam_search.input_states + next_outputs, 
            next_states,
            givens=givens)

    def _compile_logprobs_computer(self, givens):
        """Modified version of ``BeamSearch._compile_logprobs_computer``
        with ``givens``.
        """
        probs = VariableFilter(
            applications=[beam_search.generator.readout.emitter.probs],
            roles=[OUTPUT])(beam_search.inner_cg)[0]
        logprobs = -T.log(probs)
        self.logprobs_computer = function(
            [self.src_indices] + beam_search.input_states, 
            logprobs,
            givens=givens)

    def compute_logprobs(self, src_indices, states):
        """Call theano on a logprobs batch.
        """
        input_states = [states[name] for name in beam_search.input_state_names]
        return self.logprobs_computer(*([src_indices] + input_states))

    def compute_next_states(self, src_indices, states, outputs):
        """Call theano on a state update batch.
        """
        input_states = [states[name] for name in beam_search.input_state_names]
        next_values = self.next_state_computer(*([src_indices] +
                                                 input_states + [outputs]))
        return OrderedDict(equizip(beam_search.state_names, next_values))


def make_states(states_lst, outputs_lst):
    """Creates the states input variables for a decoding job consisting
    of multiple tasks. This involves stacking the corresponding 
    variables from the individual states. Note that theano does not
    require the full states (with weighted_averages and weights) but
    only outputs and states.
    
    Args:
        states_lst (list): List of states, one for each hypothesis
        outputs_lst (list): List of the last outputs, one for each
                            hypothesis
    """
    all_states = OrderedDict()
    all_states['outputs'] = np.array(outputs_lst)
    all_states['states'] = np.stack([s['states'] for s in states_lst])
    return all_states


def create_state_update_job(tasks):
    """Constructs a state update job from the given tasks.
    
    Args:
        tasks (list): List of tasks.
    
    Returns:
        ComputationJob
    """
    src_indices = []
    states = []
    outputs = []
    for task in tasks:
        src_indices.extend([task.index] * len(task.states))
        states.extend(task.states)
        outputs.extend([o[-1] for o in task.outputs])
    return ComputationJob(tasks[0].bucket, 
                          tasks, 
                          src_indices, 
                          make_states(states, outputs), 
                          outputs)


def create_logprobs_job(tasks):
    """Constructs a logprobs job from the given tasks.
    
    Args:
        tasks (list): List of tasks.
    
    Returns:
        ComputationJob
    """
    src_indices = []
    states = []
    outputs = []
    cnt = 0
    for depth in xrange(PARAM_MAX_DEPTH_PER_JOB):
        for task in tasks:
            task_depth = depth + task.cur_depth
            if task_depth < len(task.states):
                src_indices.append(task.index)
                states.append(task.states[task_depth])
                outputs.append(task.outputs[task_depth][-1])
                cnt += 1
                if depth > 0 and cnt > PARAM_MAX_ROWS_PER_JOB:
                    return ComputationJob(tasks[0].bucket, 
                                          tasks, 
                                          src_indices, 
                                          make_states(states, outputs))
    return ComputationJob(tasks[0].bucket, 
                          tasks, 
                          src_indices, 
                          make_states(states, outputs))


# Workers which consume or produce elements from/to queues in the pipeline


def computation_worker_func(pipeline):
    """This worker fetches jobs from the logprobs and state update
    queus and sends them to theano for computation. This worker
    must run on the main thread to work. The computation results are
    stored in the ``result`` attribute of the job, and the job is added
    to the *_result* queues..
    """
    logging.debug("Start computation")
    reported = False
    while not pipeline.is_finished:
        # We hope that usually one of those queues is not empty and 
        # busy waiting is not a big issue
        did_sth = False
        for bucket_id in pipeline.bucket_order:
            if not pipeline.state_update_jobs_queues[bucket_id].empty():
                job = pipeline.state_update_jobs_queues[bucket_id].get()
                job.result = job.bucket.compute_next_states(job.src_indices,
                                                            job.states,
                                                            job.outputs)
                pipeline.state_update_result_queue.put(job)
                did_sth = True
                reported = False
                break
        for bucket_id in pipeline.bucket_order:
            if not pipeline.logprobs_jobs_queues[bucket_id].empty():
                job = pipeline.logprobs_jobs_queues[bucket_id].get()
                job.result = job.bucket.compute_logprobs(job.src_indices,
                                                         job.states)
                pipeline.logprobs_result_queue.put(job)
                did_sth = True
                reported = False
                break
        if reported and not did_sth:
            logging.debug("Computation worker is idle!!!")
            reported = True
                

def task2job_worker_func(pipeline):
    """This worker assigns tasks to jobs. As soon as we have collected
    enough tasks to fill a batch, we construct a job and send it via
    the appropriate queue. If all tasks of a bucket are collected, and
    a full batch cannot be constructed, we build a smaller batch with
    the remaining tasks, either a logprobs or a state update job
    (depending on which is more urgent). If the total number of jobs in
    the computation queues falls below a threshold, we schedule all 
    tasks we have to avoid having an idle computation thread.
    """
    n_buckets = len(pipeline.buckets)
    logprobs_tasks = [[] for _ in xrange(n_buckets)]
    state_update_tasks = [[] for _ in xrange(n_buckets)]
    while True:
        new_tasks = pipeline.unscheduled_tasks.get()
        for task in new_tasks:
            if task.is_finished():
                task.bucket.n_finished += 1
                pipeline.finished_tasks_queue.put(task)
            elif task.needs_state_update:
                state_update_tasks[task.bucket.bucket_id].append(task)
            else:
                logprobs_tasks[task.bucket.bucket_id].append(task)
        for bucket_id in xrange(n_buckets):
            n_unfinished = pipeline.buckets[bucket_id].count_unfinished()
            all_tasks_waiting = len(state_update_tasks[bucket_id]) \
                                + len(logprobs_tasks[bucket_id]) == n_unfinished
            scheduled_full_logprobs = False
            scheduled_full_state_update = False
            while len(logprobs_tasks[bucket_id]) >= PARAM_MAX_TASKS_PER_JOB:
                job = create_logprobs_job(
                            logprobs_tasks[bucket_id][:PARAM_MAX_TASKS_PER_JOB])
                logprobs_tasks[bucket_id] = \
                            logprobs_tasks[bucket_id][PARAM_MAX_TASKS_PER_JOB:]
                pipeline.logprobs_jobs_queues[bucket_id].put(job)
                scheduled_full_logprobs = True
            while len(state_update_tasks[bucket_id]) >= PARAM_MAX_TASKS_PER_STATE_UPDATE_JOB:
                job = create_state_update_job(
                    state_update_tasks[bucket_id][:PARAM_MAX_TASKS_PER_STATE_UPDATE_JOB])
                state_update_tasks[bucket_id] = \
                    state_update_tasks[bucket_id][PARAM_MAX_TASKS_PER_STATE_UPDATE_JOB:]
                pipeline.state_update_jobs_queues[bucket_id].put(job)
                scheduled_full_state_update = True
            if (not all_tasks_waiting) \
                    or scheduled_full_logprobs \
                    or scheduled_full_state_update:
                continue
            if len(logprobs_tasks[bucket_id]) > len(state_update_tasks[bucket_id]):
                pipeline.logprobs_jobs_queues[bucket_id].put(
                                create_logprobs_job(logprobs_tasks[bucket_id]))
                logprobs_tasks[bucket_id] = []
            elif len(state_update_tasks[bucket_id]) > 0:
                pipeline.state_update_jobs_queues[bucket_id].put(
                        create_state_update_job(state_update_tasks[bucket_id]))
                state_update_tasks[bucket_id] = []
        # Schedule all we have if the total number of jobs is below threshold
        n_jobs = sum([q.qsize() for q in 
            pipeline.state_update_jobs_queues + pipeline.logprobs_jobs_queues])
        if n_jobs < PARAM_MIN_JOBS:
            logging.debug("Number of jobs critical: %d" % n_jobs)
            for bucket_id in xrange(n_buckets):
                if len(state_update_tasks[bucket_id]) > 0:
                    pipeline.state_update_jobs_queues[bucket_id].put(
                        create_state_update_job(state_update_tasks[bucket_id]))
                    state_update_tasks[bucket_id] = []
                if len(logprobs_tasks[bucket_id]) > 0:
                    pipeline.logprobs_jobs_queues[bucket_id].put(
                                create_logprobs_job(logprobs_tasks[bucket_id]))
                    logprobs_tasks[bucket_id] = []


def logprobs_worker_func(pipeline):
    """This worker reads out the logprobs_result_queue. If the time
    step for one entry is finished, fill the state_update_queue 
    with the selected hypotheses. Otherwise, add new jobs to the
    logprobs_queue.
    """
    while True:
        job = pipeline.logprobs_result_queue.get()
        words = np.argpartition(job.result, 
                                PARAM_BEAM_SIZE, 
                                axis=1)[:,:PARAM_BEAM_SIZE]
        for pos, task in enumerate(job.tasks):
            prev_cost = task.costs[task.cur_depth]
            prev_state = task.states[task.cur_depth]
            prev_outputs = task.outputs[task.cur_depth]
            task.continuations.extend([Continuation(
                                              prev_state, 
                                              prev_outputs + [word], 
                                              prev_cost + job.result[pos,word]) 
                                       for word in words[pos]])
            task.cur_depth += 1
        unique_tasks = set(job.tasks)
        update_bucket_order = False
        for task in unique_tasks:
            task.continuations.sort()
            task.continuations = task.continuations[:PARAM_BEAM_SIZE]
            # If all hypos of last timestep expanded or worse than beam-size
            # best continuation, trigger state update
            if (task.cur_depth >= len(task.states) or 
                   task.continuations[PARAM_BEAM_SIZE-1].cost < task.costs[
                                                            task.cur_depth]):
                task.switch_to_continuations()
                update_bucket_order = True
        pipeline.unscheduled_tasks.put(unique_tasks)
        if update_bucket_order:
            pipeline.update_bucket_order()


def state_update_worker_func(pipeline):
    """This worker reads out the state_update_result_queue. 
    """
    while True:
        job = pipeline.state_update_result_queue.get()
        pos = 0
        for task in job.tasks:
            n_hypos = len(task.costs)
            src_len = task.get_source_length()
            task.states = [{
               'states': job.result['states'][pos+idx,:],
               'weighted_averages': job.result['weighted_averages'][pos+idx,:],
               'weights': job.result['weights'][pos+idx,:src_len] 
            } for idx in xrange(n_hypos)]
            pos += n_hypos
            task.needs_state_update = False
        pipeline.unscheduled_tasks.put(job.tasks)


def finished_worker_func(pipeline):
    """This worker gathers all the finished tasks. If all tasks are
    finished, output results and terminate SGNMT.
    """
    finished_tasks = []
    n_tasks = sum([b.n_tasks for b in pipeline.buckets])
    for _ in xrange(n_tasks):
        finished_tasks.append(pipeline.finished_tasks_queue.get())
        logging.debug("Finished %d translations" % (len(finished_tasks),))
        logging.debug("Bucket order: %s" % pipeline.bucket_order)
        for bucket in pipeline.buckets:
            logging.debug("Bucket %d: %d/%d (queues: probs=%d update=%d)" % (
                    bucket.bucket_id, 
                    bucket.n_finished, 
                    bucket.n_tasks, 
                    pipeline.logprobs_jobs_queues[bucket.bucket_id].qsize(), 
                    pipeline.state_update_jobs_queues[bucket.bucket_id].qsize()))
    stop_time = time.time()
    pipeline.is_finished = True

    # Print out result
    for task in sorted(finished_tasks, key=lambda t: t.sen_id): 
        logging.info("Decoded (ID: %d): %s" % (
                        task.sen_id,
                        ' '.join([str(w) for w in task.get_best_translation()])))
        logging.info("Stats (ID: %d): %s" % (task.sen_id,
                                             task.get_stats_string()))
    logging.info("Decoding finished. Time: %.6f" % (stop_time - start_time))
    os.system('kill %d' % os.getpid())
        

# MAIN ENTRY POINT

# Get configuration
config = blocks_get_default_nmt_config()
for k in dir(args):
    if k in config:
        config[k] = getattr(args, k)
logging.info("Model options:\n{}".format(pprint.pformat(config)))
np.show_config()

nmt_model = NMTModel(config)
nmt_model.set_up()

loader = LoadNMTUtils(get_nmt_model_path_best_bleu(config),
                      config['saveto'],
                      nmt_model.search_model)
loader.load_weights()

src_sentences = load_sentences(args.src_test,
                               args.range,
                               config['src_vocab_size'])
n_sentences = len(src_sentences)

logging.info("%d source sentences loaded. Initialize decoding.." 
                    % n_sentences)

beam_search = BeamSearch(samples=nmt_model.samples)
beam_search.compile()

logging.info("Sort sentences, longest sentence first...")
src_sentences.sort(key=lambda x: len(x[1]), reverse=True)

# Bucketing
cur_bucket = Bucket(0)
buckets = [cur_bucket]
for sen_id, sen in src_sentences:
    sen_len = len(sen)
    if not cur_bucket.can_add(sen_len):
        cur_bucket.compile()
        cur_bucket = Bucket(len(buckets))
        buckets.append(cur_bucket)
    cur_bucket.add_task(DecodingTask(sen_id, sen))
cur_bucket.compile()

logging.info("Compiled %d buckets" % len(buckets))
for bucket in buckets:
    logging.debug("Bucket %d: %d tasks in [%d,%d]" % (bucket.bucket_id,
                                                      bucket.n_tasks,
                                                      bucket.min_size,
                                                      bucket.max_size))

# Compute contexts and initial states
start_time = time.time()
logging.info("Start time: %s" % start_time)

logging.info("Compute all initial states and contexts...")


all_tasks = []
for bucket in buckets:
    bucket.compute_context()
    all_tasks.extend(bucket.tasks)

pipeline = Pipeline(buckets)
pipeline.unscheduled_tasks.put(all_tasks)

task2job_worker = threading.Thread(target=task2job_worker_func,
                                   args=(pipeline,))
task2job_worker.start()
logprobs_worker = threading.Thread(target=logprobs_worker_func,
                                   args=(pipeline,))
logprobs_worker.start()
logprobs_worker2 = threading.Thread(target=logprobs_worker_func,
                                    args=(pipeline,))
logprobs_worker2.start()
finished_worker = threading.Thread(target=finished_worker_func,
                                   args=(pipeline,))
finished_worker.start()
state_update_worker = threading.Thread(target=state_update_worker_func,
                                       args=(pipeline,))
state_update_worker.start()

# We need to execute the computation worker in the main thread because
# otherwise Theano is confused
computation_worker_func(pipeline)


