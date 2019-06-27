# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains utility functions for TensorFlow such as
session handling and checkpoint loading.
"""

try:
    # This is the TF backend needed for MoE interpolation
    import tensorflow as tf
    from tensorflow.python.training import saver
    from tensorflow.python.training import training
except ImportError:
    pass # Deal with it in decode.py

import os
import logging


def session_config(n_cpu_threads=-1):
    """Creates the session config with default parameters.

    Args:
      n_cpu_threads (int): Number of CPU threads. If negative, we
                           assume either GPU decoding or that all
                           CPU cores can be used.

    Returns:
      A TF session config object.
    """
    graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))
    if n_cpu_threads < 0:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(
            allow_soft_placement=True,
            graph_options=graph_options,
            gpu_options=gpu_options,
            log_device_placement=False)
    else:
        #device_count={'CPU': n_cpu_threads},
        if n_cpu_threads >= 4:
            # This adjustment is an estimate of the effective load which
            # accounts for the sequential parts in SGNMT.
            if n_cpu_threads == 4:
                n_cpu_threads = 5
            else:
                n_cpu_threads = int(n_cpu_threads*5/1.5 - 10)
            logging.debug("Setting TF inter and intra op parallelism "
                          "to %d" % n_cpu_threads)
        config = tf.ConfigProto(
            intra_op_parallelism_threads=n_cpu_threads,
            inter_op_parallelism_threads=n_cpu_threads,
            allow_soft_placement=True,
            graph_options=graph_options,
            log_device_placement=False)
    return config


def create_session(checkpoint_path, n_cpu_threads=-1):
    """Creates a MonitoredSession.
    
    Args:
      checkpoint_path (string): Path either to checkpoint directory or
                                directly to a checkpoint file.
      n_cpu_threads (int): Number of CPU threads. If negative, we
                           assume either GPU decoding or that all
                           CPU cores can be used.
    Returns:
      A TensorFlow MonitoredSession.
    """
    try:
        if os.path.isdir(checkpoint_path):
            checkpoint_path = saver.latest_checkpoint(checkpoint_path)
        else:
            logging.info("%s is not a directory. Interpreting as direct "
                         "path to checkpoint..." % checkpoint_path)
        return training.MonitoredSession(
            session_creator=training.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                config=session_config(n_cpu_threads)))
    except tf.errors.NotFoundError as e:
        logging.fatal("Could not find all variables of the computation "
            "graph in the T2T checkpoint file. This means that the "
            "checkpoint does not correspond to the model specified in "
            "SGNMT. Please double-check pred_src_vocab_size, "
            "pred_trg_vocab_size, and all the t2t_* parameters. "
            "Also make sure that the checkpoint exists and is readable")
        raise AttributeError("Could not initialize TF session.")

