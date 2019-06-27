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

"""This module contains strategies to convert a score breakdown to
the total score. This is commonly specified via the
--combination_scheme parameter.

TODO: The breakdown2score interface is not very elegant, and has some
      overlap with the interpolation_strategy implementations.
"""

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder
import numpy as np
import logging


def breakdown2score_sum(working_score, score_breakdown, full=False):
    """Implements the combination scheme 'sum' by always returning
    ``working_score``. 
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores (not used).
        full (bool): If True, reevaluate all time steps. If False,
                     assume that this function has been called in the
                      previous time step (not used).
    
    Returns:
        float. Returns ``working_score``
    """
    return working_score


def breakdown2score_length_norm(working_score, score_breakdown, full=False):
    """Implements the combination scheme 'length_norm' by normalizing
    the sum of the predictor scores by the length of the current 
    sequence (i.e. the length of ``score_breakdown``). 
    TODO could make more efficient use of ``working_score``
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``. Not used.
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
        full (bool): If True, reevaluate all time steps. If False,
                     assume that this function has been called in the
                      previous time step (not used).
    
    Returns:
        float. Returns a length normalized ``working_score``
    """
    score = sum([Decoder.combi_arithmetic_unnormalized(s) 
                        for s in score_breakdown])
    return score / len(score_breakdown)


def breakdown2score_bayesian(working_score, score_breakdown, full=False, prev_score=None):
    """This realizes score combination following the Bayesian LM 
    interpolation scheme from (Allauzen and Riley, 2011)
    
      Bayesian Language Model Interpolation for Mobile Speech Input
    
    By setting K=T we define the predictor weights according the score
    the predictors give to the current partial hypothesis. The initial
    predictor weights are used as priors. 
    TODO could make more efficient use of ``working_score``
    
    Args:
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``. Not used.
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
        full (bool): If True, reevaluate all time steps. If False,
                     assume that this function has been called in the
                      previous time step.
    
    Returns:
        float. Bayesian interpolated predictor scores
    """
    if not score_breakdown or working_score == utils.NEG_INF:
        return working_score
    alphas = [np.log(w) for (_, w) in score_breakdown[0]]
    if full:
        acc = []
        for pos in score_breakdown: # for each position in the hypothesis
            for k, (p, _) in enumerate(pos): 
                alphas[k] += p
            alpha_part = utils.log_sum(alphas)
            scores = [alphas[k] - alpha_part + p 
                      for k, (p, _) in enumerate(pos)]
            acc.append(utils.log_sum(scores)) 
        return sum(acc)
    else: 
        if len(score_breakdown) == 1:
            scores = [np.log(w) + p for p, w in score_breakdown[0]]
            return utils.log_sum(scores)
        working_score = prev_score
        for k, (p, w) in enumerate(score_breakdown[-2]):
            alphas[k] = np.log(w) + p
        alpha_norm = alphas - utils.log_sum(alphas)
        scores = [alpha_norm[k] + p 
                for k, (p, w) in enumerate(score_breakdown[-1])]
        updated_breakdown = [(p, np.exp(alpha_norm[k]))
                for k, (p, w) in enumerate(score_breakdown[-1])]
        score_breakdown[-1] = updated_breakdown
        working_score += utils.log_sum(scores)
        return working_score


def breakdown2score_bayesian_state_dependent(working_score, score_breakdown, 
                                             full=False, prev_score=None,
                                             lambdas=None):
    """This realizes score combination following the Bayesian LM 
    interpolation scheme from (Allauzen and Riley, 2011)
    
      Bayesian Language Model Interpolation for Mobile Speech Input
    
    By setting K=T we define the predictor weights according the score
    the predictors give to the current partial hypothesis. The initial
    predictor weights are used as priors .

    Unlike breakdown2score_bayesian, define state-independent weights
    which affect how much state-dependent mixture weights (alphas) are
    affected by scores from the other model.

    Makes more efficient use of working_score and calculated priors
    when used incrementally.
    Args:                                                           
        working_score (float): Working combined score, which is the 
                               weighted sum of the scores in
                               ``score_breakdown``. Not used.
        score_breakdown (list): Breakdown of the combined score into
                                predictor scores
        full (bool): If True, reevaluate all time steps. If False,
                     assume that this function has been called in the
                      previous time step.
        prev_score: score of hypothesis without final step
        lambdas: np array of domain-task weights
    
    Returns:
        float. Bayesian interpolated predictor scores
    """
    if not score_breakdown or working_score == utils.NEG_INF:
        return working_score
    if full:
        acc = []
        alphas = [np.log(w) for (_, w) in score_breakdown[0]]
        for pos in score_breakdown: # for each position in the hypothesis
            for k, (p_k, _) in enumerate(pos):
                alphas[k] += p_k
            alpha_prob = np.exp(alphas - utils.log_sum(alphas))
            alpha_prob_lambdas = np.zeros_like(alpha_prob)
            for k in range(len(alpha_prob)):
                for t in range(len(alpha_prob)):
                    alpha_prob_lambdas[k] += alpha_prob[t] * lambdas[k, t]
            scores = [np.log(alpha_prob_lambdas[k]) + p
                      for k, (p, _) in enumerate(pos)]
            acc.append(utils.log_sum(scores))
        return sum(acc)
    else: 
        if len(score_breakdown) == 1:
            scores = [np.log(w) + p for p, w in score_breakdown[0]]
            return utils.log_sum(scores)
        working_score = prev_score
        alphas = [np.log(w) for (_, w) in score_breakdown[-2]]
        for k, (p_k, _) in enumerate(score_breakdown[-2]):
            alphas[k] += p_k 
        alpha_prob = np.exp(alphas - utils.log_sum(alphas)) 
        alpha_prob_lambdas = np.zeros_like(alpha_prob)
        for k in range(len(alpha_prob)):
            for t in range(len(alpha_prob)):
                alpha_prob_lambdas[k] += alpha_prob[t] * lambdas[k, t]
        scores = [np.log(alpha_prob_lambdas[k]) + p
                  for k, (p, _) in enumerate(score_breakdown[-1])]
        updated_breakdown = [(p, alpha_prob[k])
                             for k, (p, _) in enumerate(score_breakdown[-1])]
        score_breakdown[-1] = updated_breakdown
        working_score += utils.log_sum(scores)
        return working_score


def breakdown2score_bayesian_loglin(working_score, score_breakdown, full=False,
                                    prev_score=None):
    """Like bayesian combination scheme, but uses loglinear model
    combination rather than linear interpolation weights
   
    TODO: Implement incremental version of it, write weights into breakdowns.
    """
    if not score_breakdown:
        return working_score
    acc = []
    prev_alphas = [] # list of all alpha_i,k
    # Write priors to alphas
    for (p, w) in score_breakdown[0]:
        prev_alphas.append(np.log(w)) 
    for pos in score_breakdown: # for each position in the hypothesis
        alphas = []
        sub_acc = []
        # for each predictor (p: p_k(w_i|h_i), w: prior p(k))
        for k, (p, w) in enumerate(pos): 
            alpha = prev_alphas[k] + p
            alphas.append(alpha)
            sub_acc.append(p + alpha)
        acc.append(utils.log_sum(sub_acc) - utils.log_sum(alphas))
        prev_alphas = alphas
    return sum(acc)



