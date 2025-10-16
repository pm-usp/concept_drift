# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import datetime
from scipy.stats import entropy
from collections import defaultdict
import networkx as nx
from itertools import combinations


def get_feature_alpha_relations(TM_df_original, control_flow_feature=None,
                                max_loop_len: int = 4):
    """
    Derives process mining relations from a transition matrix, with a refined
    definition for the choice relation based on XOR gateway detection and 2-4 node cycles for loops.

    Returns: DataFrame with columns
             ['direct_succession', 'causality', 'parallel', 'choice', 'loop']
    """
    TM_df = TM_df_original.copy().reset_index()

    # Initial Data Preparation
    all_transitions = set(zip(TM_df['activity_from'], TM_df['activity_to']))

    # Map each activity to its direct successors
    out_map = TM_df.groupby('activity_from')['activity_to'].apply(set).to_dict()

    # Create a map of predecessors to check for common split points
    in_map = TM_df.groupby('activity_to')['activity_from'].apply(set).to_dict()

    # ------------------------------------------------------------------#
    # -------- CALCULATE RELATIONS BASED ON DEFINITIONS ----------------#
    # ------------------------------------------------------------------#

    # Causality (a -> b): 'b' is the ONLY successor of 'a'.
    causality = set()
    for a, outs in out_map.items():
        if len(outs) == 1:
            b = next(iter(outs))
            if a != b:
                causality.add((a, b))


    # Parallel (a || b): A pair (a,b) and (b,a) that are NOT causal and share at least one common predecessor.
    parallel = set()
    potential_pairs = {(a, b) for a, b in all_transitions if a < b and (b, a) in all_transitions}
    for a, b in potential_pairs:
        is_causal = (a, b) in causality or (b, a) in causality
        if not is_causal:
            # Check if the activities share any common predecessors (a split point)
            preds_a = in_map.get(a, set())
            preds_b = in_map.get(b, set())
            if preds_a.intersection(preds_b):
                parallel.add((a, b))
                parallel.add((b, a))


    # Choice: Any transition from an activity with multiple successors.
    choice = set()
    # Find all nodes that are split points (have more than one successor).
    split_gateways = {a for a, outs in out_map.items() if len(outs) > 1}
    # Add all outgoing transitions from these gateways to the choice set.
    for gateway in split_gateways:
        for successor in out_map.get(gateway, set()):
            choice.add((gateway, successor))


    # Loops (Self-loops and Cycles)
    self_loop = {(a, b) for a, b in all_transitions if a == b}
    G = nx.DiGraph(all_transitions)
    cycle_edges = set()
    for cycle in nx.simple_cycles(G):
        if 2 <= len(cycle) <= max_loop_len:
            # Add all cycle edges
            cycle_edges.update(
                (cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))
            )
            # Add back edge for the cycle only
            # back_edge = (cycle[-1], cycle[0])
            # cycle_edges.add(back_edge)
    all_loops = cycle_edges | self_loop

    # ------------------------------------------------------------------#
    # -------- Expand 'parallel' to touch split/join---------------- #
    # ------------------------------------------------------------------#
    # Activities that participate in a base parallel pair:
    base_parallel_nodes = {u for (u, v) in parallel} | {v for (u, v) in parallel}

    # Mark any edge that touches a base parallel node (captures split/join edges).
    parallel_edges = {e for e in all_transitions if e[0] in base_parallel_nodes or e[1] in base_parallel_nodes}

    # Unified parallel indicator (include edges that touch base parallel nodes)
    # parallel |= parallel_edges

    # ------------------------------------------------------------------#
    # -------- RESOLVE FINAL OVERLAPS ----------------------------------#
    # ------------------------------------------------------------------#
    # Choice is the most restrictive relation, so remove any overlaps first
    choice -= causality | parallel
    # 
    parallel_edges -= causality
    # Remove any loops that are part of parallel structures
    all_loops -= parallel | parallel_edges
    

    # ------------------------------------------------------------------#
    # -------- WRITE BACK TO DATAFRAME ---------------------------------#
    # ------------------------------------------------------------------#
    idx = list(zip(TM_df['activity_from'], TM_df['activity_to']))

    TM_df['causality'] = np.fromiter((e in causality for e in idx), dtype=int)
    TM_df['parallel'] = np.fromiter((e in parallel or e in parallel_edges for e in idx), dtype=int)
    TM_df['choice'] = np.fromiter((e in choice for e in idx), dtype=int)
    TM_df['loop'] = np.fromiter((e in all_loops for e in idx), dtype=int)

    return TM_df.set_index(['activity_from', 'activity_to'])[['causality', 'parallel', 'choice', 'loop']]




def get_feature_probability(TM_df_original, control_flow_feature):
    """
    Computes the probability of transitions based on their frequency.

    Args:
        TM_df_original (DataFrame): Transition matrix.
        control_flow_feature (str): Name of the control flow feature to compute.

    Returns:
        DataFrame: Transition matrix with the computed probability feature.
    """

    TM_df = TM_df_original.copy()
    TM_df = TM_df.reset_index()

    # Groupby activity_from and get sum
    TM_df_groupby = TM_df.groupby(["activity_from"], as_index=False)["frequency"].sum().set_index("activity_from")

    # Divide TM frequency column with groupby activity_from's sum
    TM_df[control_flow_feature] = TM_df.set_index("activity_from")[["frequency"]].div(TM_df_groupby).reset_index()["frequency"]

    return TM_df.set_index(['activity_from','activity_to'])[[control_flow_feature]]


def get_feature_time_avg(TM_df_original, log_transition_original, method, time_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = method + '_' + time_feature

    log_transition[process_feature_name] = (log_transition[time_feature + '_to'] - log_transition[time_feature + '_from']).dt.total_seconds() / 60
    
    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[process_feature_name].mean()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[process_feature_name]]


def get_feature_time_std(TM_df_original, log_transition_original, method, time_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = method + '_' + time_feature

    log_transition[process_feature_name] = (log_transition[time_feature + '_to'] - log_transition[time_feature + '_from']).dt.total_seconds() / 60
    
    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[process_feature_name].std()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[process_feature_name]]

def get_feature_numerical_avg(TM_df_original, log_transition_original, method, numerical_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = method + '_' + numerical_feature

    log_transition[process_feature_name] = (log_transition[numerical_feature + '_to'] + log_transition[numerical_feature + '_from'])/2
    
    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[process_feature_name].mean()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[process_feature_name]]


def get_feature_numerical_std(TM_df_original, log_transition_original, method, numerical_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = method + '_' + numerical_feature

    log_transition[process_feature_name] = (log_transition[numerical_feature + '_to'] + log_transition[numerical_feature + '_from'])/2
    
    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[process_feature_name].std()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[process_feature_name]]


def get_feature_categorical_unique(TM_df_original, log_transition_original, method, categorical_feature):
    """
    Compute unique (count distinct) values for a categorical feature and aggregate them into the TM_df.
    """
    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = f"{method}_{categorical_feature}"

    # Concatenate "_from" and "_to" for the categorical feature
    log_transition[categorical_feature] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

    # Count distinct values for the concatenated categorical feature
    feature_values = log_transition.groupby(['activity_from', 'activity_to'])[categorical_feature].nunique()

    # Reset index to prepare for joining
    feature_values = feature_values.reset_index().rename(columns={categorical_feature: process_feature_name})

    # Merge the new feature into TM_df
    TM_df = TM_df.reset_index()
    TM_df = TM_df.merge(feature_values, on=['activity_from', 'activity_to'], how='left')

    # Set the original index back
    return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]


def get_feature_categorical_encoding_probability(TM_df_original, log_transition_original, method, categorical_feature):
    """
    Compute frequencies of all unique `Role_from -> Role_to` pairs and add them as columns in TM_df.
    """
    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()

    # Create a concatenated column for Role_from -> Role_to
    role_pair = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']
    log_transition['role_pair'] = role_pair

    # Compute the frequency of each pair for each activity_from -> activity_to
    pair_frequencies = (
        log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
        .value_counts(normalize=True)
        .unstack(fill_value=0)  # Create columns for each pair
    )

    # Rename columns to include the desired method and pattern
    pair_frequencies.columns = [f"{method}_{categorical_feature}_{col}" for col in pair_frequencies.columns]

    # Reset index to prepare for merging
    pair_frequencies = pair_frequencies.reset_index()

    # Merge the new columns into TM_df
    TM_df = TM_df.reset_index()
    TM_df = TM_df.merge(pair_frequencies, on=['activity_from', 'activity_to'], how='left')

    # Fill NaN values for pairs that don't exist in a specific activity transition
    TM_df = TM_df.fillna(0)

    # Set the original index back
    return TM_df.set_index(['activity_from', 'activity_to'])[pair_frequencies.columns.drop(['activity_from', 'activity_to'])]


def get_feature_categorical_encoding_frequency(TM_df_original, log_transition_original, method, categorical_feature):
    """
    Compute frequencies of all unique `Role_from -> Role_to` pairs and add them as columns in TM_df.
    """
    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()

    # Create a concatenated column for Role_from -> Role_to
    role_pair = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']
    log_transition['role_pair'] = role_pair

    # Compute the frequency of each pair for each activity_from -> activity_to
    pair_frequencies = (
        log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
        .value_counts(normalize=False)
        .unstack(fill_value=0)  # Create columns for each pair
    )

    # Rename columns to include the desired method and pattern
    pair_frequencies.columns = [f"{method}_{categorical_feature}_{col}" for col in pair_frequencies.columns]

    # Reset index to prepare for merging
    pair_frequencies = pair_frequencies.reset_index()

    # Merge the new columns into TM_df
    TM_df = TM_df.reset_index()
    TM_df = TM_df.merge(pair_frequencies, on=['activity_from', 'activity_to'], how='left')

    # Fill NaN values for pairs that don't exist in a specific activity transition
    TM_df = TM_df.fillna(0)

    # Set the original index back
    return TM_df.set_index(['activity_from', 'activity_to'])[pair_frequencies.columns.drop(['activity_from', 'activity_to'])]


def get_feature_categorical_entropy(TM_df_original, log_transition_original, method, categorical_feature):
    """
    Compute entropy of categorical `Role_from -> Role_to` transitions and add it to TM_df.
    """
    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()
    process_feature_name = f"{method}_{categorical_feature}"

    # Create concatenated Role_from -> Role_to
    log_transition['role_pair'] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

    # Compute normalized frequencies of role pairs
    pair_frequencies = (
        log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
        .value_counts(normalize=True)
        .unstack(fill_value=0)  # Create columns for each pair
    )

    # Calculate entropy for each transition
    entropy_values = pair_frequencies.apply(lambda x: entropy(x, base=2), axis=1)

    # Reset index and prepare for merging
    entropy_values = entropy_values.reset_index()
    entropy_values.columns = ['activity_from', 'activity_to', process_feature_name]

    # Merge entropy values into TM_df
    TM_df = TM_df.reset_index()
    TM_df = TM_df.merge(entropy_values, on=['activity_from', 'activity_to'], how='left')

    # Set the original index back
    return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]


# def get_feature_categorical_gini(TM_df_original, log_transition_original, method, categorical_feature):
#     """
#     Compute Gini Impurity for categorical `Role_from -> Role_to` transitions.
#     """
#     TM_df = TM_df_original.copy()
#     log_transition = log_transition_original.copy()
#     process_feature_name = f"{method}_{categorical_feature}"

#     # Create concatenated Role_from -> Role_to
#     log_transition['role_pair'] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

#     # Compute normalized frequencies of role pairs
#     pair_frequencies = (
#         log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
#         .value_counts(normalize=True)
#         .unstack(fill_value=0)
#     )

#     # Calculate Gini impurity
#     gini_values = 1 - (pair_frequencies ** 2).sum(axis=1)

#     # Reset index and prepare for merging
#     gini_values = gini_values.reset_index()
#     gini_values.columns = ['activity_from', 'activity_to', process_feature_name]

#     # Merge Gini values into TM_df
#     TM_df = TM_df.reset_index()
#     TM_df = TM_df.merge(gini_values, on=['activity_from', 'activity_to'], how='left')

#     # Set the original index back
#     return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]


# def get_feature_categorical_effective_number(TM_df_original, log_transition_original, method, categorical_feature):
#     """
#     Compute Effective Number of Categories for categorical `Role_from -> Role_to` transitions.
#     """
#     TM_df = TM_df_original.copy()
#     log_transition = log_transition_original.copy()
#     process_feature_name = f"{method}_{categorical_feature}"

#     # Create concatenated Role_from -> Role_to
#     log_transition['role_pair'] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

#     # Compute normalized frequencies of role pairs
#     pair_frequencies = (
#         log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
#         .value_counts(normalize=True)
#         .unstack(fill_value=0)
#     )

#     # Calculate Entropy and then Effective Number of Categories
#     entropy_values = pair_frequencies.apply(lambda x: entropy(x, base=2), axis=1)
#     effective_categories = 2 ** entropy_values

#     # Reset index and prepare for merging
#     effective_categories = effective_categories.reset_index()
#     effective_categories.columns = ['activity_from', 'activity_to', process_feature_name]

#     # Merge Effective Categories values into TM_df
#     TM_df = TM_df.reset_index()
#     TM_df = TM_df.merge(effective_categories, on=['activity_from', 'activity_to'], how='left')

#     # Set the original index back
#     return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]


# def get_feature_categorical_dominance(TM_df_original, log_transition_original, method, categorical_feature):
#     """
#     Compute Dominance Index for categorical `Role_from -> Role_to` transitions.
#     """
#     TM_df = TM_df_original.copy()
#     log_transition = log_transition_original.copy()
#     process_feature_name = f"{method}_{categorical_feature}"

#     # Create concatenated Role_from -> Role_to
#     log_transition['role_pair'] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

#     # Compute normalized frequencies of role pairs
#     pair_frequencies = (
#         log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
#         .value_counts(normalize=True)
#         .unstack(fill_value=0)
#     )

#     # Calculate Dominance Index
#     dominance_values = pair_frequencies.max(axis=1)

#     # Reset index and prepare for merging
#     dominance_values = dominance_values.reset_index()
#     dominance_values.columns = ['activity_from', 'activity_to', process_feature_name]

#     # Merge Dominance values into TM_df
#     TM_df = TM_df.reset_index()
#     TM_df = TM_df.merge(dominance_values, on=['activity_from', 'activity_to'], how='left')

#     # Set the original index back
#     return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]


# def get_feature_categorical_simpsons(TM_df_original, log_transition_original, method, categorical_feature):
#     """
#     Compute Simpson's Diversity Index for categorical `Role_from -> Role_to` transitions.
#     """
#     TM_df = TM_df_original.copy()
#     log_transition = log_transition_original.copy()
#     process_feature_name = f"{method}_{categorical_feature}"

#     # Create concatenated Role_from -> Role_to
#     log_transition['role_pair'] = log_transition[categorical_feature + '_from'] + "->" + log_transition[categorical_feature + '_to']

#     # Compute normalized frequencies of role pairs
#     pair_frequencies = (
#         log_transition.groupby(['activity_from', 'activity_to'])['role_pair']
#         .value_counts(normalize=True)
#         .unstack(fill_value=0)
#     )

#     # Calculate Simpson's Diversity Index
#     simpsons_values = 1 - (pair_frequencies ** 2).sum(axis=1)

#     # Reset index and prepare for merging
#     simpsons_values = simpsons_values.reset_index()
#     simpsons_values.columns = ['activity_from', 'activity_to', process_feature_name]

#     # Merge Simpson's values into TM_df
#     TM_df = TM_df.reset_index()
#     TM_df = TM_df.merge(simpsons_values, on=['activity_from', 'activity_to'], how='left')

#     # Set the original index back
#     return TM_df.set_index(['activity_from', 'activity_to'])[[process_feature_name]]

