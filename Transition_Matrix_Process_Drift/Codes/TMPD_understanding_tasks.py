# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 17:00:46 2024

@author: Antonio Carlos Meira Neto
"""

import os
import yaml
from string import Template
import ast
import sys
thismodule = sys.modules[__name__]
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as ss
import pm4py
from pm4py.objects.dfg.obj import DFG
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from graphviz import Digraph
from IPython.display import display, Image
from openai import OpenAI
import google.generativeai as genai



# Helper function to identify the type of variable
def identify_statistical_test(series, feature_name=None):
    """
    Determines the type of statistical test to apply based on the values in the series or feature name.
    """
    # If all values are between 0 and 1, it's a proportion
    if np.all((series >= 0) & (series <= 1)):
        return 'proportion_test'
    # If all values are integers, it's a count
    if np.all(np.equal(np.mod(series, 1), 0)):
        return 'count_test'
    # Otherwise, treat as mean_test
    return 'mean_test'


# Calculate Cohen's h effect size for proportions.
def cohen_h(p1, p2):
    """
    Calculates Cohen's h effect size for proportions.

    Args:
        p1 (float): Proportion 1.
        p2 (float): Proportion 2.

    Returns:
        float: Cohen's h effect size.
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


# Calculate Cramers V statistic for categorical-categorical association
def cramers_corrected_stat(data, ref_column, det_column, ref_value, det_value):
    """
    Calculate Cramér's V statistic for categorical-categorical association with correction for continuity.

    Args:
        data (DataFrame): The data containing the variables.
        ref_column (str): The reference column name.
        det_column (str): The detection column name.
        ref_value (int): The reference value (count) for the event.
        det_value (int): The detection value (count) for the event.

    Returns:
        float: Cramér's V statistic.
    """
    # Calculate total counts for ref and det
    total_ref = data[ref_column].sum() - ref_value
    total_det = data[det_column].sum() - det_value

    # Constructing the contingency table
    contingency_table = np.array([[ref_value, total_ref], [det_value, total_det]])

    # Perform Chi-squared test with correction
    chi2 = ss.chi2_contingency(contingency_table, correction=True, lambda_='log-likelihood')[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# Function to perform count test (Chi-squared or Fisher's exact test)
def perform_count_test(merged_windows, row, variable_ref, variable_det):
    """
    Perform a count statistical test (Chi-squared or Fisher's exact test) on the given data.

    Args:
        merged_windows (DataFrame): The merged windows data.
        row (Series): The specific row of data to test.
        variable_ref (str): The reference variable (column) name.
        variable_det (str): The detection variable (column) name.

    Returns:
        float: The p-value of the statistical test.
    """
    freq_reference = row[variable_ref]
    freq_detection = row[variable_det]
    total_freq_reference = merged_windows[variable_ref].sum() - freq_reference
    total_freq_detection = merged_windows[variable_det].sum() - freq_detection
    contingency_table = [[freq_reference, total_freq_reference], [freq_detection, total_freq_detection]]

    if 0 in contingency_table or min(min(contingency_table)) < 5:
        _, p_value = fisher_exact(contingency_table)
    else:
        _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value


# Adjusted function with pseudo-count (also known as Laplace smoothing) for handling zero frequencies
def perform_proportions_test(merged_windows, row, transition, variable_ref, variable_det, pseudo_count):
    """
    Perform a proportions statistical test with pseudo-count adjustment for zero frequencies.

    Args:
        merged_windows (DataFrame): The merged windows data.
        row (Series): The specific row of data to test.
        transition (tuple): The transition (from, to) for the activity.
        variable_ref (str): The reference variable (column) name.
        variable_det (str): The detection variable (column) name.
        pseudo_count (int): The pseudo-count to add for smoothing.

    Returns:
        float: The p-value of the statistical test.
    """
    prob_reference = row[variable_ref]
    prob_detection = row[variable_det]

    # Check for equal values, return non-significant p-value
    if prob_reference == prob_detection:
        return 1

    total_count_reference = merged_windows.loc[merged_windows['activity_from'] == transition[0], 'frequency_ref'].sum()
    total_count_detection = merged_windows.loc[merged_windows['activity_from'] == transition[0], 'frequency_det'].sum()

    # Adjust counts by adding pseudo-counts directly to the successes and total counts
    # This approach ensures that zero frequencies are adjusted to allow for statistical testing
    success_reference = prob_reference * total_count_reference + pseudo_count
    success_detection = prob_detection * total_count_detection + pseudo_count
    total_count_reference += pseudo_count
    total_count_detection += pseudo_count

    # Calculate the successes and attempts for both groups, adjusting for pseudo-counts
    nobs = [total_count_reference, total_count_detection]
    count = [success_reference, success_detection]

    # Perform the proportions Z-test
    stat, p_value = proportions_ztest(count, nobs)

    return p_value


def t_test_from_summary_stats(mean1, mean2, std1, std2, n1, n2, epsilon=1e-8):
    """
    Perform Welch's t-test using summary statistics (mean, std, n) for each group.
    
    This is useful when we only have aggregated values per group (like per transition),
    and not raw sample arrays, so scipy's ttest_ind cannot be used directly.
    
    Args:
        mean1 (float): Mean of group 1 (reference).
        mean2 (float): Mean of group 2 (detection).
        std1 (float): Standard deviation of group 1.
        std2 (float): Standard deviation of group 2.
        n1 (float): Sample size (or weight/frequency) of group 1.
        n2 (float): Sample size (or weight/frequency) of group 2.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Two-tailed p-value of the Welch's t-test.
    """
    # Standard error
    se = np.sqrt((std1**2 / (n1 + epsilon)) + (std2**2 / (n2 + epsilon)))
    
    # t-statistic
    t_stat = (mean1 - mean2) / (se + epsilon)
    
    # Degrees of freedom using Welch–Satterthwaite equation
    df_num = (std1**2 / (n1 + epsilon) + std2**2 / (n2 + epsilon))**2
    df_denom = ((std1**2 / (n1 + epsilon))**2) / (n1 - 1 + epsilon) + \
               ((std2**2 / (n2 + epsilon))**2) / (n2 - 1 + epsilon)
    df = df_num / (df_denom + epsilon)
    
    # Two-tailed p-value
    p_value = 2 * (1 - ss.t.cdf(abs(t_stat), df))
    return p_value


def cohen_d_from_summary_stats(mean1, mean2, std1, std2, epsilon=1e-8):
    """
    Calculate Cohen's d effect size using summary statistics.
    
    This estimates the standardized difference between two means
    using the pooled standard deviation. Used when only aggregated
    statistics (mean and std) are available per transition.

    Args:
        mean1 (float): Mean of group 1 (reference).
        mean2 (float): Mean of group 2 (detection).
        std1 (float): Standard deviation of group 1.
        std2 (float): Standard deviation of group 2.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Cohen's d effect size.
    """
    # Pooled standard deviation
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return (mean2 - mean1) / (pooled_std + epsilon)


def significant_transition_changes_detection(self, reference_transition_matrix, detection_transition_matrix, features_windows, dfg_changes):
    """
    Detect changes in transitions between two sets of windows (e.g., reference and detection windows), with a presence/absence rule using a percentage threshold.

    Args:
        self: The instance of the class (for accessing thresholds and pseudo-count).
        reference_transition_matrix (DataFrame): The reference transition matrix data.
        detection_transition_matrix (DataFrame): The detection transition matrix data.
        features_windows (list): List of feature names to analyze.
        dfg_changes (dict, optional): Dictionary containing new and deleted activities from DFG comparison.

    Returns:
        DataFrame: A DataFrame containing the transitions with significant changes.
    """
    # Ensure all features present in both, fill missing with 0 BEFORE merging
    all_features = set(reference_transition_matrix.columns) | set(detection_transition_matrix.columns)
    all_features -= {'activity_from', 'activity_to'}
    
    # For reference transition matrix: keep only activity_from, activity_to, and features, then add missing features with 0, and rename to _ref
    ref_df = reference_transition_matrix.copy()
    for feature in all_features:
        if feature not in ref_df.columns:
            ref_df[feature] = 0
    ref_df = ref_df[['activity_from', 'activity_to'] + sorted(all_features)]
    ref_df = ref_df.rename(columns={f: f + '_ref' for f in all_features})
    
    # For detection transition matrix: same, but _det
    det_df = detection_transition_matrix.copy()
    for feature in all_features:
        if feature not in det_df.columns:
            det_df[feature] = 0
    det_df = det_df[['activity_from', 'activity_to'] + sorted(all_features)]
    det_df = det_df.rename(columns={f: f + '_det' for f in all_features})
    
    # Merge on activity_from and activity_to
    merged_windows_df = pd.merge(
        ref_df, det_df,
        on=['activity_from', 'activity_to'],
        how='outer'
    )
    # Fill any remaining NaNs with 0
    merged_windows_df = merged_windows_df.fillna(0)

    # Create perspective classification function
    def classify_feature_perspective(feature):
        """Classify the perspective of a feature based on its name."""
        # Ensure feature lists are not None
        control_flow_features = self.control_flow_features if self.control_flow_features is not None else []
        time_features = self.time_features if self.time_features is not None else []
        resource_features = self.resource_features if self.resource_features is not None else []
        data_features = self.data_features if self.data_features is not None else []

        # First check control flow (simple direct match)
        if feature in control_flow_features:
            return 'control_flow'
        
        # Check if it's a compound feature (has underscores)
        elif '_' in feature:
            # Split feature parts
            parts = feature.split('_')
            method_type = parts[0]  # e.g., 'time', 'numerical', 'categorical'
            
            # Time perspective (e.g., "time_avg_timestamp")
            if method_type == 'time':
                for method, column in time_features:
                    if feature == f"{method}_{column}":
                        return 'time'
            
            # Numerical data perspective (e.g., "numerical_avg_Amount")
            elif method_type == 'numerical':
                for method, column in data_features:
                    if feature == f"{method}_{column}":
                        return 'data'
            
            # Categorical perspective (e.g., "categorical_encoding_frequency_Role_Appraiser->Credit Analyst")
            elif method_type == 'categorical':
                # Find the position of the last method part (encoding_frequency, encoding_probability, etc)
                # and the column name to properly handle column values that contain underscores
                for method, column in resource_features:
                    method_parts = method.split('_')  # e.g., ['categorical', 'encoding', 'frequency']
                    feature_prefix = f"{method}_{column}"  # e.g., "categorical_encoding_frequency_role"
                    if feature.startswith(feature_prefix):
                        return 'resource'
                
                # Then check data features the same way
                for method, column in data_features:
                    method_parts = method.split('_')
                    feature_prefix = f"{method}_{column}"
                    if feature.startswith(feature_prefix):
                        return 'data'
        
        return 'unknown'

    # Initialize a DataFrame to store the results with perspective column
    significant_transition_changes = pd.DataFrame(columns=pd.Index(['transition', 'feature', 'perspective', 'transition_status', 'activity_status', 'p_value', 'effect_size', 'ref_value', 'det_value', 'dif_value']))

    # Perform statistical tests for each variable and transition
    i = 0
    for index, row in merged_windows_df.iterrows():
        transition = (row['activity_from'], row['activity_to'])
        for feature in features_windows:
            variable_ref = feature + '_ref'
            variable_det = feature + '_det'
            ref_value = row[variable_ref]
            det_value = row[variable_det]
            ref_total = merged_windows_df[variable_ref].sum()
            det_total = merged_windows_df[variable_det].sum()
            ref_pct = ref_value / ref_total if ref_total > 0 else 0
            det_pct = det_value / det_total if det_total > 0 else 0

            is_significant = False
            transition_status = ''
            
            # First, check if the transition is significant enough to be considered
            # Skip if both percentages are below the threshold
            if ref_pct < self.presence_percentage_threshold_localization and det_pct < self.presence_percentage_threshold_localization:
                continue
            
            if ref_value == 0 and det_value == 0:
                effect_size = 0
                p_value = 1
                transition_status = 'no change'
            elif ref_value > 0 and det_value == 0:
                is_significant = True
                effect_size = 1
                p_value = 0
                transition_status = 'deleted'
            elif det_value > 0 and ref_value == 0:
                is_significant = True
                effect_size = 1
                p_value = 0
                transition_status = 'new'
            else:
                test_type = identify_statistical_test(merged_windows_df[[variable_ref, variable_det]].values.flatten())
                if test_type == 'count_test':
                    try:
                        p_value = perform_count_test(merged_windows_df, row, variable_ref, variable_det)
                        effect_size = cramers_corrected_stat(merged_windows_df, variable_ref, variable_det, ref_value, det_value)
                    except ValueError:
                        effect_size = 0
                        p_value = 1
                    is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_threshold_localization
                elif test_type == 'mean_test':
                    ref_value = row[variable_ref]
                    det_value = row[variable_det]
                    n_ref = row['frequency_ref']
                    n_det = row['frequency_det']
                    std_ref = merged_windows_df[variable_ref].std()
                    std_det = merged_windows_df[variable_det].std()

                    p_value = t_test_from_summary_stats(ref_value, det_value, std_ref, std_det, n_ref, n_det)
                    effect_size = cohen_d_from_summary_stats(ref_value, det_value, std_ref, std_det)

                    is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_threshold_localization
                elif test_type == 'proportion_test':
                    p_value = perform_proportions_test(merged_windows_df, row, transition, variable_ref, variable_det, self.pseudo_count_localization)
                    effect_size = cohen_h(ref_value, det_value)
                    is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_threshold_localization

            if is_significant:
                perspective = classify_feature_perspective(feature)
                if transition_status == '':
                    transition_status = 'significant difference'
                
                # Determine activity status based on dfg_changes
                activity_status = 'no change'
                if dfg_changes is not None:
                    new_activities = set(dfg_changes.get('New activities added to the process', []))
                    deleted_activities = set(dfg_changes.get('Deleted activities from the process', []))
                    
                    activity_from, activity_to = transition
                    
                    # Check which activities are new or deleted
                    new_activities_in_transition = []
                    deleted_activities_in_transition = []
                    
                    if activity_from in new_activities:
                        new_activities_in_transition.append(activity_from)
                    if activity_to in new_activities:
                        new_activities_in_transition.append(activity_to)
                    if activity_from in deleted_activities:
                        deleted_activities_in_transition.append(activity_from)
                    if activity_to in deleted_activities:
                        deleted_activities_in_transition.append(activity_to)
                    
                    # Build status description
                    status_parts = []
                    
                    if new_activities_in_transition:
                        status_parts.append(f"new({', '.join(new_activities_in_transition)})")
                    if deleted_activities_in_transition:
                        status_parts.append(f"deleted({', '.join(deleted_activities_in_transition)})")
                    
                    if status_parts:
                        activity_status = " | ".join(status_parts)
                
                significant_transition_changes.loc[i] = [transition, feature, perspective, transition_status, activity_status, p_value, effect_size, ref_value, det_value, det_value - ref_value]
                i += 1

    return significant_transition_changes


def create_dfg_from_dataset(dataset):
    """
    Create a Directly-Follows Graph (DFG) from the given dataset.

    Args:
        dataset (DataFrame): The input dataset containing the event log.

    Returns:
        DFG: A DFG object representing the process.
    """
    # Creating a DFG from the dataset
    dfg_transitions = {(row['activity_from'], row['activity_to']): row['frequency'] for index, row in dataset.iterrows() if row['frequency'] > 0}

    # Identifying real start activities
    real_start_activities = set(to for from_, to in dfg_transitions if from_ == 'START')
    start_activities_freq = {activity: dfg_transitions[('START', activity)] for activity in real_start_activities}

    # Identifying real end activities
    real_end_activities = set(from_ for from_, to in dfg_transitions if to == 'END')
    end_activities_freq = {activity: dfg_transitions[(activity, 'END')] for activity in real_end_activities}

    # Removing transitions that involve 'START' and 'END'
    dfg_transitions = {k: v for k, v in dfg_transitions.items() if 'START' not in k and 'END' not in k}

    # --- Add synthetic unknown activities for orphans ---
    # Find all activities
    all_activities = set()
    for (from_act, to_act) in dfg_transitions:
        all_activities.add(from_act)
        all_activities.add(to_act)
    # Remove synthetic nodes if present
    all_activities.discard('unknown_previous_activities')
    all_activities.discard('unknown_following_activities')

    # Find activities with no incoming transitions (orphans at start)
    incoming = {to_act for (_, to_act) in dfg_transitions}
    start_orphans = all_activities - incoming - set(start_activities_freq.keys())
    # Add synthetic incoming transitions for start orphans
    unknown_prev_total = 0
    for orphan in start_orphans:
        freq = start_activities_freq.get(orphan, 1)
        dfg_transitions[('unknown_previous_activities', orphan)] = freq
        unknown_prev_total += freq
    if unknown_prev_total > 0:
        start_activities_freq['unknown_previous_activities'] = unknown_prev_total

    # Find activities with no outgoing transitions (orphans at end)
    outgoing = {from_act for (from_act, _) in dfg_transitions}
    end_orphans = all_activities - outgoing - set(end_activities_freq.keys())
    # Add synthetic outgoing transitions for end orphans
    unknown_foll_total = 0
    for orphan in end_orphans:
        freq = end_activities_freq.get(orphan, 1)
        dfg_transitions[(orphan, 'unknown_following_activities')] = freq
        unknown_foll_total += freq
    if unknown_foll_total > 0:
        end_activities_freq['unknown_following_activities'] = unknown_foll_total

    return create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq)


def create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq):
    """
    Create a Directly-Follows Graph (DFG) from the given transitions and activity frequencies.

    Args:
        dfg_transitions (dict): A dictionary with (from_activity, to_activity) as keys and frequencies as values.
        start_activities_freq (dict): A dictionary with start activities and their frequencies.
        end_activities_freq (dict): A dictionary with end activities and their frequencies.

    Returns:
        DFG: A DFG object representing the process.
    """
    dfg = DFG()

    # Adding transitions to the DFG
    for (from_act, to_act), count in dfg_transitions.items():
        dfg.graph[(from_act, to_act)] += count

    # Adding real start activities
    for act, count in start_activities_freq.items():
        dfg.start_activities[act] += count

    # Adding real end activities
    for act, count in end_activities_freq.items():
        dfg.end_activities[act] += count

    return dfg


def compare_dfgs(dfg1, dfg2):
    """
    Compare two Directly-Follows Graphs (DFGs) and identify changes in transitions and activities.

    Args:
        dfg1 (DFG): The first DFG to compare.
        dfg2 (DFG): The second DFG to compare.

    Returns:
        dict: A dictionary summarizing the changes between the two DFGs.
    """
    # Helper to check if a node is synthetic
    def is_synthetic_node(node):
        return node in {'unknown_previous_activities', 'unknown_following_activities'}

    # Retrieve transition sets from the graphs, excluding synthetic transitions
    dfg1_transitions = set((a, b) for (a, b) in dfg1.graph.keys() if not is_synthetic_node(a) and not is_synthetic_node(b))
    dfg2_transitions = set((a, b) for (a, b) in dfg2.graph.keys() if not is_synthetic_node(a) and not is_synthetic_node(b))

    # Include transitions to END and from START explicitly, but only for non-synthetic activities
    dfg1_transitions |= set(('START', act) for act in dfg1.start_activities.keys() if not is_synthetic_node(act))
    dfg2_transitions |= set(('START', act) for act in dfg2.start_activities.keys() if not is_synthetic_node(act))
    dfg1_transitions |= set((act, 'END') for act in dfg1.end_activities.keys() if not is_synthetic_node(act))
    dfg2_transitions |= set((act, 'END') for act in dfg2.end_activities.keys() if not is_synthetic_node(act))

    # Calculate new, deleted, and altered transitions
    new_transitions = dfg2_transitions - dfg1_transitions
    deleted_transitions = dfg1_transitions - dfg2_transitions

    # Get activities, excluding synthetic
    dfg1_activities = set(x for t in dfg1.graph.keys() for x in t if not is_synthetic_node(x))
    dfg2_activities = set(x for t in dfg2.graph.keys() for x in t if not is_synthetic_node(x))
    new_activities = dfg2_activities - dfg1_activities
    deleted_activities = dfg1_activities - dfg2_activities

    # Get start and end activities, excluding synthetic
    dfg1_start_activities = set(a for a in dfg1.start_activities.keys() if not is_synthetic_node(a))
    dfg2_start_activities = set(a for a in dfg2.start_activities.keys() if not is_synthetic_node(a))
    dfg1_end_activities = set(a for a in dfg1.end_activities.keys() if not is_synthetic_node(a))
    dfg2_end_activities = set(a for a in dfg2.end_activities.keys() if not is_synthetic_node(a))
    new_start_activities = dfg2_start_activities - dfg1_start_activities
    deleted_start_activities = dfg1_start_activities - dfg2_start_activities
    new_end_activities = dfg2_end_activities - dfg1_end_activities
    deleted_end_activities = dfg1_end_activities - dfg2_end_activities

    dfg_changes = {
        'New transitions added to the process': list(new_transitions) if new_transitions else ["None"],
        'Deleted transitions from the process': list(deleted_transitions) if deleted_transitions else ["None"],
        'New activities added to the process': list(new_activities) if new_activities else ["None"],
        'Deleted activities from the process': list(deleted_activities) if deleted_activities else ["None"],
        # ,'New start activities added to the process': list(new_start_activities) if new_start_activities else ["None"]
        # ,'Deleted start activities from the process': list(deleted_start_activities) if deleted_start_activities else ["None"]
        # ,'New end activities added to the process': list(new_end_activities) if new_end_activities else ["None"]
        # ,'Deleted end activities from the process': list(deleted_end_activities) if deleted_end_activities else ["None"]

        # ,'All transitions in the reference window': list(dfg1_transitions) if dfg1_transitions else ["None"]
        # ,'All transitions in the detection window': list(dfg2_transitions) if dfg2_transitions else ["None"]
        # ,'All activities in the reference window': list(dfg1_activities) if dfg1_activities else ["None"]
        # ,'All activities in the detection window': list(dfg2_activities) if dfg2_activities else ["None"]
        # ,'All start activities in the reference window': list(dfg1_start_activities) if dfg1_start_activities else ["None"]
        # ,'All start activities in the detection window': list(dfg2_start_activities) if dfg2_start_activities else ["None"]
        # ,'All end activities in the reference window': list(dfg1_end_activities) if dfg1_end_activities else ["None"]
        # ,'All end activities in the detection window': list(dfg2_end_activities) if dfg2_end_activities else ["None"]
    }

    return dfg_changes


def create_bpmn_from_dfg(dfg):
    """
    Create a BPMN diagram from the given Directly-Follows Graph (DFG).

    Args:
        dfg (DFG): The input DFG.

    Returns:
        BPMNDiagram: A BPMN diagram object representing the process.
    """
    return pm4py.discover_bpmn_inductive(dfg, noise_threshold=0)


def wrap_text(text, max_length=10):
    """
    Wrap text to a specified maximum length, breaking lines at spaces between words.

    Args:
        text (str): The input text to wrap.
        max_length (int): The maximum length of each line.

    Returns:
        str: The wrapped text.
    """
    # Replace underscores with spaces to handle words like "Loan_application_received"
    text = text.replace('_', ' ')
    words = text.split()
    wrapped_text = ""
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            wrapped_text += '\n' + word
            current_length = len(word)
        else:
            if wrapped_text:
                wrapped_text += ' '
                current_length += 1
            wrapped_text += word
            current_length += len(word)

    return wrapped_text

def localization_dfg_visualization(dfg, change_informations, significant_transition_changes, bgcolor="white", rankdir="LR", node_penwidth="2", edge_penwidth="2", show_annotations=True):
    """
    Visualize the Directly-Follows Graph (DFG) with localization changes.

    Args:
        dfg (DFG): The input DFG.
        change_informations (dict): A dictionary with information about changes (new/deleted transitions and activities).
        bgcolor (str): Background color of the graph.
        rankdir (str): Direction of the graph layout (e.g., 'LR' for left-to-right).
        node_penwidth (str): Pen width for nodes.
        edge_penwidth (str): Pen width for edges.
        show_annotations (bool): Whether to show change annotations in the DFG. Defaults to True.

    Returns:
        None: Displays the graph inline in Jupyter Notebook.
    """
    
    dfg_graph = dfg.graph
    start_activities = dfg.start_activities
    end_activities = dfg.end_activities
    new_transitions = change_informations['New transitions added to the process']
    deleted_transitions = change_informations['Deleted transitions from the process']
    new_activities = change_informations['New activities added to the process']
    deleted_activities = change_informations['Deleted activities from the process']

    edge_annotations = {}
    for key, transitions in change_informations.items():
        if key.startswith("Transitions with variations in"):
            # Get the feature name
            prefix_length = len("Transitions with variations in")
            feature = key[prefix_length:].strip()
            for transition in transitions:
                # Find the row in significant_transition_changes for this transition and feature
                match = significant_transition_changes[
                    (significant_transition_changes['transition'] == transition) &
                    (significant_transition_changes['feature'] == feature)
                ]
                if not match.empty:
                    ref_value = match.iloc[0]['ref_value']
                    det_value = match.iloc[0]['det_value']
                    p_value = match.iloc[0]['p_value']
                    effect_size = match.iloc[0]['effect_size']
                    # Format p_value: 2 decimals, but if <0.01, show 0.00
                    if p_value < 0.01:
                        p_str = "0.00"
                    else:
                        p_str = f"{p_value:.2f}"
                    annotation = f"{feature} (ref: {ref_value:.2f}, det: {det_value:.2f}, pvalue: {p_str}, effsize: {effect_size:.2f})"
                else:
                    annotation = feature
                if transition in edge_annotations:
                    edge_annotations[transition].append(annotation)
                else:
                    edge_annotations[transition] = [annotation]

    dot = Digraph(engine='dot', graph_attr={'bgcolor': bgcolor, 'rankdir': rankdir})

    # Create a unique start and end node for visualization
    dot.node('START', shape='circle', label='START', width='0.8', style='filled', fillcolor='white', penwidth=node_penwidth)
    dot.node('END', shape='doublecircle', label='END', width='0.8', style='filled', fillcolor='white', penwidth=node_penwidth)

    # Add nodes and edges to the graph
    for (source, target), count in dfg_graph.items():
        # Set node shapes and labels
        source_label = wrap_text(f"{source} ({start_activities.get(source, count)})", max_length=15)
        target_label = wrap_text(f"{target} ({end_activities.get(target, count)})", max_length=15)


        # Determine node colors based on activity status
        source_color = 'blue' if source in new_activities else 'red' if source in deleted_activities else 'black'
        target_color = 'blue' if target in new_activities else 'red' if target in deleted_activities else 'black'

        # # Add nodes
        dot.node(source, label=source_label, shape='box', style='filled', fillcolor='white', color=source_color, penwidth=node_penwidth)
        dot.node(target, label=target_label, shape='box', style='filled', fillcolor='white', color=target_color, penwidth=node_penwidth)

        # Set edge colors based on transition type
        edge_color = 'black'
        if (source, target) in new_transitions:
            edge_color = 'blue'
        elif (source, target) in deleted_transitions:
            edge_color = 'red'
        elif (source, target) in edge_annotations:
            edge_color = 'orange'

            # Add edges
        if show_annotations and (source, target) in edge_annotations:
            dot.edge(source, target
                    , label="Freq: " + str(count) + '\nDif. in '
                        + '\nDif. in '.join(edge_annotations[(source, target)])
                    , color=edge_color, penwidth=edge_penwidth) 
        else: 
            dot.edge(source, target, label="Freq: " + str(count), color=edge_color, penwidth=edge_penwidth)    # Connect the start node to the real start activities and the real end activities to the end node
    for act in start_activities:
        if act not in end_activities:  # Avoid connecting end activities again
            count = start_activities.get(act, 0)  # Get the count for the activity

            # Set edge colors based on transition type
            edge_color = 'black'
            if ('START', act) in new_transitions:
                edge_color = 'blue'
            elif ('START', act) in deleted_transitions:
                edge_color = 'red'
            elif ('START', act) in edge_annotations:
                edge_color = 'orange'

            if show_annotations and ('START', act) in edge_annotations: 
                dot.edge('START', act
                         , label="Freq: " + str(count) + '\nDif. in '
                            + '\nDif. in '.join(edge_annotations[('START', act)])
                        , color=edge_color, style='bold', penwidth=edge_penwidth)
            else:
                dot.edge('START', act, label="Freq: " + str(count), color=edge_color, style='bold', penwidth=edge_penwidth)

    for act in end_activities:
        if act not in start_activities:  # Avoid connecting start activities again
            count = end_activities.get(act, 0)  # Get the count for the activity

            # Set edge colors based on transition type
            edge_color = 'black'
            if (act, 'END') in new_transitions:
                edge_color = 'blue'
            elif (act, 'END') in deleted_transitions:
                edge_color = 'red'
            elif (act, 'END') in edge_annotations:
                edge_color = 'orange'

            if show_annotations and (act, 'END') in edge_annotations: 
                dot.edge(act, 'END'
                         , label="Freq: " + str(count) + '\nDif. in '
                            + '\nDif. in '.join(edge_annotations[(act, 'END')])
                        , color=edge_color, style='bold', penwidth=edge_penwidth)
            else:
                dot.edge(act, 'END', label="Freq: " + str(count), color=edge_color, style='bold', penwidth=edge_penwidth)

    # Render and display the graph inline in Jupyter Notebook
    png_data = dot.pipe(format='png')
    display(Image(png_data))


def create_process_tree_from_dfg(dfg, parameters):
    """
    Create a process tree from the given Directly-Follows Graph (DFG) using the inductive miner algorithm.

    Args:
        dfg (DFG): The input DFG.
        parameters (dict): Parameters for the inductive miner algorithm.

    Returns:
        ProcessTree: A process tree object representing the process.
    """
    # Create process tree
    process_tree = inductive_miner.apply(dfg, parameters=parameters) 
    # process_tree = pm4py.discover_process_tree_inductive(dfg, noise_threshold=0.0)

    # Get the bpmn text
    reference_bpmn_text = str(process_tree._get_root())

    # Replace control-flows symbols to their corresponding names
    reference_bpmn_text = reference_bpmn_text.replace('->', 'Sequence')
    reference_bpmn_text = reference_bpmn_text.replace('+', 'Parallel')
    reference_bpmn_text = reference_bpmn_text.replace('X', 'Conditional')
    reference_bpmn_text = reference_bpmn_text.replace('*', 'Loop')
    
    return reference_bpmn_text
    

def llm_instanciating(llm_company, llm_model, api_key):
    """
    Instantiate the LLM (Large Language Model) class for OpenAI or Google.

    Args:
        llm_company (str): The company providing the LLM ('openai' or 'google').
        llm_model (str): The model name or ID.
        api_key (str): The API key for authentication.

    Returns:
        llm: An instance of the LLM class for the specified company and model.
    """
        
    if llm_company == "openai":

        # insert API_KEY in the file to be read here
        if api_key:
            with open(api_key, 'r') as file:
                os.environ["OPENAI_API_KEY"] = file.read().rstrip()

        # Instanciating LLM class
        return OpenAI()
    
    elif llm_company == "google":

        # insert API_KEY in the file to be read here
        if api_key:
            with open(api_key, 'r') as file:
                os.environ["GOOGLE_CLOUD_API_KEY"] = file.read().rstrip()
                genai.configure(api_key=os.environ['GOOGLE_CLOUD_API_KEY'])

        # Instanciating LLM class
        return genai.GenerativeModel(llm_model)
    

def llm_call_response(llm_company, llm_model, llm, user_prompt):
    """
    Call the LLM with the user prompt and return the response.

    Args:
        llm_company (str): The company providing the LLM ('openai' or 'google').
        llm_model (str): The model name or ID.
        llm: The LLM instance.
        user_prompt (str): The prompt to send to the LLM.

    Returns:
        str: The response text from the LLM.
    """
    if llm_company == "openai":

        if llm_model in ["o3-mini", "o4-mini"]:
            # For o3-mini, we use the chat completions API
            response = llm.chat.completions.create(
                seed=42
                , reasoning_effort="high"
                , model=llm_model
                , messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        elif llm_model in ["gpt-5"]:
            # For o3 and o4, we use the chat completions API with temperature and top_p set to 0
            response = llm.chat.completions.create(
                seed=42
                , model=llm_model
                , messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            response = llm.chat.completions.create(
                temperature=0
                , top_p=0.000000000000001
                , seed=42
                , model=llm_model
                , messages=[
                # {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
                ]
            )

        return response.choices[0].message.content
    
    elif llm_company == "google":

        response = llm.generate_content(user_prompt,
            generation_config=genai.types.GenerationConfig(
                # Only one candidate for now.
                candidate_count=1
                # , stop_sequences=['x']
                # , max_output_tokens=None
                , temperature=0
                , top_p=0.000000000000001
                , top_k=40
            )
        )
        
        return response.text
        

def llm_instructions_load(llm_instructions_path):
    """
    Load and parse the LLM instructions from a YAML file.

    Args:
        llm_instructions_path (str): The file path to the YAML instructions.

    Returns:
        dict: A dictionary containing the parsed instructions.
    """

    # Open the YAML file for reading with UTF-8 encoding
    with open(llm_instructions_path, 'r', encoding='utf-8') as file:
        # Parse the yaml file into a Python dictionary
        llm_instructions = yaml.safe_load(file)
                            
    return llm_instructions



# def llm_bpmn_enhance_instructions(llm_instructions, reference_bpmn_text, detection_bpmn_text, change_informations):

#     llm_instructions["bpmn_diagram_enhance"] += (
#         ''' 
#         ### BPMN diagrams ###
#         - **The BPMN before the concept drift (reference window):** {0}.
#         - **The BPMN after the concept drift (detection window):** {1}.

#         ### Detailed data of transitions and activities ###
#         **List of transitions and activities:**
#         {2}
#         '''
#     ).format(reference_bpmn_text, detection_bpmn_text, change_informations)

#     return llm_instructions["bpmn_diagram_enhance"]



# def llm_transition_analysis_instructions(llm_instructions, reference_bpmn_text, detection_bpmn_text, change_informations, llm_bpmn_analysis_response):

#     llm_instructions["changes_informations"] += (
#         '''
#             ### BPMN diagrams ###
#             - The BPMN before the concept drift: {0}. \n
#             - The BPMN after the concept drift: {1}. \n

#             ### BPMN diagrams comparison analysis ### 
#             {2}

#             ### Transition and Activities comparison lists ###
#             {3}
#         ''').format(reference_bpmn_text, detection_bpmn_text, llm_bpmn_analysis_response, change_informations)

#     return llm_instructions["changes_informations"]


# def llm_bpmn_analysis_prompt(llm_instructions, reference_bpmn_text, detection_bpmn_text):
#     """
#     Add BPMN diagrams to the LLM prompt for analysis.

#     Args:
#         llm_instructions (dict): The LLM instructions dictionary.
#         reference_bpmn_text (str): The BPMN text for the reference window.
#         detection_bpmn_text (str): The BPMN text for the detection window.

#     Returns:
#         str: The updated prompt for BPMN analysis.
#     """

#     # Add BPMN diagrams to prompt
#     llm_instructions["instructions_bpmn_analysis"] += (
#     ''' 
#     \n### BPMN diagrams ###
#     - The BPMN before the concept drift: {0}. \n
#     - The BPMN after the concept drift: {1}. \n

#     ''').format(reference_bpmn_text, detection_bpmn_text)

#     return llm_instructions["instructions_bpmn_analysis"]


# def llm_classification_prompt(llm_instructions, change_informations, reference_bpmn_text, detection_bpmn_text, llm_bpmn_analysis_response):
#     """
#     Create the classification prompt for the LLM based on the changes and BPMN diagrams.

#     Args:
#         llm_instructions (dict): The LLM instructions dictionary.
#         change_informations (dict): A dictionary with information about changes (new/deleted transitions and activities).
#         reference_bpmn_text (str): The BPMN text for the reference window.
#         detection_bpmn_text (str): The BPMN text for the detection window.
#         llm_bpmn_analysis_response (str): The response from the LLM for BPMN analysis.

#     Returns:
#         str: The complete prompt for classification.
#     """

#     # Get the prompt
#     prompt = llm_instructions["instructions_classification"] 

#     # Add BPMN Diagrams Comparison Analysis to prompt
#     prompt += (
#         ''' 
#         \n### BPMN Diagrams Comparison Analysis ###
#         {0}. \n

#         ''').format(llm_bpmn_analysis_response)
    

#     ### Add Transition and Activities Changes List and Control-flow Change Patterns to prompt depending on conditions
    
#     # If there is at least a new or deleted activity, then suggest SRE, PRE, CRE, or RP
#     if change_informations['New activities added to the process'] != ['None'] or change_informations['Deleted activities from the process'] != ['None']:
        
#         prompt += (
#         ''' 
#         \n### Transition and Activities Changes List ###
#         \n'New transitions added to the process': {0}.
#         \n'Deleted transitions from the process': {1}.
#         \n'New activities added to the process': {2}.
#         \n'Deleted activities from the process': {3}.

#         ''').format(change_informations['New transitions added to the process']
#                     , change_informations['Deleted transitions from the process']
#                     , change_informations['New activities added to the process']
#                     , change_informations['Deleted activities from the process'])
        
#         prompt += (
#         ''' 
#         \n### Control-flow Change Patterns ###\n
#         ''')
#         prompt += (
#             llm_instructions['controlflow_change_patterns']['sre_instructions'] 
#             + llm_instructions['controlflow_change_patterns']['pre_instructions'] 
#             + llm_instructions['controlflow_change_patterns']['cre_instructions'] 
#             + llm_instructions['controlflow_change_patterns']['rp_instructions'] 
#         )  

#     # If the changes don't involve addition or deletion of activities but rather addition or deletion of transitions between existing activities, then suggest SM, CM, PM, or SW, CF, PL, LP,CD,  CB, or CP
#     elif change_informations['New transitions added to the process'] != ['None'] or change_informations['Deleted transitions from the process'] != ['None']:
        
#         prompt += (
#         ''' 
#         \n### Transition and Activities Changes List ###
#         \n'New transitions added to the process': {0}.
#         \n'Deleted transitions from the process': {1}.
#         \n'New activities added to the process': {2}.
#         \n'Deleted activities from the process': {3}.

#         ''').format(change_informations['New transitions added to the process']
#                     , change_informations['Deleted transitions from the process']
#                     , change_informations['New activities added to the process']
#                     , change_informations['Deleted activities from the process'])

#         prompt += (
#         ''' 
#         \n### Control-flow Change Patterns ###\n
#         ''')

#         # Movement Patterns
#         prompt += (llm_instructions['controlflow_change_patterns']['sm_instructions'] 
#                 + llm_instructions['controlflow_change_patterns']['cm_instructions'] 
#                 + llm_instructions['controlflow_change_patterns']['pm_instructions'] 
#                 + llm_instructions['controlflow_change_patterns']['sw_instructions'] 
#         )

#         # Gateway Type Changes
#         prompt += (llm_instructions['controlflow_change_patterns']['pl_instructions'] 
#                 + llm_instructions['controlflow_change_patterns']['cf_instructions'] 
#         )

#         # Synchronization (Parallel involved)
#         prompt += (llm_instructions['controlflow_change_patterns']['cd_instructions'] 
#         )

#         # Bypass (XOR involved)
#         prompt += (llm_instructions['controlflow_change_patterns']['cb_instructions'] 
#         )

#         # Loop Fragment Changes
#         prompt += (llm_instructions['controlflow_change_patterns']['lp_instructions'] 
#                    #llm_instructions['controlflow_change_patterns']['cp_instructions'] 
#         )


#     # If the changes don't involve addition or deletion of activities nor addition or deletion of transitions between existing activities, but rather only changes in the transitions, then is FR
#     else:

#         prompt += (
#         ''' 
#         \n### Transition and Activities Changes List ###\n
#         {0}

#         ''').format(change_informations)

#         prompt += (
#         ''' 
#         \n### Control-flow Change Patterns ###\n
#         ''')

#         prompt += (
#             llm_instructions['controlflow_change_patterns']['fr_instructions'] 
#         )


#     return prompt


# def llm_classification_formatting(characterization_classification):
#     """
#     Format the characterization classification string into a dictionary.

#     Args:
#         characterization_classification (str): The characterization classification string.

#     Returns:
#         dict: The formatted classification as a dictionary.
#     """

#     # Finding the start and end of the dictionary string
#     try:
#         start_str = "result_dict = {"
#         end_str = "}"
#         start_index = characterization_classification.find(start_str) + len(start_str) - 1
#         end_index = characterization_classification.find(end_str, start_index) + 1

#         return ast.literal_eval(characterization_classification[start_index:end_index].strip())
#     except:
#         return "Classification not in the expected format."


def llm_characterization_prompt(llm_instructions, reference_transition_matrix, detection_transition_matrix, tmpd_instance):
    """
    Construct the LLM prompt for process characterization using the provided instructions and data sources.

    Args:
        llm_instructions (dict): Instructions for the LLM.
        reference_transition_matrix (pd.DataFrame): Reference transition matrix data.
        detection_transition_matrix (pd.DataFrame): Detection transition matrix data.
        tmpd_instance: The TMPD class instance containing additional attributes.

    Returns:
        str: The constructed prompt for the LLM.
    """
    prompt = llm_instructions['main_instructions']
    sources = llm_instructions['sources']

    # Add information sources based on configuration
    for source in sources:
        if source == "significant_transition_changes" and hasattr(tmpd_instance, 'significant_transition_changes'):
            prompt += "\n\n## Significant Transition Changes Analysis \n"
            if not tmpd_instance.significant_transition_changes.empty:
                prompt += f"Detailed statistical analysis of {len(tmpd_instance.significant_transition_changes)} significant changes:\n"
                prompt += tmpd_instance.significant_transition_changes.to_string(index=False)
            else:
                prompt += "No statistically significant changes detected at the transition level.\n"
        
        elif source == "reference_transition_matrix":
            prompt += "\n\n## Reference Transition Matrix \n"
            prompt += reference_transition_matrix.to_string(index=False) if not reference_transition_matrix.empty else "No reference transition matrix available.\n"
        
        elif source == "detection_transition_matrix":
            prompt += "\n\n## Detection Transition Matrix \n"
            prompt += detection_transition_matrix.to_string(index=False) if not detection_transition_matrix.empty else "No detection transition matrix available.\n"
        
        elif source == "high_level_changes" and hasattr(tmpd_instance, 'high_level_changes'):
            prompt += "\n\n## High-Level Changes \n"
            for key, value in tmpd_instance.high_level_changes.items():
                prompt += f"{key}: {value}\n"
        
        elif source == "reference_bpmn_text" and hasattr(tmpd_instance, 'reference_bpmn_text'):
            prompt += "\n\n## Reference Window BPMN Diagram (Process Model) \n"
            prompt += f"{tmpd_instance.reference_bpmn_text}\n"
        elif source == "detection_bpmn_text" and hasattr(tmpd_instance, 'detection_bpmn_text'):
            prompt += "\n\n# Detection Window BPMN Diagram (Process Model) \n"
            prompt += f"{tmpd_instance.detection_bpmn_text}\n"
        
        elif source in llm_instructions:
            prompt += f"\n{llm_instructions[source]}\n"
    
    return prompt


# def llm_characterization_formatting(characterization_response):
#     """
#     Format and structure the characterization response.
    
#     Args:
#         characterization_response (str): Raw LLM response for characterization.
    
#     Returns:
#         dict: Structured characterization analysis.
#     """
#     # For now, return the response as-is
#     # In the future, this could parse the response into structured format
#     return {
#         "raw_response": characterization_response,
#         "analysis_timestamp": pd.Timestamp.now(),
#         "status": "completed"
#     }

