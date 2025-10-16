# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import sys
thismodule = sys.modules[__name__]

def get_delta_matrix(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original,
                    control_flow_features=None, time_features=None, resource_features=None, data_features=None):
    """
    Computes the difference (Delta Matrix) between the process representation of the reference transition matrix and the detection transition matrix.

    Args:
        process_representation_reference_transition_matrix_original (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix_original (DataFrame): Detection transition matrix process representation.
        control_flow_features (set): Set of control flow feature names.
        time_features (set): Set of time feature names.
        resource_features (set): Set of resource feature names.
        data_features (set): Set of data feature names.

    Returns:
        DataFrame: Delta Matrix representing the absolute differences.
    """

    # Ensure feature lists are not None
    if control_flow_features is None:
        control_flow_features = []
    if time_features is None:
        time_features = []
    if resource_features is None:
        resource_features = []
    if data_features is None:
        data_features = []

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Control Flow Features: Fill missing transitions with 0
    derived_control_flow_features_reference = [col for col in process_representation_reference_transition_matrix.columns for feat in control_flow_features if col.startswith(feat)]
    derived_control_flow_features_detection = [col for col in process_representation_detection_transition_matrix.columns for feat in control_flow_features if col.startswith(feat)]
    delta_matrix_control_flow = abs(process_representation_reference_transition_matrix[derived_control_flow_features_reference].subtract(process_representation_detection_transition_matrix[derived_control_flow_features_detection], fill_value=0))

    # Time Features: Fill missing transitions with 0
    derived_time_features_reference = [col for col in process_representation_reference_transition_matrix.columns for feat in time_features if col.startswith(feat)]
    derived_time_features_detection = [col for col in process_representation_detection_transition_matrix.columns for feat in time_features if col.startswith(feat)]
    delta_matrix_time = abs(process_representation_reference_transition_matrix[derived_time_features_reference].subtract(process_representation_detection_transition_matrix[derived_time_features_detection], fill_value=0))

    # Resource Features: Fill missing transitions with 0
    derived_resource_features_reference = [col for col in process_representation_reference_transition_matrix.columns for feat in resource_features if col.startswith(feat)]
    derived_resource_features_detection = [col for col in process_representation_detection_transition_matrix.columns for feat in resource_features if col.startswith(feat)]
    delta_matrix_resource = abs(process_representation_reference_transition_matrix[derived_resource_features_reference].subtract(process_representation_detection_transition_matrix[derived_resource_features_detection], fill_value=0))

    # Data Features: Fill missing transitions with 0
    derived_data_features_reference = [col for col in process_representation_reference_transition_matrix.columns for feat in data_features if col.startswith(feat)]
    derived_data_features_detection = [col for col in process_representation_detection_transition_matrix.columns for feat in data_features if col.startswith(feat)]
    delta_matrix_data = abs(process_representation_reference_transition_matrix[derived_data_features_reference].subtract(process_representation_detection_transition_matrix[derived_data_features_detection], fill_value=0))

    # Combine all delta matrices into a single DataFrame
    delta_matrix = pd.concat([delta_matrix_control_flow, delta_matrix_time, delta_matrix_resource, delta_matrix_data], axis=1)
    return delta_matrix


def get_delta_matrix_aggregation(delta_matrix, process_representation_reference_transition_matrix, process_representation_detection_transition_matrix, change_feature_params):
    """
    Aggregates the delta matrix values using the specified aggregation function.

    Args:
        delta_matrix (DataFrame): Delta Matrix.
        process_representation_reference_transition_matrix (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix (DataFrame): Detection transition matrix process representation.
        change_feature_params (dict): Parameters for aggregation, including the feature and aggregation function.

    Returns:
        float: Aggregated value.
    """
    result = delta_matrix[change_feature_params['process_feature']].agg(change_feature_params['agg_function'])
    if isinstance(result, pd.Series):
        # If multiple columns, aggregate again to get a single value
        result = result.agg(change_feature_params['agg_function'])
    return result


def get_delta_matrix_multiple_aggregation(delta_matrix, process_representation_reference_transition_matrix, process_representation_detection_transition_matrix, change_feature_params):
    """
    Performs multiple aggregations on the delta matrix: horizontal aggregation followed by vertical aggregation.

    Args:
        delta_matrix (DataFrame): Delta Matrix.
        process_representation_reference_transition_matrix (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix (DataFrame): Detection transition matrix process representation.
        change_feature_params (dict): Parameters for aggregation, including the feature and aggregation function.

    Returns:
        float: Result of the multiple aggregations.
    """

    # Filter columns that start with the specified prefix
    columns_to_aggregate = delta_matrix.filter(like=change_feature_params['process_feature']).columns
    
    # Apply the horizontal aggregation
    horizontal_agg = delta_matrix[columns_to_aggregate].apply(change_feature_params['agg_function'], axis=1)
    
    # Apply the vertical aggregation
    return horizontal_agg.agg(change_feature_params['agg_function'])


def get_delta_matrix_percentage(delta_matrix, process_representation_reference_transition_matrix, process_representation_detection_transition_matrix, change_feature_params):
    """
    Get the proportion of difference to maximum difference possible

    Args:
        delta_matrix (DataFrame): Delta Matrix.
        process_representation_reference_transition_matrix (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix (DataFrame): Detection transition matrix process representation.
        change_feature_params (dict): Parameters for percentage calculation, including the feature.

    Returns:
        float: Percentage of difference.
    """

    total_delta = process_representation_reference_transition_matrix[change_feature_params['process_feature']].sum() + process_representation_detection_transition_matrix[change_feature_params['process_feature']].sum()

    return delta_matrix[change_feature_params['process_feature']].sum()/total_delta


def get_delta_matrix_aggregation_weight(delta_matrix, process_representation_reference_transition_matrix, process_representation_detection_transition_matrix, change_feature_params):
    """
    Aggregates the weighted delta matrix values using the specified aggregation function.

    Args:
        delta_matrix (DataFrame): Delta Matrix.
        process_representation_reference_transition_matrix (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix (DataFrame): Detection transition matrix process representation.
        change_feature_params (dict): Parameters for aggregation, including the feature, weight feature, and aggregation function.

    Returns:
        float: Aggregated weighted value.
    """

    return delta_matrix[change_feature_params['process_feature']].multiply(delta_matrix[change_feature_params['weight_feature']].values, axis="index", fill_value=0).agg(change_feature_params['agg_function'])


def get_delta_matrix_strategy(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params_dict, control_flow_features=None, time_features=None, resource_features=None, data_features=None):
    """
    Computes the difference between the process representation of the reference transition matrix and the detection transition matrix using various strategies.
    Passes feature sets to get_delta_matrix for correct fill logic.

    Args:
        process_representation_reference_transition_matrix_original (DataFrame): Reference transition matrix process representation.
        process_representation_detection_transition_matrix_original (DataFrame): Detection transition matrix process representation.
        change_feature_params_dict (dict): Dictionary containing the strategies for computing differences, including:
            - 'frequency_delta' : {'process_feature':'frequency', 'method':'aggregation', 'agg_function' : 'sum'}
            - 'probability_delta' : {'process_feature':'probability', 'method':'aggregation', 'agg_function' : 'sum'}
            - 'frequency_delta_percentage' : {'process_feature':'frequency', 'method':'percentage'}
            - 'prob_freq_delta' : {'process_feature':'probability', 'method':'aggregation_weight', 'agg_function' : 'sum', 'weight_feature' : 'frequency'}

    Returns:
        dict: Dictionary containing the computed differences for each strategy.
    """

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Call delta matrix function with feature sets
    delta_matrix = get_delta_matrix(process_representation_reference_transition_matrix, process_representation_detection_transition_matrix,
                                    control_flow_features=control_flow_features,
                                    time_features=time_features,
                                    resource_features=resource_features,
                                    data_features=data_features)

    # Loop to call the change features methods to aggregate all differences - Delta Vector
    change_features_delta_matrix_dict = {}
    for change_feature, change_feature_params in change_feature_params_dict.items():

        try:
            change_features_delta_matrix_dict[change_feature] = getattr(thismodule, "get_delta_matrix_" + change_feature_params['method'])(
                delta_matrix,
                process_representation_reference_transition_matrix,
                process_representation_detection_transition_matrix,
                change_feature_params
            )

        except Exception as e:
            print("Unknown change feature: ", change_feature)
            print("Error: ", e)

    return change_features_delta_matrix_dict

                    
def get_contingency_matrix(distribution_reference, distribution_detection, contingency_matrix_sum_value=5, remove_zeros=True):
    """
    Create a contingency matrix from two distributions for statistical testing.
    
    Args:
        distribution_reference (DataFrame): Reference distribution with columns including 'activity_from', 'activity_to', and feature columns.
        distribution_detection (DataFrame): Detection distribution with same structure as reference.
        contingency_matrix_sum_value (int): Value to add to the contingency matrix to avoid zeros.
        remove_zeros (bool): Whether to remove rows with all zeros.
    
    Returns:
        numpy.ndarray: 2x2 contingency matrix for statistical testing.
    """
    # Create the contingency table
    contingency_matrix = pd.merge(distribution_reference, distribution_detection
                                , left_index=True, right_index=True, how='outer', suffixes=('_refeference_window', '_detection_window')).fillna(0).astype(int)

    # Remove zeros 
    if remove_zeros:
        contingency_matrix = contingency_matrix.loc[(contingency_matrix!=0).any(axis=1)]

    # Add value to all  
    contingency_matrix += contingency_matrix_sum_value

    return contingency_matrix


def cramers_corrected_stat(confusion_matrix):

    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    """

    chi2 = ss.chi2_contingency(confusion_matrix, correction=True, lambda_='log-likelihood')[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def get_statistic_test_cramers_v(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_transition_matrix[[change_feature_params['process_feature']]], process_representation_detection_transition_matrix[[change_feature_params['process_feature']]])

    # Cramer's V corrected
    return cramers_corrected_stat(contingency_matrix)


def get_statistic_test_g_test(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_transition_matrix[[change_feature_params['process_feature']]], process_representation_detection_transition_matrix[[change_feature_params['process_feature']]])

    # Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
    return ss.chi2_contingency(contingency_matrix, correction=True, lambda_='log-likelihood')[1]


def get_statistic_test_multiple_g_test(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params):
    """
    Perform G-test (log-likelihood ratio test) for each unique value in the categorical feature and aggregate the results.

    Args:
        process_representation_reference_transition_matrix_original (DataFrame): Reference transition matrix dataframe.
        process_representation_detection_transition_matrix_original (DataFrame): Detection transition matrix dataframe.
        change_feature_params (dict): Parameters for the test, including:
            - 'process_feature': Prefix of the feature to filter columns.
            - 'method': Statistical method (e.g., 'g_test').
            - 'contingency_matrix_sum_value': Value to add to the contingency matrix (default: 5).
            - 'remove_zeros': Whether to remove rows with all zeros.

    Returns:
        float: Aggregated p-value from the G-tests.
    """
    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Filter columns that start with the specified prefix
    columns_to_use_reference = process_representation_reference_transition_matrix.filter(like=change_feature_params['process_feature']).columns
    columns_to_use_detection = process_representation_detection_transition_matrix.filter(like=change_feature_params['process_feature']).columns

    # Ensure all relevant columns are included, even if they exist in only one dataframe
    all_columns = columns_to_use_reference.union(columns_to_use_detection)

    # Align both dataframes to have the same columns
    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix.reindex(columns=all_columns, fill_value=0)
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix.reindex(columns=all_columns, fill_value=0)

    # Initialize a list to store p-values
    p_values = []

    # Perform G-test for each unique value (column)
    for column in all_columns:
        # Create a contingency matrix for the current column
        contingency_matrix = get_contingency_matrix(
            process_representation_reference_transition_matrix[[column]],
            process_representation_detection_transition_matrix[[column]],
            contingency_matrix_sum_value=change_feature_params.get('contingency_matrix_sum_value', 5)  # Default to 5 if not provided
        )

        # Perform G-test and store the p-value
        p_value = ss.chi2_contingency(contingency_matrix, correction=True, lambda_='log-likelihood')[1]
        p_values.append(p_value)

    # Aggregate the p-values (e.g., using Fisher's method or another aggregation strategy)
    # Here, we use the geometric mean of p-values as an example
    aggregated_p_value = np.exp(np.mean(np.log(p_values)))

    return aggregated_p_value



def get_statistic_test_chi2_test(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_transition_matrix[[change_feature_params['process_feature']]], process_representation_detection_transition_matrix[[change_feature_params['process_feature']]])

    # Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
    return ss.chi2_contingency(contingency_matrix, correction=True)[1]


def get_statistic_test_strategy(process_representation_reference_transition_matrix_original, process_representation_detection_transition_matrix_original, change_feature_params_dict, control_flow_features=None, time_features=None, resource_features=None, data_features=None):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_transition_matrix = process_representation_reference_transition_matrix_original.copy()
    process_representation_detection_transition_matrix = process_representation_detection_transition_matrix_original.copy()


    # Loop to call the change features methods to aggregate all differences - Delta Vector
    change_features_statistic_test_dict = {}
    for change_feature, change_feature_params in change_feature_params_dict.items():

        try:
            change_features_statistic_test_dict[change_feature] = getattr(thismodule, "get_statistic_test_" + change_feature_params['method'])(process_representation_reference_transition_matrix
                                                                                                                                    , process_representation_detection_transition_matrix
                                                                                                                                    , change_feature_params)

        except Exception as e:
            print("Unknown change feature: ", change_feature)
            print("Error: ", e)

    return change_features_statistic_test_dict
