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

def get_delta_matrix(process_representation_reference_window_df_original, process_representation_detection_window_df_original):

    """Get the difference between the process representation reference window and the rocess representation detection window - Delta Matrix
    Args:
        log_transition (DataFrame): Event log as Pandas Dataframe. 
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()

    delta_matrix = abs(process_representation_reference_window_df.subtract(process_representation_detection_window_df, fill_value=0))

    return delta_matrix


def get_delta_matrix_aggregation(delta_matrix, process_representation_reference_window_df, process_representation_detection_window_df, change_feature_params):
    return delta_matrix[change_feature_params['process_feature']].agg(change_feature_params['agg_function'])


def get_delta_matrix_percentage(delta_matrix, process_representation_reference_window_df, process_representation_detection_window_df, change_feature_params):
    """Get the proportion of difference to maximum difference possible
    Args:
        log_transition (DataFrame): Event log as Pandas Dataframe. 
    """

    total_delta = process_representation_reference_window_df[change_feature_params['process_feature']].sum() + process_representation_detection_window_df[change_feature_params['process_feature']].sum()

    return delta_matrix[change_feature_params['process_feature']].sum()/total_delta


def get_delta_matrix_aggregation_weight(delta_matrix, process_representation_reference_window_df, process_representation_detection_window_df, change_feature_params):

    return delta_matrix[change_feature_params['process_feature']].multiply(delta_matrix[change_feature_params['weight_feature']].values, axis="index", fill_value=0).agg(change_feature_params['agg_function'])


def get_delta_matrix_strategy(process_representation_reference_window_df_original, process_representation_detection_window_df_original, change_feature_params_dict):

    """Get the difference between the rocess representation reference window and the rocess representation detection window - Delta Matrix
    Args:
        log_transition (DataFrame): Event log as Pandas Dataframe. 
        {
                'frequency_delta' : {'process_feature':'frequency', 'method':'aggregation', 'agg_function' : 'sum'}
                , 'probability_delta' : {'process_feature':'probability', 'method':'aggregation', 'agg_function' : 'sum'}
                , 'frequency_delta_percentage' : {'process_feature':'frequency', 'method':'percentage'}
                , 'prob_freq_delta' : {'process_feature':'probability', 'method':'aggregation_weight', 'agg_function' : 'sum', 'weight_feature' : 'frequency'}
            }
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()

    # Call delta matrix function
    delta_matrix = get_delta_matrix(process_representation_reference_window_df, process_representation_detection_window_df)

    # Loop to call the change features methods to aggregate all differences - Delta Vector
    change_features_delta_matrix_dict = {}
    for change_feature, change_feature_params in change_feature_params_dict.items():

        try:
            change_features_delta_matrix_dict[change_feature] = getattr(thismodule, "get_delta_matrix_" + change_feature_params['method'])(delta_matrix
                                                                                                                                    , process_representation_reference_window_df
                                                                                                                                    , process_representation_detection_window_df
                                                                                                                                    , change_feature_params)

        except Exception as e:
            print("Unknown change feature: ", change_feature)
            print("Error: ", e)

    return change_features_delta_matrix_dict     

                    
def get_contingency_matrix(ditribution_reference, ditribution_detection):

    """...
    Args:
        log_transition (DataFrame): Event log as Pandas Dataframe. 
    """

    # Create the contingency table
    contingency_matrix = pd.merge(ditribution_reference, ditribution_detection
                                , left_index=True, right_index=True, how='outer', suffixes=('_refeference_window', '_detection_window')).fillna(0).astype(int)

    # Remove zeros 
    contingency_matrix = contingency_matrix.loc[(contingency_matrix!=0).any(axis=1)]

    # Add value to all  
    contingency_matrix += 5

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


def get_statistic_test_cramers_v(process_representation_reference_window_df_original, process_representation_detection_window_df_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_window_df[[change_feature_params['process_feature']]], process_representation_detection_window_df[[change_feature_params['process_feature']]])

    # Cramer's V corrected
    return cramers_corrected_stat(contingency_matrix)


def get_statistic_test_g_test(process_representation_reference_window_df_original, process_representation_detection_window_df_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_window_df[[change_feature_params['process_feature']]], process_representation_detection_window_df[[change_feature_params['process_feature']]])

    # Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
    return ss.chi2_contingency(contingency_matrix, correction=True, lambda_='log-likelihood')[1]


def get_statistic_test_chi2_test(process_representation_reference_window_df_original, process_representation_detection_window_df_original, change_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()

    # Call contingency matrix function
    contingency_matrix = get_contingency_matrix(process_representation_reference_window_df[[change_feature_params['process_feature']]], process_representation_detection_window_df[[change_feature_params['process_feature']]])

    # Chi2 or G-test (add: lambda_='log-likelihood' in chi2_contingency)
    return ss.chi2_contingency(contingency_matrix, correction=True)[1]


def get_statistic_test_strategy(process_representation_reference_window_df_original, process_representation_detection_window_df_original, change_feature_params_dict):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    process_representation_reference_window_df = process_representation_reference_window_df_original.copy()
    process_representation_detection_window_df = process_representation_detection_window_df_original.copy()


    # Loop to call the change features methods to aggregate all differences - Delta Vector
    change_features_statistic_test_dict = {}
    for change_feature, change_feature_params in change_feature_params_dict.items():

        try:
            change_features_statistic_test_dict[change_feature] = getattr(thismodule, "get_statistic_test_" + change_feature_params['method'])(process_representation_reference_window_df
                                                                                                                                    , process_representation_detection_window_df
                                                                                                                                    , change_feature_params)

        except Exception as e:
            print("Unknown change feature: ", change_feature)
            print("Error: ", e)

    return change_features_statistic_test_dict 
