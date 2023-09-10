# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import sys
thismodule = sys.modules[__name__]
import ruptures as rpt
import operator

def dingOptimalNumberOfPoints(algo):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """
    point_detection_penalty = 15
    x_lines = algo.predict(pen=point_detection_penalty)
    

    while point_detection_penalty >= len(x_lines):
        point_detection_penalty -= 1
        x_lines = algo.predict(pen=point_detection_penalty)

    if len(x_lines) > 15:
        x_lines = x_lines[-1:]

    return x_lines


def get_cpd_pelt(self, change_representation_df, detection_feature_params):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    # Get the defined change feature(s) vector and apply a 'smoothing'
    signals = change_representation_df[detection_feature_params['change_features']].rolling(window=int(detection_feature_params['smooth'])).mean().dropna()
    
    # Define pelt parameters
    try: model = detection_feature_params['model']
    except: model = 'rbf'

    try: cost = detection_feature_params['cost']
    except: cost = 'rpt.costs.CostRbf()'

    try: min_size = detection_feature_params['min_size']
    except: 
        if self.overlap == True:
            min_size = str(int(self.window_size/self.sliding_step)+1)
        else: 
             min_size = '1'
  
    try: jump = detection_feature_params['jump']
    except: jump = '1'

    # fit pelt algorithm
    pelt_algo = rpt.Pelt(model = model
        , min_size = int(min_size)
        , jump = int(jump)
        , custom_cost = cost
    ).fit(signals)

    # predict pelt algorithm 
    try: 
        pen = detection_feature_params['pen']
        try: result = pelt_algo.predict(pen=pen)
        except Exception as e:
            print("Error in get_cpd_pelt: ", e)
            result = []
    except: 
        try: result = dingOptimalNumberOfPoints(pelt_algo)
        except Exception as e:
            print("Error in get_cpd_pelt: ", e)
            result = []

    # Smooth correction
    result = [item + int(detection_feature_params['smooth']) for item in result]

    return result


def get_time_series_strategy(self, detection_task_params_dict):

    """...
    Args:
        detection_task_strategy_dict = {
            'time_series_strategy': 
            {
                'cpd_frequency_delta' : {'change_features':['frequency_delta'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
                , 'cpd_prob_freq_delta' : {'change_features':['prob_freq_delta_weight'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
            }
            , 'threshold_strategy' : 
            {
                'gtest_frequency' : {'change_features':['frequency_gtest_pvalue'], 'method':'comparison_operator', 'operator' : 'le', 'threshold_value' : '0.025', 'smooth' : '3'}
                , 'fixed_frequency_delta_percentage' : {'change_features':['frequency_delta_percentage'], 'method':'comparison_operator', 'operator' : 'ge', 'threshold_value' : '0.05', 'smooth' : '3'}
            }
        }

    """

    change_representation_df = self.change_representation_df.copy()

    # 
    detection_strategy_result_dict = {}
    for detection_feature, detection_feature_params in detection_task_params_dict.items():

        try:
            detection_strategy_result_dict[detection_feature] = getattr(thismodule, "get_" + detection_feature_params['method'])(self, change_representation_df, detection_feature_params)

        except Exception as e:
            print("Error in get_time_series_strategy: ", detection_feature)
            print("Error: ", e)

    return detection_strategy_result_dict 

def get_comparison_operator(change_representation_df, detection_feature_params):

    """...
    Args:
        {
            'gtest_frequency' : {'change_features':['frequency_gtest_pvalue'], 'method':'comparison_operator', 'operator' : 'le', 'threshold_value' : '0.025', 'smooth' : '3'}
            , 'fixed_frequency_delta_percentage' : {'change_features':['frequency_delta_percentage'], 'method':'comparison_operator', 'operator' : 'ge', 'threshold_value' : '0.05', 'smooth' : '3'}
        }
    """

    # Get the defined change feature(s) vector
    signals = change_representation_df[detection_feature_params['change_features']]

    # Apply operator comparison
    signals = getattr(signals, detection_feature_params['operator'])(float(detection_feature_params['threshold_value'])).astype(int)

    # Apply a 'smoothing'
    signals = signals.rolling(window=int(detection_feature_params['smooth'])).mean().dropna()
    signals = pd.DataFrame(np.where(signals==1, 1, np.where(signals==0, 0, np.nan)), columns=detection_feature_params['change_features']).ffill(axis = 0)

    # Check changes
    result = np.abs(signals.diff(1)).eq(1) 

    # Correct smooth
    result =  result[result[detection_feature_params['change_features'][0]]].index #- (int(detection_feature_params['smooth']) - 1)

    # return indexes of changes
    return result.tolist() + [len(change_representation_df)]
    


def get_threshold_strategy(self, detection_task_params_dict):

    """...
    Args:
        detection_task_strategy_dict = {
            'time_series_strategy': 
            {
                'cpd_frequency_delta' : {'change_features':['frequency_delta'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
                , 'cpd_prob_freq_delta' : {'change_features':['prob_freq_delta_weight'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
            }
            , 'threshold_strategy' : 
            {
                'gtest_frequency' : {'change_features':['frequency_gtest_pvalue'], 'method':'comparison_operator', 'operator' : 'le', 'threshold_value' : '0.025', 'smooth' : '3'}
                , 'fixed_frequency_delta_percentage' : {'change_features':['frequency_delta_percentage'], 'method':'comparison_operator', 'operator' : 'ge', 'threshold_value' : '0.05', 'smooth' : '3'}
            }
        }

    """
    
    change_representation_df = self.change_representation_df.copy()

    # 
    detection_strategy_result_dict = {}
    for detection_feature, detection_feature_params in detection_task_params_dict.items():

        try:
            detection_strategy_result_dict[detection_feature] = getattr(thismodule, "get_" + detection_feature_params['method'])(change_representation_df, detection_feature_params)

        except Exception as e:
            print("Error in get_threshold_strategy: ", detection_feature)
            print("Error: ", e)

    return detection_strategy_result_dict 