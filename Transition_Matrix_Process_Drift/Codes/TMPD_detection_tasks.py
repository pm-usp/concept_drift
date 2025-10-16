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
from scipy.signal import find_peaks

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
    """
    Applies the PELT algorithm for change point detection on the specified feature.

    Args:
        change_representation_df (DataFrame): Change representation data.
        detection_feature_params (dict): Parameters for the PELT algorithm, including model, cost, and smoothing.

    Returns:
        list: Detected change points.
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
        pen = float(detection_feature_params['pen'])
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
    """
    Executes time series-based detection strategies.

    Args:
        detection_task_params_dict (dict): Dictionary of detection strategies and their parameters.

    Returns:
        dict: Results of the detection strategies.
    """

    change_representation_df = self.change_representation_df.copy()

    # 
    detection_strategy_result_dict = {}
    for detection_feature, detection_feature_params in detection_task_params_dict.items():
        try:
            params = detection_feature_params.copy()  # Make a copy to avoid modifying the original
            
            # Choose the appropriate method based on window_ref_mode
            if params['method'] == 'cpd_pelt':
                if self.window_ref_mode == "Sliding":
                    method_to_use = 'get_peak_detection'
                else:
                    method_to_use = 'get_cpd_pelt'
            else:
                method_to_use = 'get_' + params['method']
            
            # Call the selected method
            detection_strategy_result_dict[detection_feature] = getattr(thismodule, method_to_use)(self, change_representation_df, params)

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


def get_peak_detection(self, change_representation_df, detection_feature_params):
    """
    Applies peak detection using scipy's find_peaks for detecting changes in sliding window mode.
    
    Args:
        change_representation_df (DataFrame): Change representation data.
        detection_feature_params (dict): Parameters for peak detection, including height, distance, and smoothing.
    
    Returns:
        list: Detected peak locations representing change points.
    """
    # Get the defined change feature(s) vector and apply smoothing
    signals = change_representation_df[detection_feature_params['change_features']].rolling(
        window=int(detection_feature_params['smooth'])).mean().dropna()
    
    # Convert to numpy array if multiple features
    if len(detection_feature_params['change_features']) > 1:
        signal_array = signals.mean(axis=1).values
    else:
        signal_array = signals.values.flatten()
    
    # Set default parameters for find_peaks
    try: 
        height = float(detection_feature_params.get('height', 0))
    except:
        height = None
        
    try:
        distance = int(detection_feature_params.get('distance', self.window_size/2))
    except:
        distance = int(self.window_size/2)
        
    try:
        prominence = float(detection_feature_params.get('prominence', None))
    except:
        prominence = None
    
    # Find peaks in the signal
    try:
        peaks, _ = find_peaks(signal_array, 
                             height=height,
                             distance=distance,
                             prominence=prominence)
        
        # Adjust indices for smoothing window and convert to list
        peaks = [int(p) + int(detection_feature_params['smooth']) for p in peaks]
        
        # Add the last point of the signal
        last_point = len(change_representation_df)
        if not peaks or peaks[-1] != last_point:
            peaks.append(last_point)
            
    except Exception as e:
        print("Error in get_peak_detection: ", e)
        peaks = []
    
    return peaks


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