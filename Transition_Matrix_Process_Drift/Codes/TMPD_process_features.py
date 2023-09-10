# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import datetime

def get_feature_probability(TM_df_original, control_flow_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    TM_df = TM_df.reset_index()

    # Groupby activity_from and get sum
    TM_df_groupby = TM_df.groupby(["activity_from"], as_index=False)["frequency"].sum().set_index("activity_from")

    # Divide TM frequency column with groupby activity_from's sum
    TM_df[control_flow_feature] = TM_df.set_index("activity_from")[["frequency"]].div(TM_df_groupby).reset_index()["frequency"]

    return TM_df.set_index(['activity_from','activity_to'])[[control_flow_feature]]


def get_feature_causality(TM_df_original, control_flow_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()

    # Direct succession: x>y if for some case x is directly followed by y
    TM_df['direct_succession'] = np.where(TM_df['frequency']>0, 1, 0)
    
    # Opposite direction: if y>x
    TM_df_inverted = TM_df.reset_index()[['activity_from','activity_to','direct_succession']]
    TM_df_inverted.columns = ['activity_to','activity_from', 'opposite_direction']
    TM_df_inverted.set_index(['activity_from','activity_to'], inplace=True)
    TM_df = pd.merge(TM_df, TM_df_inverted, on=['activity_from', 'activity_to'], how='left')

    # Causality: x→y if x>y and not y>x
    TM_df[control_flow_feature] = np.where((TM_df['direct_succession']==1) & (TM_df['opposite_direction']!=1), 1, 0)

    return TM_df[[control_flow_feature]]

def get_feature_parallel(TM_df_original, control_flow_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()

    # Direct succession: x>y if for some case x is directly followed by y
    TM_df['direct_succession'] = np.where(TM_df['frequency']>0, 1, 0)
    
    # Opposite direction: if y>x
    TM_df_inverted = TM_df.reset_index()[['activity_from','activity_to','direct_succession']]
    TM_df_inverted.columns = ['activity_to','activity_from', 'opposite_direction']
    TM_df_inverted.set_index(['activity_from','activity_to'], inplace=True)
    TM_df = pd.merge(TM_df, TM_df_inverted, on=['activity_from', 'activity_to'], how='left')
    
    # Parallel: x||y if x>y and y>x
    TM_df[control_flow_feature] = np.where((TM_df['direct_succession']==1) & (TM_df['opposite_direction']==1), 1, 0)

    return TM_df[[control_flow_feature]]

def get_feature_choice(TM_df_original, control_flow_feature):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()

    # Direct succession: x>y if for some case x is directly followed by y
    TM_df['direct_succession'] = np.where(TM_df['frequency']>0, 1, 0)
    
    # Opposite direction: if y>x
    TM_df_inverted = TM_df.reset_index()[['activity_from','activity_to','direct_succession']]
    TM_df_inverted.columns = ['activity_to','activity_from', 'opposite_direction']
    TM_df_inverted.set_index(['activity_from','activity_to'], inplace=True)
    TM_df = pd.merge(TM_df, TM_df_inverted, on=['activity_from', 'activity_to'], how='left')
    
    # Choice: x#y if not x>y and not y>x and not x--->y
    TM_df[control_flow_feature] = np.where((TM_df['direct_succession']!=1) & (TM_df['opposite_direction']!=1), 1, 0)

    return TM_df[[control_flow_feature]]


def get_feature_avg_time(TM_df_original, log_transition_original, time_feature, time_feature_original):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()

    log_transition[time_feature] = (log_transition[time_feature_original + '_to'] - log_transition[time_feature_original + '_from']).dt.total_seconds() / 60

    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[time_feature].mean()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[time_feature]]


def get_feature_time_std(TM_df_original, log_transition_original, time_feature, time_feature_original):

    """...
    Args:
        'frequency_gtest' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
    """

    TM_df = TM_df_original.copy()
    log_transition = log_transition_original.copy()

    log_transition[time_feature] = (log_transition[time_feature_original + '_to'] - log_transition[time_feature_original + '_from']).dt.total_seconds() / 60
    
    TM_df_time_feature = log_transition.groupby(['activity_from','activity_to'])[time_feature].std()

    TM_df = TM_df.reset_index()

    TM_df = TM_df.join(TM_df_time_feature, on=['activity_from','activity_to'], how='left')

    return TM_df.set_index(['activity_from','activity_to'])[[time_feature]]
