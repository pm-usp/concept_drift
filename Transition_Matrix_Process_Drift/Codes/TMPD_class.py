# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import TMPD_utils
import TMPD_process_features
import TMPD_change_features
import TMPD_detection_tasks
import pm4py

from tqdm.notebook import tqdm_notebook
import time

import ruptures as rpt
from ruptures.metrics import precision_recall, meantime

from pm4py.objects.log.importer.xes import importer as xes_importer


class TMPD():
    """
    Transition Matrix Process Drift.
    Class for deal with Process Drift (or Concept Drift) in Process Mining using transition matrices as a unified data structure. 
    """

    def __init__(self, scenario: str = 'offline') -> None:
        """
        Initialize a TMPD instance.
        
        Args:
            scenario: Data scenario. 'online' (stream) or 'offline' (batch).
        """

        self.scenario = scenario
  
       
    def set_transition_log(self, event_log: pd.DataFrame, case_id: str, activity_key: str, 
                           timestamp_key: str, timestamp_format: str = 'infer', 
                           other_columns_keys: list[str] = []) -> None:
        """
        Set the transition log to be used for process drift detection.

        Args:
            event_log: The event log as a Pandas DataFrame.
            case_id: The name of the column containing the case ID.
            activity_key: The name of the column containing the activity name.
            timestamp_key: The name of the column containing the timestamp.
            timestamp_format: The format of the timestamp (default 'infer').
            other_columns_keys: A list of additional column names to include in the transition log (default []).
            
        Returns:
            None
        """

        self.case_id = case_id
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        self.other_columns_keys = other_columns_keys
        self.event_log = event_log.copy()
        

    def run_transition_log(self) -> None:
        """
        Computes a transition log from the event log and stores it in the object.
        A transition log records transitions between activities in each case of an event log.
        The transition log has one row per transition and the following columns:
            - case_id: ID of the case the transition belongs to.
            - activity_from: Activity the case was in before the transition.
            - timestamp_from: Timestamp of the last event before the transition.
            - other columns_from: Other columns of the last event before the transition.
            - activity_to: Activity the case moved to after the transition.
            - timestamp_to: Timestamp of the first event after the transition.
            - other columns_to: Other columns of the first event after the transition.
            - original_index: Index of the last event before the transition in the original event log.
            - transition_id: Unique identifier for the transition.
            - case_order: Integer indicating the position of the transition within the case.

        Args:
            None
            
        Returns:
            None
        """

        # event_log = self.event_log.copy()

        # transition_log = (event_log[[self.case_id, self.activity_key, self.timestamp_key] + self.other_columns_keys]
        #                   .rename(columns={self.case_id: 'case_id', self.activity_key: 'activity', self.timestamp_key: 'timestamp'})
        #                   .assign(timestamp=lambda df: pd.to_datetime(df['timestamp'], infer_datetime_format=True))
        #                   .pipe(lambda df: pd.concat([df[['case_id']], df.add_suffix('_from'), df.groupby('case_id').shift(-1).add_suffix('_to')], axis=1).drop(columns=['case_id_from']))
        #                   .dropna(subset=['activity_to'])
        #                   .reset_index(names='original_index')
        #                   .assign(transition_id=lambda df: df.index, case_order=lambda df: df.groupby('case_id').cumcount()))

        # self.transition_log = transition_log

        try:
        
            # Copy the event log to avoid modifying it
            event_log = self.event_log.copy()

            # Select the columns needed for the transition log and rename them
            event_log = event_log[[self.case_id, self.activity_key, self.timestamp_key] + self.other_columns_keys]
            event_log = event_log.rename(columns={self.case_id:'case_id', self.activity_key:'activity', self.timestamp_key:'timestamp'})

            # Convert timestamp column to datetime format
            if self.timestamp_format == 'infer':
                event_log['timestamp'] = pd.to_datetime(event_log['timestamp'], infer_datetime_format=True)
            elif self.timestamp_format is not None:
                event_log['timestamp'] = pd.to_datetime(event_log['timestamp'], format=self.timestamp_format)


            # Create the transition log
            transition_log = pd.concat([event_log[['case_id']], event_log.add_suffix('_from'), event_log.groupby('case_id').shift(-1).add_suffix('_to')], axis=1).drop(columns=['case_id_from'])
            transition_log = transition_log.dropna(subset = ['activity_to'])
            transition_log = transition_log.reset_index(names='original_index') 
            transition_log["transition_id"] = transition_log.index
            transition_log['case_order'] = transition_log.groupby('case_id').cumcount()

            # Store the transition log in the object
            self.transition_log = transition_log

        except Exception as e:
                print("Error in run_transition_log: ", e)


    def get_transition_log(self) -> pd.DataFrame:
        """
        Returns the current transition log that has been generated by `run_transition_log()` and stored in the object.

        Args:
            None

        Returns:
            transition_log: A Pandas DataFrame containing the transition log with the following columns:
                            - 'case_id': The case identifier
                            - 'activity_from': The activity at the start of the transition
                            - 'activity_to': The activity at the end of the transition
                            - 'timestamp_from': The timestamp of the start of the transition
                            - 'timestamp_to': The timestamp of the end of the transition
                            - 'other columns': The other columns from the original event log
                            - 'transition_id': A unique identifier for each transition
                            - 'case_order': The order of each transition within its case
        """

        return self.transition_log

        
    def set_windowing_strategy(self, window_size_mode: str = "Fixed", window_size: int = 100,
                           window_size_max: int = 150, window_size_min: int = 50,
                           window_ref_mode: str = "Fixed", overlap: bool = True, sliding_step: int = 50,
                           continuous: bool = True, gap_size: int = 0) -> None:
        """
        Set the windowing strategy to be used in the subsequent analysis.

        Args:
        - window_size_mode (str): The mode to be used for determining the window size. Defaults to "Fixed".
        - window_size (int): The fixed window size to be used. Ignored if window_size_mode is not "Fixed". Defaults to 100.
        - window_size_max (int): The maximum window size to be used. Ignored if window_size_mode is not "Adaptive". Defaults to 150.
        - window_size_min (int): The minimum window size to be used. Ignored if window_size_mode is not "Adaptive". Defaults to 50.
        - window_ref_mode (str): The mode to be used for determining the reference window. "Fixed" reference window will be the first window. Defaults to "Fixed".
        - overlap (bool): Whether or not to allow overlapping windows. Defaults to True.
        - sliding_step (int): The sliding step size to be used. Ignored if overlap is False. Defaults to 50.
        - continuous (bool): Whether or not to allow continuous windows. Defaults to True.
        - gap_size (int): The maximum allowed gap size between events within a window. Ignored if continuous is True. Defaults to 0.

        Returns:
            None
        """

        self.window_size_mode = window_size_mode
        self.window_size = window_size
        self.window_size_max = window_size_max
        self.window_size_min = window_size_min
        self.window_ref_mode = window_ref_mode
        self.overlap = overlap
        self.sliding_step = sliding_step
        self.continuous = continuous
        self.gap_size = gap_size


    def run_windowing_strategy(self) -> None:
        """
        This function applies the configured windowing strategy to the transition log, generating the windows index. 
        The windows index is stored in the object as a dictionary with the following structure:
        
        {window_index: {'start': <window start index>, 'end': <window end index>}}

        The windows_index_dict attribute is a dictionary that stores the window start and end indexes.
        The keys of the dictionary correspond to the index of each window, and the values are themselves 
        dictionaries with the following keys:
        
            'start': int - The index of the first event of the window
            'end': int - The index of the last event of the window

        Args:
            None

        Returns:
            None
        """

        # # Create a new empty dataframe to hold the windows information
        # windows_df = pd.DataFrame()

        # # Check the windowing mode and generate the window start and end indexes
        # if self.window_size_mode == 'Fixed':
        #     # If the window size is fixed, calculate the start and end indexes for each window
        #     start_idx = range(0, len(self.transition_log), self.sliding_step) if self.overlap else range(0, len(self.transition_log), self.window_size + self.gap_size)
        #     windows_df['start'] = start_idx
        #     windows_df['end'] = windows_df['start'] + self.window_size
        #     # Discard any incomplete windows that go beyond the end of the log
        #     windows_df = windows_df[windows_df['end'] <= len(self.transition_log)]
        # else:
        #     # TODO: Handle other windowing modes
        #     raise NotImplementedError("Windowing mode not implemented yet")

        # # Add a unique window index to each window
        # windows_df['window_index'] = windows_df.index

        # # Convert the windows dataframe to a dictionary and store it in the object
        # self.windows_index_dict = windows_df.to_dict('index')


        # Create a new empty dataframe to hold the windows information
        windows_df = pd.DataFrame()

        # Get the windows index based on the configured windowing strategy
        if self.window_size_mode == 'Fixed':

            # If there is no overlap between windows
            if self.overlap == False:
                # If the window is continuous
                if self.continuous == True:
                    windows_df['start'] = range(0, len(self.transition_log), self.window_size)
                # If the window is not continuous, a gap size is added between each window
                else:
                    windows_df['start'] = range(0, len(self.transition_log), self.window_size + self.gap_size)
            # If there is overlap between windows, a sliding step is used to move the window
            else:
                windows_df['start'] = range(0, len(self.transition_log), self.sliding_step)

            # Calculate the end index of each window
            windows_df['end'] = windows_df['start'] + self.window_size

            # Remove windows that are incomplete
            windows_df = windows_df[windows_df['end'] <= len(self.transition_log)]

        else:
            # TODO: Handle other windowing modes
            raise NotImplementedError("Windowing mode not implemented yet")

        # Add a unique window index to each window
        windows_df['window_index'] = windows_df.index

        # Convert the windows dataframe to a dictionary and store it in the object
        self.windows_df_dict = windows_df.to_dict('index') 


    def get_windowing_strategy(self) -> dict:
        """
        Returns the current windows index dictionary generated by the windowing strategy.

        Args:
            None

        Returns:
            dict: A dictionary containing the window index, start and end indices of each window.
        """

        return self.windows_df_dict
        # return 'window_size_mode: ' + self.window_size_mode + ', window_size: ' + str(self.window_size) + ', window_size_max: ' + str(self.window_size_max) + \
        #     ', window_size_min: ' + str(self.window_size_min) + ', window_ref_mode: ' + self.window_ref_mode + ', overlap: ' + self.overlap + ', sliding_step: ' + \
        #         str(self.sliding_step) + ', continuous: ' + self.continuous + ', gap_size: ' + str(self.gap_size)



    def set_process_representation(self, threshold_anomaly=0
                                    # , control_flow_features=['frequency', 'probability']
                                    , control_flow_features={'frequency', 'probability'}
                                    , time_features=None
                                    , resource_features=None
                                    , data_features=None) -> None: 
        """
        Initializes the transition matrix data structure to represent the process using the given features list.

        Args:
            threshold_anomaly (int, optional): Filter for anomaly transitions (few frequency). 
                If less than 1, it is considered a percentage threshold; otherwise, it is a frequency threshold.
            control_flow_features (dict, optional): List of control-flow features to be used. Default features list is 
                ['frequency', 'probability']. Possible features are: 
                    'frequency'
                    'probability'
                    'causality'
                    'parallel'
                    'choice'
            time_features (dict, optional): Dictionary of time-related features to be used. Default is an empty dictionary.
            resource_features (dict, optional): Dictionary of resource-related features to be used. Default is an empty dictionary.
            data_features (dict, optional): Dictionary of data-related features to be used. Default is an empty dictionary.
        
        Returns:
            None
        """

        self.threshold_anomaly = threshold_anomaly
        self.control_flow_features = control_flow_features
        self.time_features = time_features if time_features is not None else {}
        self.resource_features = resource_features if resource_features is not None else {}
        self.data_features = data_features if data_features is not None else {}



    def run_process_representation(self, transition_log_sample_original) -> None:
        """
        Generates a process representation based on the provided transition log.

        Args:
            transition_log_sample_original (DataFrame): Original transition log as a Pandas DataFrame.

        Returns:
            None
        """

        # Create a copy of the original transition log
        transition_log = transition_log_sample_original.copy()
        
        # Create a standard Transition Matrix (TM) process representation using frequency and percentual features for anomaly filter
        process_representation_df = pd.crosstab(transition_log["activity_from"], transition_log["activity_to"], normalize=False).stack().reset_index().rename(columns={0: "frequency"})
        process_representation_df["percentual"] = process_representation_df["frequency"]/process_representation_df["frequency"].sum()
        process_representation_df = process_representation_df.sort_values(by=['activity_from', 'activity_to'], ascending=[True, True]).set_index(['activity_from','activity_to'])

        # Set transitions with lower percentage or frequency than the threshold to zero (anomaly filter)
        if self.threshold_anomaly < 1 :
            process_representation_df.iloc[process_representation_df['percentual'] <= self.threshold_anomaly] = 0
        else:
            process_representation_df.iloc[process_representation_df['frequency'] <= self.threshold_anomaly] = 0

        # Remove all transition with zero frequency
        process_representation_df = process_representation_df[process_representation_df["frequency"]>0]

        # Initialize a dictionary to store process features
        process_features_dict = {
            "frequency": process_representation_df[["frequency"]]
            , "percentual": process_representation_df[["percentual"]]
        }

        # Add control-flow features
        for control_flow_feature in self.control_flow_features - {'frequency', 'percentual'}:
            try:
                process_features_dict[control_flow_feature] = getattr(TMPD_process_features, "get_feature_" + control_flow_feature)(process_representation_df, control_flow_feature)
            except Exception as e:
                print("Error in control_flow_feature representation: ", control_flow_feature)
                print("Error: ", e)

        # Add time features
        for time_feature in self.time_features:
            try:
                process_features_dict[time_feature] = getattr(TMPD_process_features, "get_feature_" + time_feature)(process_representation_df, transition_log, time_feature, self.time_features[time_feature])
            except Exception as e:
                print("Error in time_feature representation: ", time_feature)
                print("Error: ", e)

        # Add resource features
        for resource_feature in self.resource_features:
            try:
                process_features_dict[resource_feature] = getattr(TMPD_process_features, "get_feature_" + resource_feature)(process_representation_df, transition_log, resource_feature, self.resource_features[resource_feature])
            except Exception as e:
                print("Error in resource_feature representation: ", resource_feature)
                print("Error: ", e)

        # Add data features
        for data_feature in self.data_features:
            try:
                process_features_dict[data_feature] = getattr(TMPD_process_features, "get_feature_" + data_feature)(process_representation_df, transition_log, data_feature, self.data_features[data_feature])
            except Exception as e:
                print("Error in data_feature representation: ", data_feature)
                print("Error: ", e)

        # Merge all features transition matrices
        process_representation_df = reduce(lambda  left,right: pd.merge(left, right, on=['activity_from', 'activity_to'], how='outer'), process_features_dict.values()).fillna(0)

        # Keep only the defined features
        self.process_representation_df = process_representation_df[list(self.control_flow_features) + list(self.time_features.keys()) + list(self.resource_features.keys()) + list(self.data_features.keys())]


    def get_process_representation(self) -> pd.DataFrame:
        """
        Retrieves the process representation generated by the run_process_representation method.
        The returned DataFrame contains the process representation with the selected control flow features
        (frequency, percentual) and any additional time, resource, or data features specified during
        the process representation setup.

        Args:
            None

        Returns:
            DataFrame: Process representation as a Pandas DataFrame.
        """

        return self.process_representation_df
    

    def set_change_representation(self, change_features_strategy_dict) -> None:
        """
        Sets the change representation strategies for analyzing feature changes in the process.

        Args:
            change_features_strategy_dict (dict): Dictionary specifying the change representation strategies.
                The dictionary should have the following structure:
                {
                    'delta_matrix_strategy': {
                        'feature_name': {
                            'process_feature': 'process_feature_name',
                            'method': 'aggregation|percentage|aggregation_weight',
                            'agg_function': 'sum' (required for method='aggregation'),
                            'weight_feature': 'weight_feature_name' (required for method='aggregation_weight')
                        },
                        ...
                    },
                    'statistic_test_strategy': {
                        'feature_name': {
                            'process_feature': 'process_feature_name',
                            'method': 'g_test|cramers_v',
                            'contingency_matrix_sum_value': 'sum_value',
                            'remove_zeros': 'True|False'
                        },
                        ...
                    },
                    ...
                }
                The 'delta_matrix_strategy' and 'statistic_test_strategy' are example keys representing different change
                representation strategies. You can add more strategies as needed. Each strategy contains multiple features
                with their corresponding process feature, method, and additional parameters if required.

                Available methods for 'delta_matrix_strategy':
                - 'aggregation': Computes the change by aggregating the process feature values.
                - 'percentage': Computes the change as a percentage.
                - 'aggregation_weight': Computes the change by aggregating the process feature values using a weight feature.

                Available methods for 'statistic_test_strategy':
                - 'g_test': Performs the G-test statistical test to measure the association between features.
                - 'cramers_v': Computes Cramer's V statistic to measure the association between features.

        Note:
            The change representation strategies define how feature changes in the process are quantified and analyzed.
            The strategies can include various statistical methods or aggregation techniques to capture different aspects of change.

        Returns:
            None
        """
        self.change_features_strategy_dict = change_features_strategy_dict

        # change_features_strategy_dict = {
        # 'delta_matrix_strategy': 
        #     {
        #         'frequency_delta' : {'process_feature':'frequency', 'method':'aggregation', 'agg_function' : 'sum'}
        #         , 'probability_delta' : {'process_feature':'probability', 'method':'aggregation', 'agg_function' : 'sum'}
        #         , 'frequency_delta_percentage' : {'process_feature':'frequency', 'method':'percentage'}
        #         , 'prob_freq_delta_weight' : {'process_feature':'probability', 'method':'aggregation_weight', 'agg_function' : 'sum', 'weight_feature' : 'frequency'}
        #     }
        # , 'statistic_test_strategy' : 
        #     {
        #         'frequency_gtest_pvalue' : {'process_feature':'frequency', 'method':'g_test', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        #         , 'frequency_cramersv' : {'process_feature':'frequency', 'method':'cramers_v', 'contingency_matrix_sum_value' : '5', 'remove_zeros':'True'}
        #     }
        # }
        

    def run_change_representation(self) -> None:

        """
        Runs the change representation analysis based on the defined strategies and the process representation.

        Args:
            transition_log (DataFrame): Event log as a Pandas DataFrame.

        Returns:
            None
        """

        # Initiating the change representation dictionary
        change_representation_dict = self.windows_df_dict.copy()

        # Iterate over each window index and window information
        for window_index, window_info in self.windows_df_dict.items():

            # Run process representation with the current detection window
            self.run_process_representation(self.transition_log.iloc[window_info['start'] : window_info['end']])
            process_representation_detection_window_df = self.get_process_representation()

            # Check if it's the first window
            if window_index > 0:

                # Add information about which window was used as a reference
                change_representation_dict[window_index].update({'reference_window_index' : str(reference_window_index)})
                
                # Loop through the change feature strategies and their corresponding parameters
                for change_feature_strategy, change_feature_params_dict in self.change_features_strategy_dict.items():
                    try:
                        # Call the corresponding change feature method and update the results in the dictionary
                        change_representation_dict[window_index].update(
                            getattr(TMPD_change_features, "get_" + change_feature_strategy)(
                                process_representation_reference_window_df
                                , process_representation_detection_window_df
                                , change_feature_params_dict
                            )
                        )
                    except Exception as e:
                        print("Error in run_change_representation: ", change_feature_strategy)   
                        print("Error: ", e)
                
                # If the window reference mode is Sliding, the current detection window becomes the reference window, otherwise the reference window remains the first window. 
                if self.window_ref_mode == "Sliding":
                    process_representation_reference_window_df = process_representation_detection_window_df.copy()
                    reference_window_index = window_index

            # If it's the first window, the detection window just becomes the reference window 
            else:
                process_representation_reference_window_df = process_representation_detection_window_df.copy()
                reference_window_index = window_index

        # Create a DataFrame from the change representation dictionary
        self.change_representation_df = pd.DataFrame.from_dict(change_representation_dict, orient='index') #.fillna(0)


    def get_change_representation(self) -> pd.DataFrame:
        """
        Returns the change representation generated by the run_change_representation method.

        Args:
            transition_log (DataFrame): Event log as a Pandas DataFrame.
        
        Returns:
            DataFrame: Change representation DataFrame.
        """
        return self.change_representation_df


    def set_detection_task(self, detection_task_strategy_dict) -> None:
        """
        Sets the detection task strategy dictionary.

        Args:
            detection_task_strategy_dict (dict): Dictionary defining the detection task strategies and their parameters.
                The dictionary should have the following structure:
                {
                    'time_series_strategy': {
                        'strategy_name_1': {
                            'change_features': [list of change features],
                            'method': detection method,
                            'parameter_name_1': parameter_value_1,
                            'parameter_name_2': parameter_value_2,
                            ...
                        },
                        ...
                    },
                    'threshold_strategy': {
                        'strategy_name_1': {
                            'change_features': [list of change features],
                            'method': detection method,
                            'parameter_name_1': parameter_value_1,
                            'parameter_name_2': parameter_value_2,
                            ...
                        },
                        ...
                    },
                    ...
                }

                The dictionary is organized into different strategy groups, such as 'time_series_strategy' and
                'threshold_strategy'. Within each strategy group, multiple detection strategies can be defined.
                Each strategy is identified by a unique strategy name and has the following parameters:
                - 'change_features': A list of change features to consider for the detection. The change features should be
                one or more feature names defined in the `set_change_representation` function.
                - 'method': The detection method to use.
                - Additional parameters can be added based on the specific requirements of the detection method.

                Example detection methods:
                - 'cpd_pelt': Change point detection using the Pruned Exact Linear Time (PELT) algorithm.
                - 'comparison_operator': Change detection based on comparison operators.

                Additional parameters can be added based on the specific requirements of the detection method. Here are some examples:
                - For 'cpd_pelt' method:
                    - 'smooth': Smoothing parameter for the detection algorithm.
                    - 'model': Model to use for change point detection (e.g., 'rbf', 'linear', 'normal', etc.).
                    - 'cost': Cost function for the detection algorithm.
                    - 'min_size': Minimum size of change points to detect.
                    - 'jump': Jump parameter for the detection algorithm.

                - For 'comparison_operator' method:
                    - 'operator': Comparison operator to use (e.g., 'le', 'ge', 'eq', etc.).
                    - 'threshold_value': Threshold value for the comparison.

                You can customize the parameters based on your specific detection requirements.

        Returns:
            None

        """

        self.detection_task_strategy_dict = detection_task_strategy_dict

        # detection_task_strategy_dict = {
        #     'time_series_strategy': 
        #     {
        #         'cpd_frequency_delta' : {'change_features':['frequency_delta'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
        #         , 'cpd_prob_freq_delta' : {'change_features':['prob_freq_delta_weight'], 'method':'cpd_pelt', 'model' : 'rbf', 'cost' : 'rpt.costs.CostRbf()', 'min_size' : '1', 'jump' : '1', 'smooth' : '3'}
        #     }
        #     , 'threshold_strategy' : 
        #     {
        #         'gtest_frequency' : {'change_features':['frequency_gtest_pvalue'], 'method':'comparison_operator', 'operator' : 'le', 'threshold_value' : '0.025', 'smooth' : '3'}
        #         , 'fixed_frequency_delta_percentage' : {'change_features':['frequency_delta_percentage'], 'method':'comparison_operator', 'operator' : 'ge', 'threshold_value' : '0.05', 'smooth' : '3'}
        #     }
        # }


    def run_detection_task(self) -> None:
        """
        Runs the detection task based on the configured detection strategies.

        Args:
            None
        
        Returns:
            None
        """

        # Initiating the detection task result
        detection_task_result_dict = {}

        # Loop to call the detection strategies
        for detection_task_strategy, detection_task_params_dict in self.detection_task_strategy_dict.items():
            try:
                # Call detection task function based on strategy and update results in the dictionary
                detection_task_result_dict[detection_task_strategy] = getattr(
                    TMPD_detection_tasks, "get_" + detection_task_strategy)(self, detection_task_params_dict)

            except Exception as e:
                print("Error in run_detection_task: ", detection_task_strategy)  
                print("Error: ", e)

        # Prepare result dict to dataframe
        detection_task_result_df = pd.DataFrame.from_dict(detection_task_result_dict, orient='index')
        detection_task_result_df = detection_task_result_df.reset_index(names='detection_strategy').melt(
            id_vars=['detection_strategy'], var_name='detection_feature', value_name='detection_results')

        # Drop rows with NaN values and reset index
        self.detection_task_result_df = detection_task_result_df.dropna(axis=0).reset_index(drop=True)


    def get_detection_task(self) -> pd.DataFrame:

        """
        Retrieves the results of the detection task.

        Args:
            None
        
        Returns:
            DataFrame: Detection task results as a Pandas DataFrame.
        """

        return self.detection_task_result_df
