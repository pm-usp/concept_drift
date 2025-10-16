"""
TMPD_class.py

Main class for Transition Matrix Process Drift (TMPD) library.
Provides unified process drift detection, change analysis, and process mining utilities
using transition matrices as the core data structure.

Author: Antonio Carlos Meira Neto
License: MIT
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from functools import reduce
import TMPD_utils
import TMPD_process_features
import TMPD_change_features
import TMPD_detection_tasks
import TMPD_understanding_tasks
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from tqdm.notebook import tqdm_notebook
import time
import ruptures as rpt
from ruptures.metrics import precision_recall, meantime
from typing import Optional, List, Dict, Any

class TMPD:
    """
    Transition Matrix Process Drift (TMPD).
    Main class for process drift detection and analysis in process mining using transition matrices.
    """

    def __init__(self, scenario: str = 'offline') -> None:
        """
        Initialize a TMPD instance.
        Args:
            scenario (str): Data scenario. 'online' (stream) or 'offline' (batch). Default is 'offline'.
        """
        self.scenario = scenario

    def set_transition_log(self, event_log: pd.DataFrame, case_id: str, activity_key: str, 
                           timestamp_key: str, timestamp_format: str = 'infer', 
                           other_columns_keys: Optional[List[str]] = None) -> None:
        """
        Set the transition log to be used for process drift detection.
        Args:
            event_log (pd.DataFrame): The event log as a Pandas DataFrame.
            case_id (str): The name of the column containing the case ID.
            activity_key (str): The name of the column containing the activity name.
            timestamp_key (str): The name of the column containing the timestamp.
            timestamp_format (str): The format of the timestamp (default 'infer').
            other_columns_keys (List[str], optional): Additional column names to include in the transition log.
        """
        self.case_id = case_id
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.timestamp_format = timestamp_format
        self.other_columns_keys = other_columns_keys if other_columns_keys is not None else []
        self.event_log = event_log.copy()
        self.alpha_relations = None

    def run_transition_log(self, replace_spaces_in_activities: bool = True) -> None:
        """
        Computes a transition log from the event log and stores it in the object.
        Optionally replaces spaces with underscores in activity columns.
        A transition log records transitions between activities in each case of an event log.
        """
        try:
            event_log = self.event_log.copy()
            # Select and rename columns
            event_log = event_log[[self.case_id, self.activity_key, self.timestamp_key] + self.other_columns_keys].copy()
            event_log.columns = ['case_id' if col == self.case_id else 'activity' if col == self.activity_key else 'timestamp' if col == self.timestamp_key else col for col in event_log.columns]
            # Optionally replace spaces with underscores in activity column
            if replace_spaces_in_activities:
                event_log['activity'] = event_log['activity'].astype(str).str.replace(' ', '_', regex=False)
            # Convert timestamp column to datetime format
            if self.timestamp_format == 'infer':
                event_log['timestamp'] = pd.to_datetime(event_log['timestamp'], infer_datetime_format=True)
            elif self.timestamp_format is not None:
                event_log['timestamp'] = pd.to_datetime(event_log['timestamp'], format=self.timestamp_format)
            # Create an id based on the order of the event in the raw event log
            event_log["event_order"] = event_log.index
            # Add Start and End activities if not present
            event_log = TMPD_utils.add_start_end_activities(event_log=event_log, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp")
            # Create the transition log
            transition_log = pd.concat([
                event_log[['case_id']],
                event_log.add_suffix('_from'),
                event_log.groupby('case_id').shift(-1).add_suffix('_to')
            ], axis=1).drop(columns=['case_id_from'])
            transition_log = transition_log.dropna(subset=['activity_to'])
            transition_log['case_order'] = transition_log.groupby('case_id').cumcount()
            transition_log.sort_values(by=['event_order_from', 'event_order_to'], inplace=True)
            transition_log.reset_index(drop=True, inplace=True) 
            transition_log["transition_id"] = transition_log.index
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
            window_size_mode (str): Mode for determining window size. Defaults to "Fixed".
            window_size (int): Fixed window size. Ignored if not "Fixed". Defaults to 100.
            window_size_max (int): Max window size for adaptive mode. Defaults to 150.
            window_size_min (int): Min window size for adaptive mode. Defaults to 50.
            window_ref_mode (str): Reference window mode. "Fixed" or "Sliding".Defaults to "Fixed".
            overlap (bool): Whether to allow overlapping windows. Defaults to True.
            sliding_step (int): Sliding step size if overlap is True. Defaults to 50.
            continuous (bool): Whether to allow continuous windows. Defaults to True.
            gap_size (int): Gap size between events if not continuous. Defaults to 0.
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
        Applies the configured windowing strategy to the transition log, generating the windows index.
        Stores the result as a dictionary: {window_index: {'start': <start>, 'end': <end>}}
        """
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
        # Fix: use orient='index' for to_dict
        self.windows_df_dict: Dict[int, Dict[str, int]] = windows_df.set_index('window_index').to_dict(orient='index')

    def get_windowing_strategy(self) -> Dict[int, Dict[str, int]]:
        """
        Returns the current windows index dictionary generated by the windowing strategy.
        Returns:
            dict: {window_index: {'start': int, 'end': int}}
        """
        return self.windows_df_dict

    def set_process_representation(self, threshold_anomaly: int = 0,
                                    control_flow_features: Optional[set] = None,
                                    time_features: Optional[dict] = None,
                                    resource_features: Optional[dict] = None,
                                    data_features: Optional[dict] = None) -> None:
        """
        Configures the process representation settings.
        
        Args:
            threshold_anomaly (int): Threshold for anomaly detection. Defaults to 0.
            control_flow_features (set, optional): Set of control flow features to include. Defaults to {'frequency', 'probability'}.
            time_features (dict, optional): Dictionary of time-related features. Defaults to {}.
            resource_features (dict, optional): Dictionary of resource-related features. Defaults to {}.
            data_features (dict, optional): Dictionary of data-related features. Defaults to {}.
        """
        self.threshold_anomaly = threshold_anomaly
        self.control_flow_features = control_flow_features if control_flow_features is not None else {'frequency', 'probability'}
        self.time_features = time_features if time_features is not None else {}
        self.resource_features = resource_features if resource_features is not None else {}
        self.data_features = data_features if data_features is not None else {}

    def run_process_representation(self, transition_log_sample_original: pd.DataFrame, control_flow_features: Optional[set] = None, time_features: Optional[dict] = None, resource_features: Optional[dict] = None, data_features: Optional[dict] = None) -> None:
        """
        Executes the process representation analysis.
        
        Args:
            transition_log_sample_original (pd.DataFrame): Original transition log sample.
            control_flow_features (set, optional): Set of control flow features to include. Defaults to {'frequency', 'probability'}.
            time_features (dict, optional): Dictionary of time-related features. Defaults to {}.
            resource_features (dict, optional): Dictionary of resource-related features. Defaults to {}.
            data_features (dict, optional): Dictionary of data-related features. Defaults to {}.
        """
        # Set the features
        if control_flow_features is None:
            control_flow_features = self.control_flow_features
        if time_features is None:
            time_features = self.time_features
        if resource_features is None:
            resource_features = self.resource_features
        if data_features is None:
            data_features = self.data_features

        # Create a copy of the original transition log
        transition_log = transition_log_sample_original.copy()
        
        # Create a standard Transition Matrix (TM) process representation using frequency and percentual features for anomaly filter
        process_representation_df = pd.crosstab(transition_log["activity_from"], transition_log["activity_to"], normalize=False).stack().reset_index().rename(columns={0: "frequency"})
        process_representation_df["percentual"] = process_representation_df["frequency"]/process_representation_df["frequency"].sum()
        process_representation_df = process_representation_df.sort_values(by=['activity_from', 'activity_to'], ascending=[True, True]).set_index(['activity_from','activity_to'])

        # Set transitions with lower percentage or frequency than the threshold to zero (anomaly filter)
        if self.threshold_anomaly < 1:
            process_representation_df.loc[process_representation_df['percentual'] <= self.threshold_anomaly] = 0
        else:
            process_representation_df.loc[process_representation_df['frequency'] <= self.threshold_anomaly] = 0

        # Remove all transition with zero frequency
        process_representation_df = process_representation_df[process_representation_df["frequency"] > 0]

        # Initialize a dictionary to store process features
        process_features_dict = {
            "frequency": process_representation_df[["frequency"]],
            "percentual": process_representation_df[["percentual"]]
        }

        # Identify alpha relations
        alpha_relations = {'causality', 'parallel', 'choice', 'loop'}
        requested_alpha_features = control_flow_features & alpha_relations
        normal_features = control_flow_features - alpha_relations - {'frequency', 'percentual'}

        # Add control-flow features (excluding alpha relations)
        for control_flow_feature in normal_features:
            try:
                process_features_dict[control_flow_feature] = getattr(TMPD_process_features, "get_feature_" + control_flow_feature)(process_representation_df, control_flow_feature)
            except Exception as e:
                print("Error in control_flow_feature representation: ", control_flow_feature)
                print("Error: ", e)

        # Add alpha relations in one call if needed
        if requested_alpha_features:
            try:
                alpha_df = TMPD_process_features.get_feature_alpha_relations(process_representation_df)
                # Only keep requested columns (convert set to sorted list for indexing)
                alpha_cols = [col for col in ['causality', 'parallel', 'choice', 'loop'] if col in requested_alpha_features]
                alpha_df = alpha_df[alpha_cols]
                for col in alpha_df.columns:
                    process_features_dict[col] = alpha_df[[col]]
            except Exception as e:
                print("Error in alpha relations representation: ", requested_alpha_features)
                print("Error: ", e)

        # Add time features
        for method, time_feature in time_features:
            try:
                process_features_dict[f"{method}_{time_feature}"] = getattr(TMPD_process_features, "get_feature_" + method)(process_representation_df, transition_log, method, time_feature)
            except Exception as e:
                print("Error in time_feature representation: ", method, " and ", time_feature)
                print("Error: ", e)

        # Add resource features
        for method, resource_feature in resource_features:
            try:
                process_features_dict[f"{method}_{resource_feature}"] = getattr(TMPD_process_features, "get_feature_" + method)(process_representation_df, transition_log, method, resource_feature)
            except Exception as e:
                print("Error in resource_feature representation: ", method, " and ", resource_feature)
                print("Error: ", e)

        # Add data features
        for method, data_feature in data_features:
            try:
                process_features_dict[f"{method}_{data_feature}"] = getattr(TMPD_process_features, "get_feature_" + method)(process_representation_df, transition_log, method, data_feature)
            except Exception as e:
                print("Error in data_feature representation: ", method, " and ", data_feature)
                print("Error: ", e)

        # Merge all features transition matrices - Fix: ensure all values are DataFrames
        dataframes_to_merge = [df for df in process_features_dict.values() if isinstance(df, pd.DataFrame)]
        if dataframes_to_merge:
            process_representation_df = reduce(lambda left, right: pd.merge(left, right, on=['activity_from', 'activity_to'], how='outer'), dataframes_to_merge).fillna(0)
        else:
            process_representation_df = pd.DataFrame()

        # Store the process representation
        self.process_representation_df = process_representation_df

    def get_process_representation(self) -> pd.DataFrame:
        """
        Retrieves the process representation generated by the run_process_representation method.
        
        Returns:
            pd.DataFrame: Process representation as a Pandas DataFrame.
        """
        return self.process_representation_df

    def set_change_representation(self, change_features_strategy_dict: dict) -> None:
        """
        Configures the strategies for analyzing feature changes in the process.
        
        Args:
            change_features_strategy_dict (dict): Dictionary specifying the change representation strategies.
                Example:
                {
                    'delta_matrix_strategy': {
                        'frequency_delta': {'process_feature': 'frequency', 'method': 'aggregation', 'agg_function': 'sum'},
                        'probability_delta': {'process_feature': 'probability', 'method': 'aggregation', 'agg_function': 'sum'},
                        'frequency_delta_percentage': {'process_feature': 'frequency', 'method': 'percentage'},
                        'prob_freq_delta_weight': {'process_feature': 'probability', 'method': 'aggregation_weight', 'agg_function': 'sum', 'weight_feature': 'frequency'}
                    },
                    'statistic_test_strategy': {
                        'frequency_gtest_pvalue': {'process_feature': 'frequency', 'method': 'g_test', 'contingency_matrix_sum_value': '5', 'remove_zeros': 'True'},
                        'frequency_cramersv': {'process_feature': 'frequency', 'method': 'cramers_v', 'contingency_matrix_sum_value': '5', 'remove_zeros': 'True'}
                    }
                }
        """
        self.change_features_strategy_dict = change_features_strategy_dict

    def run_change_representation(self) -> None:
        """
        Executes the change representation analysis based on the defined strategies.
        Compares process representations between windows to detect changes.
        """
        # Initiating the change representation dictionary
        change_representation_dict = copy.deepcopy(self.windows_df_dict)

        # Iterate over each window index and window information
        for window_index, window_info in self.windows_df_dict.items():
            # Run process representation with the current detection window
            self.run_process_representation(self.transition_log.iloc[window_info['start']: window_info['end']])
            process_representation_detection_window_df = self.get_process_representation()

            # Check if it's the first window
            if window_index > 0:
                # Add information about which window was used as a reference
                change_representation_dict[window_index]['reference_window_index'] = reference_window_index
                
                # Loop through the change feature strategies and their corresponding parameters
                for change_feature_strategy, change_feature_params_dict in self.change_features_strategy_dict.items():
                    try:
                        # Call the corresponding change feature method and update the results in the dictionary
                        change_representation_dict[window_index].update(
                            getattr(TMPD_change_features, "get_" + change_feature_strategy)(
                                process_representation_reference_window_df,
                                process_representation_detection_window_df,
                                change_feature_params_dict,
                                list(self.control_flow_features),
                                [feature for feature, _ in self.time_features],
                                [feature for feature, _ in self.resource_features],
                                [feature for feature, _ in self.data_features],
                            )
                        )
                    except Exception as e:
                        print("Error in run_change_representation: ", change_feature_strategy)   
                        print("Error: ", e)
                
                # If the window reference mode is Sliding, the current detection window becomes the reference window
                if self.window_ref_mode == "Sliding":
                    process_representation_reference_window_df = process_representation_detection_window_df.copy()
                    reference_window_index = window_index

            # If it's the first window, the detection window just becomes the reference window 
            else:
                process_representation_reference_window_df = process_representation_detection_window_df.copy()
                reference_window_index = window_index

        # Create a DataFrame from the change representation dictionary
        self.change_representation_df = pd.DataFrame.from_dict(change_representation_dict, orient='index')

    def get_change_representation(self) -> pd.DataFrame:
        """
        Retrieves the change representation generated by the run_change_representation method.
        
        Returns:
            pd.DataFrame: Change representation DataFrame.
        """
        return self.change_representation_df

    def set_detection_task(self, detection_task_strategy_dict: dict) -> None:
        """
        Configures the detection task strategies.
        
        Args:
            detection_task_strategy_dict (dict): Dictionary defining the detection task strategies and their parameters.
                Example:
                {
                    'time_series_strategy': {
                        'cpd_frequency_delta': {'change_features': ['frequency_delta'], 'method': 'cpd_pelt', 'model': 'rbf', 'cost': 'rpt.costs.CostRbf()', 'min_size': '1', 'jump': '1', 'smooth': '3'},
                        'cpd_prob_freq_delta': {'change_features': ['prob_freq_delta_weight'], 'method': 'cpd_pelt', 'model': 'rbf', 'cost': 'rpt.costs.CostRbf()', 'min_size': '1', 'jump': '1', 'smooth': '3'}
                    },
                    'threshold_strategy': {
                        'gtest_frequency': {'change_features': ['frequency_gtest_pvalue'], 'method': 'comparison_operator', 'operator': 'le', 'threshold_value': '0.025', 'smooth': '3'},
                        'fixed_frequency_delta_percentage': {'change_features': ['frequency_delta_percentage'], 'method': 'comparison_operator', 'operator': 'ge', 'threshold_value': '0.05', 'smooth': '3'}
                    }
                }
        """
        self.detection_task_strategy_dict = detection_task_strategy_dict

    def run_detection_task(self) -> None:
        """
        Executes the detection task based on the configured strategies.
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
        
        Returns:
            pd.DataFrame: Detection task results as a Pandas DataFrame.
        """
        return self.detection_task_result_df
    
    def set_localization_task(self, reference_window_index: int, detection_window_index: int, 
                             pvalue_threshold: float = 0.05, effect_threshold: float = 0.2, 
                             presence_percentage_threshold: float = 0.01, pseudo_count: int = 5,
                             perspectives: Optional[List[str]] = None) -> None:
        """
        Configures the localization task parameters.
        
        Args:
            reference_window_index (int): Index of the reference window.
            detection_window_index (int): Index of the detection window.
            pvalue_threshold (float): P-value threshold for statistical tests. Defaults to 0.05.
            effect_threshold (float): Effect size threshold. Defaults to 0.2.
            presence_percentage_threshold (float): Minimum presence percentage threshold. Defaults to 0.02.
            pseudo_count (int): Pseudo-count for smoothing. Defaults to 5.
            perspectives (List[str], optional): List of perspectives to include in the analysis. 
                                             Can include: 'control_flow', 'time', 'resource', 'data'.
                                             Defaults to all perspectives if None.
        """
        self.reference_window_index_localization = reference_window_index
        self.detection_window_index_localization = detection_window_index
        self.pvalue_threshold_localization = pvalue_threshold
        self.effect_threshold_localization = effect_threshold
        self.pseudo_count_localization = pseudo_count
        self.presence_percentage_threshold_localization = presence_percentage_threshold
        # Set perspectives, default to all if None
        all_perspectives = ['control_flow', 'time', 'resource', 'data']
        self.perspectives_localization = perspectives if perspectives is not None else all_perspectives
        # Validate perspectives
        if perspectives is not None:
            invalid_perspectives = set(perspectives) - set(all_perspectives)
            if invalid_perspectives:
                raise ValueError(f"Invalid perspectives: {invalid_perspectives}. Must be one or more of {all_perspectives}")

    def run_localization_task(self) -> None:
        """
        Executes the localization task to identify specific changes in process transitions.
        """
        # Getting the reference window
        reference_window_data = self.transition_log.iloc[
            self.get_windowing_strategy()[self.reference_window_index_localization]['start']:
            self.get_windowing_strategy()[self.reference_window_index_localization]['end']
        ]
        self.run_process_representation(reference_window_data)
        reference_transition_matrix = self.get_process_representation()
        reference_transition_matrix = reference_transition_matrix.reset_index()

        # Getting the detection window
        detection_window_data = self.transition_log.iloc[
            self.get_windowing_strategy()[self.detection_window_index_localization]['start']:
            self.get_windowing_strategy()[self.detection_window_index_localization]['end']
        ]
        self.run_process_representation(detection_window_data)
        detection_transition_matrix = self.get_process_representation()
        detection_transition_matrix = detection_transition_matrix.reset_index()

        # Get DFGs
        self.reference_dfg = TMPD_understanding_tasks.create_dfg_from_dataset(reference_transition_matrix)
        self.detection_dfg = TMPD_understanding_tasks.create_dfg_from_dataset(detection_transition_matrix)

        # Compare DFGs
        dfg_changes = TMPD_understanding_tasks.compare_dfgs(self.reference_dfg, self.detection_dfg)

        # Convert DFG to process trees and get BPMN diagram text
        self.reference_bpmn_text = TMPD_understanding_tasks.create_process_tree_from_dfg(self.reference_dfg, parameters={"noise_threshold": 0})
        self.detection_bpmn_text = TMPD_understanding_tasks.create_process_tree_from_dfg(self.detection_dfg, parameters={"noise_threshold": 0})

        # Extracting features from both reference and detection windows
        features_windows = (
            set(reference_transition_matrix.columns) | set(detection_transition_matrix.columns)
        ) - {'activity_from', 'activity_to', 'percentual'}

        # Get significant transition changes list
        self.significant_transition_changes = TMPD_understanding_tasks.significant_transition_changes_detection(self, reference_transition_matrix, detection_transition_matrix, features_windows, dfg_changes)

        # Filtering only the perspectives selected for the localization task
        if isinstance(self.significant_transition_changes, pd.DataFrame):
            self.significant_transition_changes = self.significant_transition_changes[
                self.significant_transition_changes['perspective'].isin(self.perspectives_localization)
            ].reset_index(drop=True) 

        # # Filter out perspective feature changes for new or deleted transitions
        # new_transitions = set(dfg_changes.get('New transitions added to the process', []))
        # deleted_transitions = set(dfg_changes.get('Deleted transitions from the process', []))
        # new_deleted_transitions = list(new_transitions | deleted_transitions)

        # # Filter based on perspective column - exclude perspective features for new/deleted transitions
        # mask = ~(
        #     self.significant_transition_changes['transition'].isin(new_deleted_transitions)
        #     & ~self.significant_transition_changes['perspective'].isin(self.perspectives_localization)
        # )
        # self.significant_transition_changes = self.significant_transition_changes[mask].reset_index(drop=True)

        # Converting significant transition changes list to a dict
        significant_transition_changes_dict = {}
        for feature in features_windows:
            # Check if the feature has any significant transition changes
            if isinstance(self.significant_transition_changes, pd.DataFrame) and feature in self.significant_transition_changes['feature'].unique():
                # Extract transitions for the feature
                transitions = self.significant_transition_changes[self.significant_transition_changes['feature'] == feature]['transition'].tolist()
            
            # Assign ['None'] if there are no transitions for the feature
            # else:
            #     transitions = ['None']

                # Add the transitions to the dictionary with a modified key
                significant_transition_changes_dict["Transitions with variations in " + str(feature)] = transitions

        # Combine significant transition changes list with DFG changes
        self.high_level_changes = significant_transition_changes_dict | dfg_changes

    def get_localization_task(self, show_localization_dfg: bool = True, show_original_dfg: bool = False, 
                             show_original_bpmn: bool = False, show_annotations: bool = True) -> tuple:
        """
        Retrieves the results of the localization task.
        
        Args:
            show_localization_dfg (bool): Whether to show DFGs with localization results. Defaults to True.
            show_original_dfg (bool): Whether to show original DFGs. Defaults to False.
            show_original_bpmn (bool): Whether to show original BPMN diagrams. Defaults to False.
            show_annotations (bool): Whether to show annotations of the changes in the DFG. Defaults to True.
            
        Returns:
            tuple: (significant_transition_changes, high_level_changes, reference_bpmn_text, detection_bpmn_text, window_info)
                where window_info contains:
                - reference_window_index: Index of the reference window
                - detection_window_index: Index of the detection window
                - reference_window_range: {'start': int, 'end': int} - transition log indices
                - detection_window_range: {'start': int, 'end': int} - transition log indices
        """
        # Show DFGs
        if show_original_dfg:
            pm4py.view_dfg(self.reference_dfg.graph, self.reference_dfg.start_activities, self.reference_dfg.end_activities)
            pm4py.view_dfg(self.detection_dfg.graph, self.detection_dfg.start_activities, self.detection_dfg.end_activities)

        # Show BPMN
        if show_original_bpmn:
            # Get BPMNs
            self.reference_bpmn = TMPD_understanding_tasks.create_bpmn_from_dfg(self.reference_dfg)
            self.detection_bpmn = TMPD_understanding_tasks.create_bpmn_from_dfg(self.detection_dfg)
            pm4py.view_bpmn(self.reference_bpmn)
            pm4py.view_bpmn(self.detection_bpmn)

        # Show DFGs with Localization results
        if show_localization_dfg:
            TMPD_understanding_tasks.localization_dfg_visualization(self.reference_dfg, self.high_level_changes, self.significant_transition_changes, bgcolor="white", rankdir="LR", node_penwidth="2", edge_penwidth="2", show_annotations=show_annotations)
            TMPD_understanding_tasks.localization_dfg_visualization(self.detection_dfg, self.high_level_changes, self.significant_transition_changes, bgcolor="white", rankdir="LR", node_penwidth="2", edge_penwidth="2", show_annotations=show_annotations)

        
        return self.significant_transition_changes, self.high_level_changes, self.reference_bpmn_text, self.detection_bpmn_text
    
    def set_characterization_task(self, llm_company: str = "google", llm_model: str = "gemini-2.0-flash", 
                                 api_key_path: Optional[str] = None, llm_instructions_path: Optional[str] = None) -> None:
        """
        Configures the characterization task parameters for LLM-based analysis.
        
        Args:
            llm_company (str): LLM provider company. Defaults to "google".
            llm_model (str): LLM model name. Defaults to "gemini-2.0-flash".
            api_key_path (str, optional): Path to API key file. Defaults to None.
            llm_instructions_path (str, optional): Path to LLM instructions file. Defaults to None.
        """
        self.llm_company = llm_company
        self.llm_model = llm_model
        self.llm = TMPD_understanding_tasks.llm_instanciating(llm_company, llm_model, api_key_path)

        # Load LLM Instructions json and add contextualized informations
        self.llm_instructions = TMPD_understanding_tasks.llm_instructions_load(llm_instructions_path)

    def run_characterization_task(self) -> None:
        """
        Executes the comprehensive characterization task using LLM-based analysis.
        Provides holistic, contextualized understanding of what changed in the process.
        """
        # Get the process representation DataFrames from localization task
        reference_window_data = self.transition_log.iloc[
            self.get_windowing_strategy()[self.reference_window_index_localization]['start']:
            self.get_windowing_strategy()[self.reference_window_index_localization]['end']
        ]
        self.run_process_representation(reference_window_data, control_flow_features={'frequency', 'percentual', 'causality', 'parallel', 'choice', 'loop'}, time_features={}, resource_features={}, data_features={})
        reference_transition_matrix = self.get_process_representation()
        reference_transition_matrix = reference_transition_matrix.reset_index()

        detection_window_data = self.transition_log.iloc[
            self.get_windowing_strategy()[self.detection_window_index_localization]['start']:
            self.get_windowing_strategy()[self.detection_window_index_localization]['end']
        ]
        self.run_process_representation(detection_window_data, control_flow_features={'frequency', 'percentual', 'causality', 'parallel', 'choice', 'loop'}, time_features={}, resource_features={}, data_features={})
        detection_transition_matrix = self.get_process_representation()
        detection_transition_matrix = detection_transition_matrix.reset_index()
        
        # Prepare the comprehensive characterization prompt
        self.llm_characterization_prompt = TMPD_understanding_tasks.llm_characterization_prompt(
            self.llm_instructions, 
            reference_transition_matrix, 
            detection_transition_matrix, 
            self
        )
        
        # Call LLM response
        self.characterization_response = TMPD_understanding_tasks.llm_call_response(
            self.llm_company, 
            self.llm_model, 
            self.llm, 
            self.llm_characterization_prompt
        )


    def get_characterization_task(self) -> tuple:
        """
        Retrieves the results of the comprehensive characterization task.
        
        Returns:
            tuple: (characterization_prompt, characterization_response)
        """
        return self.llm_characterization_prompt, self.characterization_response


    def set_explanation_task(self) -> None:
        """
        Configures the explanation task parameters.
        TODO: Implement explanation task configuration.
        """
        pass

    def run_explanation_task(self) -> None:
        """
        Executes the explanation task.
        TODO: Implement explanation task execution.
        """
        pass

    def get_explanation_task(self) -> pd.DataFrame:
        """
        Retrieves the results of the explanation task.
        TODO: Implement explanation task results retrieval.
        
        Returns:
            pd.DataFrame: Explanation task results.
        """
        # TODO: Implement actual return value
        return pd.DataFrame()

    # def set_characterization_task_phd(self, llm_company: str = "google", llm_model: str = "gemini-2.0-flash", 
    #                                  api_key_path: Optional[str] = None, llm_instructions_path: Optional[str] = None) -> None:
    #     """
    #     Configures the PhD-level characterization task parameters for LLM-based analysis.
        
    #     Args:
    #         llm_company (str): LLM provider company. Defaults to "google".
    #         llm_model (str): LLM model name. Defaults to "gemini-2.0-flash".
    #         api_key_path (str, optional): Path to API key file. Defaults to None.
    #         llm_instructions_path (str, optional): Path to LLM instructions file. Defaults to None.
    #     """
    #     self.llm_company = llm_company
    #     self.llm_model = llm_model
    #     self.llm = TMPD_understanding_tasks.llm_instanciating(llm_company, llm_model, api_key_path)

    #     # Load LLM Instructions json and add contextualized informations
    #     self.llm_instructions = TMPD_understanding_tasks.llm_instructions_load(llm_instructions_path)

    # def run_characterization_task_phd(self) -> None:
    #     """
    #     Executes the PhD-level characterization task using LLM-based analysis.
    #     """
    #     ### BPMN analysis
    #     # Prepare the prompt
    #     self.llm_bpmn_analysis_prompt = TMPD_understanding_tasks.llm_bpmn_analysis_prompt(self.llm_instructions, self.reference_bpmn_text, self.detection_bpmn_text)
    #     print("################################ llm_bpmn_analysis_prompt #####################################")
    #     print(self.llm_bpmn_analysis_prompt)
    #     # Call LLM response
    #     self.llm_bpmn_analysis_response = TMPD_understanding_tasks.llm_call_response(self.llm_company, self.llm_model, self.llm, self.llm_bpmn_analysis_prompt)
    #     print("################################ llm_bpmn_analysis_response #####################################")
    #     print(self.llm_bpmn_analysis_response)

    #     ### Classification prompt
    #     # Prepare the prompt
    #     self.llm_classification_prompt = TMPD_understanding_tasks.llm_classification_prompt(self.llm_instructions, self.high_level_changes, self.reference_bpmn_text, self.detection_bpmn_text, self.llm_bpmn_analysis_response) 
    #     print("################################ llm_classification_prompt #####################################")
    #     print(self.llm_classification_prompt)
    #     # Call LLM response
    #     self.characterization_classification_response = TMPD_understanding_tasks.llm_call_response(self.llm_company, self.llm_model, self.llm, self.llm_classification_prompt) 
    #     print("################################ characterization_classification_response #####################################")
    #     print(self.characterization_classification_response)

    #     # Call LLM classification formatting
    #     self.characterization_classification_dict = TMPD_understanding_tasks.llm_classification_formatting(self.characterization_classification_response)
    #     print("################################ characterization_classification_dict #####################################")
    #     print(self.characterization_classification_dict)

    # def get_characterization_task_phd(self) -> tuple:
    #     """
    #     Retrieves the results of the PhD-level characterization task.
        
    #     Returns:
    #         tuple: (characterization_classification_dict, characterization_classification_response, llm_bpmn_analysis_response)
    #     """
    #     return self.characterization_classification_dict, self.characterization_classification_response, self.llm_bpmn_analysis_response