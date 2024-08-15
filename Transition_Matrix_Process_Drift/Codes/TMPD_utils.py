# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:00:46 2022

@author: Antonio Carlos Meira Neto
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import scipy.stats as ss
import gzip as gzip_lib
from datetime import datetime, timedelta

def process_instance(el):
    """
        Process each 'process instance' element from the .mxml file
        and returns as dict
    """
    resp = []
    for entry in el[1:]:
        r = {
            "CaseId": el.get("id")
        }
        for item in entry:
            if item.tag == 'Data':
                r[item.tag] = item[-1].text
            else:
                r[item.tag] = item.text
        resp.append(r)
    return resp

def read_mxml(file):
    """
        Read MXML file into a Pandas DataFrame
    """
    root = ET.parse(file).getroot()
    process = root[-1]
    
    resp = []
    for p_instance in process:
        for r in process_instance(p_instance):
            resp.append(r)
    
    return pd.DataFrame.from_dict(resp)


def cumulative_counting(traces):
    """
        Cumulative counting in column
    """
    t_ant = None
    cnt = 0
    
    resp = []
    for t in traces:
        if t != t_ant:
            cnt += 1
            t_ant = t
        resp.append(cnt)
        
    return(pd.Series(resp) - 1)
    

def parse_mxml(file, gzip=False, aliases=None, replace_whitespaces="_"):
    """
        Runs all basic prep and return preped DataFrame
    """
    if gzip:
        df = read_mxml(gzip_lib.open(file,'r'))
    else:
        df = read_mxml(file)

    df = df[df['CaseId'].notnull()]
    
    df["WorkflowModelElement"] = df.WorkflowModelElement.apply(lambda x: x.replace(' ', replace_whitespaces))
    
    if aliases is not None:
        df["Activity"] = df.WorkflowModelElement.replace(aliases)
    else:
        df["Activity"] = df.WorkflowModelElement

    return df


def add_start_end_activities(event_log, case_id_col, activity_col, timestamp_col):
    """
    Add START and END activities for each case ID in an event log.
    Use sorting by multiple criteria to position START and END correctly.

    :param event_log: DataFrame containing the event log data
    :param case_id_col: String name of the column containing the case IDs
    :param activity_col: String name of the column containing the activities
    :param timestamp_col: String name of the column containing the timestamps
    :return: DataFrame with modified event log
    """
    # Create a new variable that will be the case_id numerical for sort
    event_log['case_id_numerical'] = pd.factorize(event_log[case_id_col])[0]

    # Sort by CaseId and Timestamp to ensure the order
    event_log.sort_values(by=['case_id_numerical', timestamp_col], inplace=True)

    # Add a SortOrder column with default value 1
    event_log['SortOrder'] = 1

    # Create empty lists to store new rows
    start_rows = []
    end_rows = []

    # Iterate over each CaseId and add START and END rows
    for case_id in event_log[case_id_col].unique():
        case_df = event_log[event_log[case_id_col] == case_id]
        first_activity = case_df.iloc[0]
        last_activity = case_df.iloc[-1]

        # Create a START activity row
        start_row = first_activity.copy()
        start_row[activity_col] = 'START'
        start_row['SortOrder'] = 0
        start_rows.append(start_row)

        # Create an END activity row
        end_row = last_activity.copy()
        end_row[activity_col] = 'END'
        end_row['SortOrder'] = 2
        end_rows.append(end_row)

    # Use pd.concat to append the new rows to the DataFrame
    event_log = pd.concat([event_log, pd.DataFrame(start_rows + end_rows)], ignore_index=True)

    # Sort by CaseId, Timestamp, and SortOrder
    event_log.sort_values(by=['case_id_numerical', timestamp_col, 'SortOrder'], inplace=True)

    # Remove the SortOrder column as it's no longer needed
    event_log.drop(['case_id_numerical', 'SortOrder'], axis=1, inplace=True)

    return event_log

def list_match_metrics(gt_list, pred_list):

    # Convert lists to sets and lowercase
    gt_set = set(item.lower() for item in gt_list)
    pred_set = set(item.lower() for item in pred_list)
    
    # Proceed with set operations
    true_positives = len(gt_set.intersection(pred_set))
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1