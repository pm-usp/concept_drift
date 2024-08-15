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
from langchain_openai import ChatOpenAI
import google.generativeai as genai



# Helper function to identify the type of variable
def identify_statistical_test(series):
    """
    Identify the type of variable based on its values.
    - 'proportion' if values are between 0 and 1.
    - 'continuous' otherwise.
    """
    if np.all((series >= 0) & (series <= 1)):
        return 'proportion_test'
    return 'count_test'


# Calculate Cohen's h effect size for proportions.
def cohen_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


# Calculate Cramers V statistic for categorical-categorical association
def cramers_corrected_stat(data, ref_column, det_column, ref_value, det_value):
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

    # # Add pseudo-count
    # success_reference = int(round(prob_reference * (total_count_reference + pseudo_count)))
    # success_detection = int(round(prob_detection * (total_count_detection + pseudo_count)))

    # # Calculate standard deviation
    # std = np.sqrt((prob_reference * (1 - prob_reference) / (total_count_reference + pseudo_count)) +
    #               (prob_detection * (1 - prob_detection) / (total_count_detection + pseudo_count)))

    # # Avoid divide by zero
    # if std == 0:
    #     return 1

    # # Perform the test
    # stat, p_value = proportions_ztest([success_reference, success_detection], 
    #                                   [total_count_reference + pseudo_count, total_count_detection + pseudo_count])
    
    return p_value


def changed_transitions_detection(self, merged_windows_df, features_windows):

    # Initialize a dictionary to store the results
    changed_transitions = pd.DataFrame(columns=['transition', 'feature', 'p_value', 'effect_size', 'ref_value', 'det_value', 'dif_value'])

    # Perform statistical tests for each variable and transition
    i=0
    for index, row in merged_windows_df.iterrows():
        transition = (row['activity_from'], row['activity_to'])

        for feature in features_windows:

            variable_ref = feature + '_ref'
            variable_det = feature + '_det'

            ref_value = row[variable_ref]
            det_value = row[variable_det]

            # Check if the feature is a proportion or count
            test_type = identify_statistical_test(merged_windows_df[[variable_ref, variable_det]].values.flatten())

            # Perform the appropriate statistical test
            if test_type == 'count_test':
                p_value = perform_count_test(merged_windows_df, row, variable_ref, variable_det)
                # Calculate corrected Cramér's V
                effect_size = cramers_corrected_stat(merged_windows_df, variable_ref, variable_det, ref_value, det_value)
                is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_count_threshold_localization
            else:
                p_value = perform_proportions_test(merged_windows_df, row, transition, variable_ref, variable_det, self.pseudo_count_localization)
                # Calculate Cohen's h
                effect_size = cohen_h(ref_value, det_value)
                is_significant = p_value is not None and p_value < self.pvalue_threshold_localization and abs(effect_size) > self.effect_prop_threshold_localization
            # Record the significant result
            if is_significant:
                changed_transitions.loc[i] = [transition, feature, p_value, effect_size, ref_value, det_value, det_value - ref_value]
                i += 1
        
    return changed_transitions  


def create_dfg_from_dataset(dataset):
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

    return create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq)


def create_dfg_from_transitions(dfg_transitions, start_activities_freq, end_activities_freq):
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
    # Retrieve transition sets from the graphs
    dfg1_transitions = set(dfg1.graph.keys())
    dfg2_transitions = set(dfg2.graph.keys())

    # Calculate new, deleted, and altered transitions
    new_transitions = dfg2_transitions - dfg1_transitions
    deleted_transitions = dfg1_transitions - dfg2_transitions
    
    # Get activities
    dfg1_activities = set(t[0] for t in dfg1.graph.keys()) | set(t[1] for t in dfg1.graph.keys())
    dfg2_activities = set(t[0] for t in dfg2.graph.keys()) | set(t[1] for t in dfg2.graph.keys())

    # Calculate new and deleted activities
    new_activities = dfg2_activities - dfg1_activities
    deleted_activities = dfg1_activities - dfg2_activities
    
    # Get start and end activities
    dfg1_start_activities = set(dfg1.start_activities.keys())
    dfg2_start_activities = set(dfg2.start_activities.keys())
    dfg1_end_activities = set(dfg1.end_activities.keys())
    dfg2_end_activities = set(dfg2.end_activities.keys())

    # Calculate new and deleted start and end activities
    new_start_activities = dfg2_start_activities - dfg1_start_activities
    deleted_start_activities = dfg1_start_activities - dfg2_start_activities
    new_end_activities = dfg2_end_activities - dfg1_end_activities
    deleted_end_activities = dfg1_end_activities - dfg2_end_activities

    dfg_changes = {
        'New transitions added to the process': list(new_transitions) if new_transitions else ["None"]
        ,'Deleted transitions from the process': list(deleted_transitions) if deleted_transitions else ["None"]
        ,'New activities added to the process': list(new_activities) if new_activities else ["None"]
        ,'Deleted activities from the process': list(deleted_activities) if deleted_activities else ["None"]
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
    return pm4py.discover_bpmn_inductive(dfg, noise_threshold=0)


def wrap_text(text, max_length=10):
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

def localization_dfg_visualization(dfg, change_informations, bgcolor="white", rankdir="LR", node_penwidth="2", edge_penwidth="2"):
    
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
            # Strip the unwanted prefix and get the part after it
            prefix_length = len("Transitions with variations in")
            suffix = key[prefix_length:].strip()  # Optionally remove any leading or trailing whitespace
            for transition in transitions:
                if transition in edge_annotations:
                    edge_annotations[transition].append(suffix)
                else:
                    edge_annotations[transition] = [suffix]


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
        if (source, target) in edge_annotations:
            dot.edge(source, target
                    , label="Freq: " + str(count) + '\nDif. in '
                        + '\nDif. in '.join(edge_annotations[(source, target)])
                    , color=edge_color, penwidth=edge_penwidth) 
        else: 
            dot.edge(source, target, label="Freq: " + str(count), color=edge_color, penwidth=edge_penwidth)

    # Connect the start node to the real start activities and the real end activities to the end node
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

            if ('START', act) in edge_annotations: 
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

            if (act, 'END') in edge_annotations: 
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
    if llm_company == "openai":

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

    # Open the JSON file for reading
    with open(llm_instructions_path, 'r') as file:
        # Parse the yaml file into a Python dictionary
        llm_instructions = yaml.safe_load(file)

    # transform controlflow_change_patterns in dict
    consolidated_dict = {}
    for instruction_dict in llm_instructions["controlflow_change_patterns"]:
        for key, value in instruction_dict.items():
            consolidated_dict[key] = value
    llm_instructions["controlflow_change_patterns"] = consolidated_dict
                            
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


def llm_bpmn_analysis_prompt(llm_instructions, reference_bpmn_text, detection_bpmn_text):

    # Add BPMN diagrams to prompt
    llm_instructions["instructions_bpmn_analysis"] += (
    ''' 
    \n### BPMN diagrams ###
    - The BPMN before the concept drift: {0}. \n
    - The BPMN after the concept drift: {1}. \n

    ''').format(reference_bpmn_text, detection_bpmn_text)

    return llm_instructions["instructions_bpmn_analysis"]


def llm_classification_prompt(llm_instructions, change_informations, reference_bpmn_text, detection_bpmn_text, llm_bpmn_analysis_response):

    # prompt += (
    # ''' 
    # \n### BPMN diagrams ###
    # - The BPMN before the concept drift: {0}. \n
    # - The BPMN after the concept drift: {1}. \n

    # ''').format(reference_bpmn_text, detection_bpmn_text)

    # Get the prompt
    prompt = llm_instructions["instructions_classification"] 

    # Add BPMN Diagrams Comparison Analysis to prompt
    prompt += (
        ''' 
        \n### BPMN Diagrams Comparison Analysis ###
        {0}. \n

        ''').format(llm_bpmn_analysis_response)
    

    ### Add Transition and Activities Changes List and Control-flow Change Patterns to prompt depending on conditions
    
    # If there is at least a new or deleted activity, then suggest SRE, PRE, CRE, or RP
    if change_informations['New activities added to the process'] != ['None'] or change_informations['Deleted activities from the process'] != ['None']:
        
        prompt += (
        ''' 
        \n### Transition and Activities Changes List ###
        \n'New transitions added to the process': {0}.
        \n'Deleted transitions from the process': {1}.
        \n'New activities added to the process': {2}.
        \n'Deleted activities from the process': {3}.

        ''').format(change_informations['New transitions added to the process']
                    , change_informations['Deleted transitions from the process']
                    , change_informations['New activities added to the process']
                    , change_informations['Deleted activities from the process'])
        
        prompt += (
        ''' 
        \n### Control-flow Change Patterns ###\n
        ''')
        prompt += (
            llm_instructions['controlflow_change_patterns']['sre_instructions'] 
            + llm_instructions['controlflow_change_patterns']['pre_instructions'] 
            + llm_instructions['controlflow_change_patterns']['cre_instructions'] 
            + llm_instructions['controlflow_change_patterns']['rp_instructions'] 
        )  

    # If the changes don't involve addition or deletion of activities but rather addition or deletion of transitions between existing activities, then suggest SM, CM, PM, or SW, CF, PL, LP,CD,  CB, or CP
    elif change_informations['New transitions added to the process'] != ['None'] or change_informations['Deleted transitions from the process'] != ['None']:
        
        prompt += (
        ''' 
        \n### Transition and Activities Changes List ###
        \n'New transitions added to the process': {0}.
        \n'Deleted transitions from the process': {1}.
        \n'New activities added to the process': {2}.
        \n'Deleted activities from the process': {3}.

        ''').format(change_informations['New transitions added to the process']
                    , change_informations['Deleted transitions from the process']
                    , change_informations['New activities added to the process']
                    , change_informations['Deleted activities from the process'])

        prompt += (
        ''' 
        \n### Control-flow Change Patterns ###\n
        ''')

        # Movement Patterns
        prompt += (llm_instructions['controlflow_change_patterns']['sm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['cm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['pm_instructions'] 
                + llm_instructions['controlflow_change_patterns']['sw_instructions'] 
        )

        # Gateway Type Changes
        prompt += (llm_instructions['controlflow_change_patterns']['pl_instructions'] 
                + llm_instructions['controlflow_change_patterns']['cf_instructions'] 
        )

        # Synchronization (Parallel involved)
        prompt += (llm_instructions['controlflow_change_patterns']['cd_instructions'] 
        )

        # Bypass (XOR involved)
        prompt += (llm_instructions['controlflow_change_patterns']['cb_instructions'] 
        )

        # Loop Fragment Changes
        prompt += (llm_instructions['controlflow_change_patterns']['lp_instructions'] 
                   #llm_instructions['controlflow_change_patterns']['cp_instructions'] 
        )


    # If the changes don't involve addition or deletion of activities nor addition or deletion of transitions between existing activities, but rather only changes in the transitions, then is FR
    else:

        prompt += (
        ''' 
        \n### Transition and Activities Changes List ###\n
        {0}

        ''').format(change_informations)

        prompt += (
        ''' 
        \n### Control-flow Change Patterns ###\n
        ''')

        prompt += (
            llm_instructions['controlflow_change_patterns']['fr_instructions'] 
        )


    return prompt



def llm_classification_formatting(characterization_classification):

    # Finding the start and end of the dictionary string
    try:
        start_str = "result_dict = {"
        end_str = "}"
        start_index = characterization_classification.find(start_str) + len(start_str) - 1
        end_index = characterization_classification.find(end_str, start_index) + 1

        return ast.literal_eval(characterization_classification[start_index:end_index].strip())
    except:
        return "Classification not in the expected format."
    
    