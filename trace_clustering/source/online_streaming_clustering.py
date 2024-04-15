from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.stats import skew
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from copy import deepcopy

from .MOACluStream import CluStream
# from clusopt_core.cluster import CluS tream

import gc
gc.enable()

def __get_model(model_name, model_parameters):
    if model_name == 'MOACluStream':
        return CluStream.MOACluStream(**model_parameters)
    
    # elif model_name == 'BIRCH':
    #     return Birch(**model_parameters)
    
    # elif model_name == 'DenStream':
    #     return streams.DenStream(**model_parameters)

    return None


def __partial_fit(model_name, model, values, i=0):
    if model_name == 'MOACluStream':
        return model.offline_cluster(values, i)

    elif model_name in ['BIRCH', 'DenStream', 'CluStream']:
        if len(values.shape) == 1:
            return model.partial_fit(values.reshape(1,-1))
        else:
            return model.partial_fit(values)

# # # # # # # # # #
###################
#  MICRO CLUSTERS 
###################
# # # # # # # # # #

def __get_micro_clusters_identifiers(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'DenStream':
        return model.get_micro_clusters_ids()
    
    return []

def __get_micro_clusters_centers(model_name, model, colnames=None, to_dataframe=False):
    if model_name == 'MOACluStream':
        if to_dataframe: 
            return pd.DataFrame(
                np.array([x.get_center() for x in model.kernels]),
                columns=colnames,
                index=__get_micro_clusters_identifiers(model_name, model)
            )
        else:
            return np.array([x.get_center() for x in model.kernels])
    
    elif model_name == 'CluStream':
        if to_dataframe:
            return pd.DataFrame(
                model.get_kernel_centers(),
                columns=colnames
            )
        else:
            return model.get_kernel_centers()            

    elif model_name == 'BIRCH':
        if to_dataframe:
            return pd.DataFrame(
                model.subcluster_centers_,
                columns=colnames
            )
        else:
            return model.subcluster_centers_

    elif model_name == 'DenStream':
        m_centers = np.array([
            x.center().tolist() for x in model.p_micro_clusters
        ])

        if to_dataframe:
            if len(m_centers) > 0:
                return pd.DataFrame(
                    m_centers,
                    columns=colnames,
                    index=__get_micro_clusters_identifiers(model_name, model)
                )
            else:
                return pd.DataFrame(columns=colnames)
        else:
            return m_centers


def __get_micro_clusters_weights(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'DenStream':
       return model.get_micro_clusters_weights()

    elif model_name == 'BIRCH':
        micro_w = np.array([x.n_samples_ for x in model.root_.subclusters_])
        return micro_w/np.sum(micro_w)

    return np.array()


def __get_micro_clusters_radius(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'DenStream':
        return model.get_micro_clusters_radius()

    elif model_name == 'BIRCH':
        return np.array([
            x.radius for x in model.root_.subclusters_
        ])
        
    return np.array()


# # # # # # # # # #
###################
# OUTLIERS CLUSTERS
###################
# # # # # # # # # #

def __get_outliers_clusters_centers(model_name, model, colnames=None):
    if model_name == 'DenStream':
        o_centers = np.array([
                x.center().tolist() for x in model.o_micro_clusters
        ])

        if len(o_centers) > 0:
            return pd.DataFrame(
                o_centers,
                columns=colnames
            )
        else:
            return pd.DataFrame(
                columns=colnames
            )
    
    return np.nan

def __get_outliers_clusters_ids(model_name, model) -> list:
    if model_name == 'DenStream':
        return model.get_outliers_clusters_ids()
    return []


def __get_outliers_clusters_weights(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'BIRCH':
        return np.nan

    if model_name == 'DenStream':
        return model.get_outliers_clusters_weights()
        
    return np.array([])


def __get_outliers_clusters_radius(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'BIRCH':
        return np.nan

    if model_name == 'DenStream':
        model.get_outliers_clusters_radius()
        
    return np.array([])



# # # # # # # # # #
###################
#  MACRO CLUSTERS 
###################
# # # # # # # # # #

def __get_macro_clusters_centers(model_name, model, colnames=None, to_dataframe = False):
    if model_name == 'MOACluStream':
        if to_dataframe:
            return pd.DataFrame(
                model.get_macro_clusters_centers(),
                columns=colnames
            )
        else:
            return model.get_macro_clusters_centers()

    elif model_name == 'BIRCH':
        centers_ = pd.DataFrame(
            model.subcluster_centers_
        ).groupby(model.subcluster_labels_).mean()
        
        if to_dataframe:
            return pd.DataFrame(
                centers_.values,
                columns=colnames
            )
        else:
            return centers_.values

    elif model_name == 'DenStream':
        ma_centers = model.get_macro_clusters_centers()
        
        if to_dataframe:
            if len(ma_centers) > 0:
                return pd.DataFrame(
                    ma_centers,
                    columns=colnames
                )
            else:
                return pd.DataFrame(
                    columns=colnames
                )
        return ma_centers

def __get_macro_clusters_weights(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'DenStream':
        return model.get_macro_clusters_weights()

    elif model_name == 'BIRCH':
        return np.array([])


def __get_macro_clusters_radius(model_name, model):
    if model_name == 'MOACluStream' or model_name == 'DenStream':
        return model.get_macro_clusters_radius()

    elif model_name == 'BIRCH':
        return np.array([])

 
#######
# MAIN
#######
def run_online_streaming_clustering(
    model_name, model_parameters, df, use_tqdm=False, expand_weights=False, get_macro_clusters=True
):
    """
        Performs streaming clustering algorithm over dataset

        Parameters:
        -----------
                   model_name (str): One of ['MOACluStream', 'DenStream', 'BIRCH']
            model_parameters (dict): Parameters to pass to the model __init__
                  df (pd.DataFrame): DataFrame to be clustered
                    use_tqdm (bool): Whether to use tqdm as progress bar
    """
    model = __get_model(model_name=model_name, model_parameters=model_parameters)
    df_ = df.copy()
    col_names = df_.columns.tolist()

    resp = []  

    if use_tqdm:
        it = tqdm_notebook(range(len(df_)))
    else:
        it = range(len(df_))

    for i in it:
        row = df_.iloc[i]

        r = __partial_fit(
            model_name=model_name, model=model, values=row.values, i=i
        )

        x = {
            "index": i,
            "col_names": col_names,

            "micro": __get_micro_clusters_centers(model_name=model_name, model=model, colnames=col_names),
            "micro_ids": __get_micro_clusters_identifiers(model_name=model_name, model=model),
            "micro_radius": __get_micro_clusters_radius(model_name=model_name, model=model),
            "micro_weights": __get_micro_clusters_weights(model_name=model_name, model=model),

            "outliers": __get_outliers_clusters_centers(model_name=model_name, model=model, colnames=col_names),
            "outliers_ids": __get_outliers_clusters_ids(model_name=model_name, model=model),
            "outliers_radius": __get_outliers_clusters_radius(model_name=model_name, model=model),
            "outliers_weights": __get_outliers_clusters_weights(model_name=model_name, model=model),
        }

        if get_macro_clusters:
            # Only runs Macro phase if microclusters have changed
            if r != "FITS_EXISTING_MICROCLUSTER":
                macro_ = __get_macro_clusters_centers(model_name=model_name, model=model, colnames=col_names)
            else:
                macro_ = resp[-1]['macro']

            x.update({
                "macro": macro_,
                "macro_weights": __get_macro_clusters_weights(model_name=model_name, model=model),
                "macro_radius": __get_macro_clusters_radius(model_name=model_name, model=model),
            })

            macro_

        if model_name == 'MOACluStream':
            x.update({
                "FITS_EXISTING_MICROCLUSTER": 1 if r == "FITS_EXISTING_MICROCLUSTER" else 0,
                "FORGET_OLD_MICROCLUSTER": 1 if r == "FORGET_OLD_MICROCLUSTER" else 0,
                "MERGED_MICROCLUSTER": 1 if r == "MERGED_MICROCLUSTER" else 0
            })

        if model_name == 'DenStream':
            x.update({
                "FITS_EXISTING_OUTLIERCLUSTER": 1 if r == "FITS_EXISTING_OUTLIERCLUSTER" else 0,
                "FITS_EXISTING_MICROCLUSTER": 1 if r == "FITS_EXISTING_MICROCLUSTER" else 0,
                "NEW_OUTLIERCLUSTER": 1 if r == "NEW_OUTLIERCLUSTER" else 0
            })

        calculate_time_independent_features(x)

        resp.append(x)


    resp = pd.DataFrame(resp).set_index('index')

    if expand_weights:
        # indx = resp['micro_weights'].apply(lambda x: list(x.keys())).explode().dropna().unique()
        # for n in indx:
        nro_max_micro = resp['micro_weights'].apply(lambda x: len(x.keys())).max()
        for n in range(nro_max_outliers):
            resp['micro_weights_' + str(n)] = resp['micro_weights'].apply(
                lambda x: x[n] if len(x) > n else 0 # if n in x else 0 # 
            )
            resp['micro_radius_' + str(n)] = resp['micro_radius'].apply(
                lambda x: x[n] if len(x) > n else 0 # if n in x else 0 # 
            )

        nro_max_macro = resp['macro_weights'].apply(lambda x: len(x)).max()
        # indx = resp['macro_weights'].apply(lambda x: list(x.keys())).explode().unique()
        for n in range(nro_max_macro):
            resp['macro_weights_' + str(n)] = resp['macro_weights'].apply(
                lambda x: x[n] if len(x) > n else 0
            )
            resp['macro_radius_' + str(n)] = resp['macro_radius'].apply( 
                lambda x: x[n] if len(x) > n else 0
            )


        nro_max_outliers = resp['outliers_weights'].apply(lambda x: len(x) if not isinstance(x, float) else 0).max()
        for n in range(nro_max_outliers):
        # indx = resp['outliers_weights'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else []).explode().dropna().unique()
        # for n in indx:
            resp['outliers_weights_' + str(n)] = resp['outliers_weights'].apply(
                lambda x: x[n] if len(x) > n else 0 # if n in x else 0 # 
            )
            resp['outliers_weights_' + str(n)] = resp['outliers_weights'].apply(
                lambda x: x[n] if len(x) > n else 0 # if n in x else 0 # if len(x) > n else 0
            )

    return resp


def pairwise_MSE(v, count_non_zero=False):
    MSEs = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            sq_error = (v[i] - v[j]) ** 2
            
            if count_non_zero:
                mse = np.count_nonzero(sq_error)/len(v[i])
            else:
                mse = np.mean(sq_error)
            
            MSEs.append(mse)
    return np.array(MSEs)


#######################################
# # # # # # # # # # # # # # # # # # # # 
# CALCULATE TIME INDEPENDENT FEATURES #
# # # # # # # # # # # # # # # # # # # #
#######################################
def calculate_time_independent_features(x):
    for type__ in ['micro', 'macro', 'outliers']:
        if type__ in x and not isinstance(x[type__], float):
            x.update({
                'AVG__' + type__ + '_interclusters_distance': distance.pdist(x[type__]).mean()                            if len(x[type__]) > 1 else np.nan,
                'STD__' + type__ + '_interclusters_distance': distance.pdist(x[type__]).std()                             if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_interclusters_jaccard_distance': distance.pdist(x[type__], metric='jaccard').mean()  if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_interclusters_cosine_distance': distance.pdist(x[type__], metric='cosine').mean()    if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_interclusters_MSE': pairwise_MSE(x[type__]).mean()                                   if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_interclusters_countnonzero_MSE': pairwise_MSE(x[type__], count_non_zero=True).mean() if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_centroids_dimensions_std': x[type__].std(axis=0).mean()                              if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_centroids_std': x[type__].std(axis=1).mean()                                         if len(x[type__]) > 1 else np.nan,
                'SSQ__' + type__ + '_centroids': (x[type__] ** 2).sum()                                                   if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_radius': np.mean(x[type__ + '_radius'])                                              if len(x[type__]) > 1 else np.nan,
                'AVG__' + type__ + '_weights': np.mean(x[type__ + '_weights'])                                            if len(x[type__]) > 1 else np.nan,
            })


##########################################
# # # # # # # # # # # # # # # # # # # # # # 
# CALCULATE FEATURES BETWEEN CLUSTERINGS  #
# # # # # # # # # # # # # # # # # # # # # #
##########################################
def distance_between_centroids(c1, c2):
    if isinstance(c1.micro, pd.DataFrame) or isinstance(c1.micro, np.floating):
        c1_micro = c1.micro.values
        c2_micro = c2.micro.values

        try:
            c1_macro = c1.macro.values
            c2_macro = c2.macro.values
        except: 
            c1_macro = np.array([])
            c2_macro = np.array([])
    else:
        c1_micro = c1.micro
        c2_micro = c2.micro

        try:
            c1_macro = c1.macro
            c2_macro = c2.macro
        except: 
            c1_macro = np.array([[]])
            c2_macro = np.array([[]])


    if isinstance(c1.outliers, pd.DataFrame):
        c1_outliers = c1.outliers.values
        c2_outliers = c2.outliers.values
    else:
        c1_outliers = c1.outliers
        c2_outliers = c2.outliers


    dist_micro = distance.cdist(c1_micro, c2_micro)
    dist_macro = distance.cdist(c1_macro, c2_macro)
    
    # Hungarian method
    _, col_ind_micro = linear_sum_assignment(dist_micro)
    _, col_ind_macro = linear_sum_assignment(dist_macro)
    
    try:
        mse_micro = np.mean((c1_micro - c2_micro[col_ind_micro]) ** 2, axis=0)
        error_micro = np.sum(c1_micro - c2_micro[col_ind_micro], axis=1)

        mse_micro_per_cluster = np.mean((c1_micro - c2_micro[col_ind_micro]) ** 2, axis=1)
        
        diff_radius_micro = c1.micro_radius - c2.micro_radius[col_ind_micro]
        diff_weight_micro = c1.micro_weights - c2.micro_weights[col_ind_micro]
        # diff_radius_micro = [
        #     np.nan if key not in c1.micro_radius else value - c1.micro_radius[key] 
        #     for key, value in c2.micro_radius.items()
        # ]

        # diff_weight_micro = [
        #     np.nan if key not in c1.micro_weights else value - c1.micro_weights[key] 
        #     for key, value in c2.micro_weights.items()
        # ]
    except Exception as e:
        mse_micro = 0
        error_micro = 0
        diff_radius_micro = 0
        diff_weight_micro = 0
        mse_micro_per_cluster = [0]
        
    try:
        mse_macro = np.mean((c1_macro - c2_macro[col_ind_macro]) ** 2, axis=0)
        error_macro = np.sum(c1_macro - c2_macro[col_ind_macro], axis=1)

        mse_macro_per_cluster = np.mean((c1_macro - c2_macro[col_ind_macro]) ** 2, axis=1)

        diff_radius_macro = c1.macro_radius - c2.macro_radius[col_ind_macro]
        diff_weight_macro = c1.macro_weights - c2.macro_weights[col_ind_macro]
    except Exception as e:
        mse_macro = 0
        error_macro = 0  
        diff_weight_macro = 0  
        diff_radius_macro = 0   
        mse_macro_per_cluster = [0] 
    
    try:
        new_c = np.sum([1 if c not in c2.micro_ids else 0 for c in c1.micro_ids])
    except Exception as e:
        new_c = 0

    try:
        new_outliers = np.sum([
            1 if c not in c2.outliers_ids else 0 for c in c1.outliers_ids
        ])
        
        # dist_outliers = distance.cdist(c1.outliers, c2.outliers)
        #  _, col_ind_micro = linear_sum_assignment(dist_outliers)

        # mse_micro = np.mean((c1.micro - c2.micro[col_ind_micro]) ** 2, axis=0)
        # error_micro = np.sum(c1.micro - c2.micro[col_ind_micro], axis=1)

    except Exception as e:
        new_outliers = 0
    
    resp = {        
        'avg_distance_between_micro_centroids': np.mean(dist_micro.min(axis=0)) if len(dist_micro) > 0 else 0,
        'avg_distance_between_macro_centroids': np.mean(dist_macro.min(axis=0)) if len(dist_macro) > 0 else 0,
        
        'std_distance_between_micro_centroids': np.std(dist_micro.min(axis=0)) if len(dist_micro) > 0 else 0,
        'std_distance_between_macro_centroids': np.std(dist_macro.min(axis=0)) if len(dist_macro) > 0 else 0,
        
        'total_MSE_per_micro_centroids': np.sum(mse_micro_per_cluster),
        'total_MSE_per_macro_centroids': np.sum(mse_macro_per_cluster),

        'std_MSE_per_micro_centroids': np.std(mse_micro_per_cluster),
        'std_MSE_per_macro_centroids': np.std(mse_macro_per_cluster),

        'total_MSE_between_micro_centroids': np.sum(mse_micro),
        'total_MSE_between_macro_centroids': np.sum(mse_macro),

        'std_MSE_between_micro_centroids': np.std(mse_micro),
        'std_MSE_between_macro_centroids': np.std(mse_macro),

        'ERROR_between_micro_centroids': np.sum(error_micro),
        'ERROR_between_macro_centroids': np.sum(error_macro),
        
        'count_non_zero_MSE_micro': np.count_nonzero(mse_micro),
        'count_non_zero_MSE_macro': np.count_nonzero(mse_macro),

        'count_non_zero_MSE_per_cluster_micro': np.count_nonzero(mse_micro_per_cluster)/len(mse_micro_per_cluster),
        'count_non_zero_MSE_per_cluster_macro': np.count_nonzero(mse_macro_per_cluster)/len(mse_micro_per_cluster),        

        'std_diff_radius_micro': np.std(diff_radius_micro),
        'std_diff_radius_macro': np.std(diff_radius_macro),

        'sum_diff_radius_micro': np.sum(diff_radius_micro),
        'sum_diff_radius_macro': np.sum(diff_radius_macro),

        'sum_diff_weight_micro': np.sum(diff_weight_micro),
        'sum_diff_weight_macro': np.sum(diff_weight_macro),

        'std_diff_weight_micro': np.std(diff_weight_micro),
        'std_diff_weight_macro': np.std(diff_weight_macro),

        'diff_centroids': c1_macro - c2_macro[col_ind_macro],
        'diff_centroids_micro': c1_micro - c2_micro[col_ind_micro],

        'new_centroids': new_c,
        'new_outliers': new_outliers,
        
        'micro_len': 0 if isinstance(c2_micro, np.floating) else len(c2_micro),
        'macro_len': 0 if isinstance(c2_macro, np.floating) else len(c2_macro),
        'outliers_len': 0 if isinstance(c2_outliers, np.floating) else len(c2_outliers),
    }
    
    # try:
    #     for k in range(len(col_ind_macro)):
    #         resp['ERROR_cluster_' + str(k)] = np.mean(c1_macro[k] - c2_macro[col_ind_macro][k])
    # except Exception as e:
    #     raise e
        
    return resp


def extract_features_dataframe(df, use_tqdm=True, step=1):
    features = []
    
    if use_tqdm:
        loop = tqdm_notebook(range(step, len(df)))
    else:
        loop = range(step, len(df))
    
    for i in loop:
        features.append(
            distance_between_centroids(
                df.iloc[i - step], 
                df.iloc[i]
            )
        )
    ft = pd.DataFrame(features)
    ft.index = ft.index + step
    return ft