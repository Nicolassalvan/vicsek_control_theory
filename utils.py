# -*- coding: utf-8 -*-
"""

Created on Fri Jun 21 15:11:06 2024
@author: Nicolas Salvan (S6155197) - Ian Soede (S4385764), at University of Groningen, Biomimitecs group
"""
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from scipy.spatial import distance
from scipy.signal import correlate
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from numba import jit
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================================= #
# =============================== Model functions ============================= #
# ============================================================================= #

def smallest_angle_between_vectors(angle1, angle2):
    
    """
    Find the smallest angle between two vectors specified with two angles, that is to say the acute angle.

    Parameters
    ----------
    angle1 : float
        Angle in radians
    angle2 : float
        Angle in radians
        
    Returns
    -------
    angle_diff : float
        Smallest angle between two vectors in radians
    """
        
    angle_diff = angle1-angle2
    return (angle_diff + np.pi) % (2 * np.pi) - np.pi

def smoothing(brut_signal,L):
    res = np.copy(brut_signal) # duplication of values
    for i in range (1,len(brut_signal)-1): # every value except the first and the last

        L_g = min(i,L) # number of values available to the left
        L_d = min(len(brut_signal)-i-1,L) # number of values available to the right
        Li=min(L_g,L_d) # number of values available in total

        res[i]=np.sum(brut_signal[i-Li:i+Li+1])/(2*Li+1)
    return res

def getNeighbours(positions, radius=1):

    distanceMatrix = distance.cdist(positions, positions, 'euclidean')
    neighbours = distanceMatrix < radius
    return neighbours

def cross_correlation(series1, series2, max_lag):
    # handle nans 
    series1 = np.nan_to_num(series1)
    series2 = np.nan_to_num(series2)
    # print("Series1", series1.shape)
    # print("Series2", series2.shape)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = correlate(series1, series2, mode='full', method='auto')
    mid = len(corr) // 2
    corr = corr[mid - max_lag: mid + max_lag + 1]
    return lags, corr

def compute_lag_estimation_on_mean(df, df_labels, cluster, max_lag=300):
    mean_orientation = mean_orientation_cluster(df, df_labels, cluster)
    flock_orientation = get_flock_orientation(df)
    lag_estimation = np.zeros(df_labels.shape[1])
    for bird in range(df_labels.shape[1]):
        lags, corr = cross_correlation(mean_orientation, flock_orientation[:, bird], max_lag=max_lag)
        lag_estimation[bird] = lags[np.argmax(corr)]
    return lag_estimation


def granger_causality_matrix(data, max_lag=5, test='ssr_chi2test'):
    """
    Computes the Granger Causality test for each pair of variables in the data. 

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    max_lag : int, optional
        The maximum lag to consider. The default is 5.
    test : str, optional
        The test to use. The default is 'ssr_chi2test'.

    Returns
    -------
    df_matrix : pd.DataFrame
        Dataframe of the square matrix of p-values of the Granger Causality test for each pair of variables.
    """
    variables = data.columns
    df_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    
    for c in df_matrix.columns:
        for r in df_matrix.index:
            # print(f'Calculating Granger Causality for pair ({r}, {c})')
            if c != r:
                test_result = grangercausalitytests(data[[r, c]], max_lag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1], 4) for i in range(max_lag)]
                min_p_value = np.min(p_values)
                df_matrix.loc[r, c] = min_p_value
        # print(f'{c}', end = "...", flush=True)
    return df_matrix

def granger_causality_matrix_flock(df, max_lag=5):
    # Appliquer la fonction
    flock_orientation = get_flock_orientation(df)
    result_matrix = granger_causality_matrix(pd.DataFrame(np.nan_to_num(flock_orientation)), max_lag=max_lag)
    return result_matrix

def granger_causality_matrix_significant(df, max_lag=5, significance_level=0.05):
    result_matrix = granger_causality_matrix_flock(df, max_lag)
    causal_relations = (result_matrix < significance_level).astype(int)
    return causal_relations

def granger_causality_mean(data, max_lag=5, test='ssr_chi2test'):
    variables = data.columns
    df = data.copy()
    df['mean'] = np.mean(data, axis=1)
    ret = np.zeros(len(variables))
    for c in data.columns:
        # print(f'Calculating Granger Causality for pair (mean, {c})')
        test_result = grangercausalitytests(df[['mean', c]], max_lag, verbose=False)
        p_values = [round(test_result[i+1][0][test][1], 4) for i in range(max_lag)]
        min_p_value = np.min(p_values)
        ret[c] = min_p_value
    return ret

def compute_lag_matrix(df, max_lag=5):
    flock_orientation = get_flock_orientation(df)
    N = flock_orientation.shape[1]
    lag_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                lags, corr = cross_correlation(flock_orientation[:, i], flock_orientation[:, j], max_lag)
                lag_matrix[i, j] = lags[np.argmax(corr)]
    return lag_matrix


# ============================================================================= #
# ============================= Coefficients, ... ============================= #
# ============================================================================= #

def average_orientation(df):
    """
    Computes the average orientation of the birds in the simulation at each time step and returns it as a numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        The simulation data.

    Returns
    -------
    np.ndarray
        The average orientation of the birds at each time step. 
    """
    df_orient = extract_orientations_from_dataframe(df)
    theta_x = df_orient.filter(like="theta_x").to_numpy()
    theta_y = df_orient.filter(like="theta_y").to_numpy()
    angle = np.arctan2(theta_y, theta_x)
    angle_mean = np.mean(angle, axis=1)
    return angle_mean


def order_factor(df):
    df_orient = extract_orientations_from_dataframe(df)
    truc = df_orient.filter(like="theta_x").mean(axis=1)
    res = np.sqrt(df_orient.filter(like="theta_x").mean(axis=1)**2 + df_orient.filter(like="theta_y").mean(axis=1)**2)
    return res


def stationnary_order_factor(df):
    """
    Compute the stationnary order factor of the flock using Simpson integration.
    """
    tmax = df['t'].iloc[-1]
    order = order_factor(df)
    x = np.linspace(0, tmax, len(order))
    int_order = integrate.simpson(order, x=x) 
    return int_order/tmax

def binder_cumulant(df):
    tmax = df['t'].iloc[-1]
    order = order_factor(df)
    x = np.linspace(0, tmax, len(order))
    phi2 = integrate.simpson(order*order, x=x) 
    phi4 = integrate.simpson(order**4, x=x)
    return 1 - phi4 / (3 * phi2**2)

# ============================================================================= #
# =============================  Data clustering  ============================= #
# ============================================================================= #

### Distance computing 
@jit(nopython=True)
def distance_periodic_scaled(X, Y, periods):
    dims = periods.shape[0]
    n = X.shape[0]
    dist = 0 
    for d in range(dims):
        dist += (np.abs(X[d] - Y[d]) / periods[d])**2
    return np.sqrt(dist)


@jit(nopython=True)
def distance_periodic(X : np.array, Y : np.array, periods : np.array) -> float:
    """
    Compute the distance between two points in a periodic space.
    
    Parameters
    ----------
    X : np.array
        First point.
    Y : np.array
        Second point.
    periods : np.array
        Periods of the space.
        
    Returns
    -------
    float
        Euclidian distance between the two points, with periodic boundary conditions.
    """
    dims = periods.shape[0]
    n = X.shape[0]
    dist = 0 
    for d in range(dims):
        delta = np.abs(X[d] - Y[d])
        if delta > periods[d] / 2:
            delta = periods[d] - delta
        dist += (delta)**2
    return np.sqrt(dist)

### Clustering functions - Create labels for the clustering
def naive_clustering_labels_positions(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_pos = get_positions(df, i)
    df_pos_scaled = scaler.fit_transform(df_pos)
    db_pos = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_pos_scaled)
    return db_pos.labels_

def naive_clustering_labels_orientations(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_orient = get_angles(df, i)
    df_orient_scaled = scaler.fit_transform(df_orient.to_numpy().reshape(-1, 1))
    db_orient = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_orient_scaled)
    return db_orient.labels_

def naive_clustering_labels_positions_and_orientations(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_all = get_positions_and_orientations(df, i)
    df_all_scaled = scaler.fit_transform(df_all)
    db_all = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_all_scaled)
    return db_all.labels_


def periodic_clustering_labels_positions(df, i, k_coef, L, min_samples=5):
    list_pos = get_positions(df, i).to_numpy()
    N = len(list_pos)

    rho = N / L**2
    threshold = np.sqrt(min_samples / (np.pi * rho * k_coef))
    #compute the distance matrix
    # t_mat_start = time.time()
    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            dist_mat[i, j] = distance_periodic(list_pos[i], list_pos[j], np.array([L, L]))
            dist_mat[j, i] = dist_mat[i, j]
    # t_mat_end = time.time()
    db_pos = cluster.DBSCAN(eps=threshold, min_samples=min_samples, metric = 'precomputed').fit(dist_mat)
    # t_db_end = time.time()
    # print(f"Time for distance matrix computation: {t_mat_end - t_mat_start}")
    # print(f"Time for DBSCAN: {t_db_end - t_mat_end}")
    return db_pos.labels_

def periodic_clustering_labels_pos_ang(df, i, k_coef, L, delta_theta, min_samples=5):
    list_pos = get_positions(df, i).to_numpy()
    list_ang = get_angles(df, i).to_numpy()
    list_pos_ang = np.concatenate((list_pos, list_ang), axis=1)
    N = len(list_pos)

    rho = N / L**2
    threshold = np.sqrt(delta_theta**2 + min_samples / (np.pi * rho * k_coef))

    #compute the distance matrix
    # t_mat_start = time.time()
    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            dist_mat[i, j] = distance_periodic(list_pos_ang[i], list_pos_ang[j], np.array([L, L, 2 * np.pi]))
            dist_mat[j, i] = dist_mat[i, j]
    # t_mat_end = time.time()
    db_pos = cluster.DBSCAN(eps=threshold, min_samples=min_samples, metric = 'precomputed').fit(dist_mat)
    # t_db_end = time.time()
    # print(f"Time for distance matrix computation: {t_mat_end - t_mat_start}")
    # print(f"Time for DBSCAN: {t_db_end - t_mat_end}")
    return db_pos.labels_

### Clustering functions - Utils  
def clustering_labels_stats(labels):
    """
    Computes the number of clusters and the number of noise points in the clustering.

    Parameters
    ----------
    labels : np.ndarray
        Labels of the clustering.

    Returns
    -------
    n_clusters_ : int
        Number of clusters.
    n_noise : int   
        Number of noise points.
    """
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return n_clusters_, n_noise


def coloring_clusters(labels, cmap_name='rainbow'):
    n = len(set(labels))
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(labels+1 / max(labels+1))
    colors[labels == -1] = [0, 0, 0, 1]
    return pd.DataFrame(colors)

def cluster_count(df_labels):
    data = df_labels.to_numpy()
    cluster_count_list = []
    for i in range(len(data)):
        cluster_count_list.append(len(set(data[i])) - 1)

    cluster_count_arr = np.array(cluster_count_list)
    return cluster_count_arr

def cluster_biggest_size(df_labels):
    data = df_labels.to_numpy()
    cluster_sizes = []
    for i in range(len(data)):
        cnt = Counter(data[i])
        if len(cnt) > 1:
            most_common = cnt.most_common(2)
            if most_common[0][0] == -1: # if noise is the most common cluster
                cluster_sizes.append(most_common[1][1])
            else : # biggest cluster is not noise
                cluster_sizes.append(most_common[0][1])
        else : 
            most_common = cnt.most_common()
            if most_common[0][0] == -1:
                cluster_sizes.append(0)
            else:
                cluster_sizes.append(most_common[0][1]) 
    cluster_sizes = np.array(cluster_sizes)
    return cluster_sizes
# DOESNT WORK YET
def neighbours_heuristic(X, periods, n_neighbours, quantile = 0.95, plot = False, metric_func = distance_periodic):
    neighbours = NearestNeighbors(n_neighbors=10, metric=metric_func, metric_params={'periods': periods})
    neighbours.fit(X)
    dist, index = neighbours.kneighbors(X)
    distances = np.sort(dist[:, -1])
    eps = distances[int(quantile * len(distances))]
    if plot:
        plt.plot(distances)
        plt.title('K-distance Graph')
        plt.xlabel('Points')
        plt.ylabel('Distance')
        plt.title(f'K-distance Graph - eps = {eps : .3f}')
        plt.show()
    return eps


def living_clusters(df_labels):
    """
    Returns a matrix of shape (tmax, max_label+1) where is_alive[t, label] is True if the label is present at time t.as_integer_ratio

    Parameters
    ----------
    df_labels : pd.DataFrame
        Dataframe containing the labels of the clusters at each time step.
    
    Returns
    -------
    is_alive : np.array
        Matrix of shape (tmax, max_label+1) where is_alive[t, label] is True if the label is present
    
    """
    max_label = np.max(df_labels.to_numpy())
    tmax = df_labels.shape[0]
    is_alive = np.zeros((tmax, max_label+1))
    for t in range(df_labels.shape[0]):
        for label in range(max_label+1):
            is_alive[t, label] = label in df_labels.iloc[t].values
    return is_alive

def life_span(df_labels):
    """
    Returns the maximum life span of each cluster and the life span of each cluster.

    Parameters
    ----------
    df_labels : pd.DataFrame
        Dataframe containing the labels of the clusters at each time step.

    Returns
    -------
    life_span_max : np.array
        Array of size (max_label+1) containing the maximum life span of each cluster.
    life_span_all : list
        List of size (max_label+1) containing the life span of each cluster. life_span_all[label] is a list of integers 
        representing the life span of the cluster label, that is to say the number of consecutive time steps where the cluster 
        label is present.
    """
    max_label = np.max(df_labels.to_numpy())
    tmax = df_labels.shape[0]
    is_alive = living_clusters(df_labels)
    # duree de vie des clusters
    current_life_span = np.zeros(max_label+1)
    life_span_max = np.zeros(max_label+1)
    life_span_all = [[] for _ in range(max_label+1)]

    for t in range(df_labels.shape[0]):
        for label in range(max_label+1):
            # if the label is present at time t
            if is_alive[t, label]:
                current_life_span[label] += 1
            if is_alive[t, label] == False and current_life_span[label] > 0: # if the label is not present at time t and was present before 
                # update max 
                life_span_max[label] = max(life_span_max[label], current_life_span[label])
                # update all
                life_span_all[label].append(current_life_span[label])
                # reset current
                current_life_span[label] = 0
    # Adding the last values
    for label in range(max_label+1):
        if current_life_span[label] > 0:
            life_span_max[label] = max(life_span_max[label], current_life_span[label])
            life_span_all[label].append(current_life_span[label])
    return life_span_max, life_span_all

def categorize_flock(life_span, optimal_k):
    X = np.array(life_span).reshape(-1, 1)
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(X)

    # Get the labels
    labels = kmeans.labels_

    # Print the results 
    clusters = {}
    for label, duration in zip(labels, life_span):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(duration)

    # print("Results")
    clustering_stats = []
    for label, cluster in clusters.items():
        # print(f"Cluster {label + 1}: {cluster}, size = {len(cluster)}, mean = {np.mean(cluster):.2f}, STD = {np.std(cluster):.2f}")
        clustering_stats.append([len(cluster), np.mean(cluster), np.std(cluster)])

    clustering_stats = np.array(clustering_stats)
    return clustering_stats

def cluster_inertia(life_span, K_max=10):
    # List of life spans
    durations = life_span

    # Convert the list to a numpy array
    X = np.array(durations).reshape(-1, 1)

    warnings.filterwarnings("ignore") # Memory warning for KMeans - Can be ignored
    # Determine the optimal number of clusters using the elbow method
    inertias = []
    K_max = min(K_max, len(np.array(durations)))
    K = range(1, K_max+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        inertias.append(kmeans.inertia_)
    warnings.filterwarnings("default")
    return inertias

def categorize_flock(life_span, optimal_k):
    X = np.array(life_span).reshape(-1, 1)
    kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(X)

    # Get the labels
    labels = kmeans.labels_

    # Print the results 
    clusters = {}
    for label, duration in zip(labels, life_span):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(duration)

    # print("Results")
    clustering_stats = []
    for label, cluster in clusters.items():
        # print(f"Cluster {label + 1}: {cluster}, size = {len(cluster)}, mean = {np.mean(cluster):.2f}, STD = {np.std(cluster):.2f}")
        clustering_stats.append([len(cluster), np.mean(cluster), np.std(cluster)])

    clustering_stats = np.array(clustering_stats)
    return clustering_stats

def longer_lasting_cluster(df_labels):
    max, span = life_span(df_labels)
    # print(max)
    max = np.array(max)
    span = [item for sublist in span for item in sublist]
    span = np.array(span)
    return np.argmax(max)

def mean_orientation_cluster(df, df_labels, cluster):
    mean_orientation = np.zeros(df_labels.shape[0])
    for t in range(df_labels.shape[0]):
        mask = (df_labels.iloc[t] == cluster).to_numpy().astype(bool)
        orientation = get_angles(df, t).to_numpy()
        if np.sum(mask) > 0:
            mean_orientation[t] = np.mean(orientation[mask])
        else :
            mean_orientation[t] = np.nan
        # wait = input("PRESS ENTER TO CONTINUE.")
    return mean_orientation

def mean_position_cluster(df, df_labels, cluster):
    mean_position = np.zeros((df_labels.shape[0], 2))
    for t in range(df_labels.shape[0]):
        mask = (df_labels.iloc[t] == cluster)
        positions = get_positions(df, t).to_numpy()
        if np.sum(mask) > 0:
            mean_position[t] = np.mean(positions[mask], axis=0)
        else:
            mean_position[t] = np.nan
    return mean_position.T

def bird_correlation_with_mean(df):
    """
    Compute the correlation of each bird with the mean orientation of the flock.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing the simulation data.

    Returns:
    --------
    correlation_with_mean : np.array
        Array containing the correlation of each bird with the mean orientation of the flock.
    """
    flock_orientation = get_flock_orientation(df) # shape (tmax, N) where N is the number of birds
    df_bird_orientation = pd.DataFrame(flock_orientation) 
    df_bird_orientation['mean'] = df_bird_orientation.mean(axis=1)
    correlation_with_mean = df_bird_orientation.corr()['mean']
    correlation_with_mean = correlation_with_mean[:-1]
    correlation_with_mean = correlation_with_mean.to_numpy()
    # correlation_with_mean[i] is the correlation of bird i with the mean orientation of the flock
    return correlation_with_mean

def cluster_effectif(df_labels):
    # check if labels is dataframe 
    # matLabels = np.array(matLabels)
    # effectif au cours du temps 
    matLabels = df_labels.to_numpy()    
    max_cluster = np.max(matLabels)
    eff = np.zeros((df_labels.shape[0], max_cluster+1))
    noise = np.zeros(df_labels.shape[0])
    for t in range(df_labels.shape[0]):
        noise[t] = np.sum(matLabels[t] == -1)
        for i in range(max_cluster+1):
            eff[t][i] = np.sum(matLabels[t] == i)
    return eff, noise

# ============================================================================= #
# ============================= Data manipulation ============================= #
# ============================================================================= #

### Data manipulation - Utils for the simulation data
def simulationDataToCSV(simulationData, save_path):
    """
    Save simulation data to a CSV file.

    Parameters
    ----------
    simulationData : tuple
        Simulation data.
    save_path : str
        Path to save the CSV file.

    Returns
    -------
    None.

    """
    df = simulationDataToDataframe(simulationData)
    df.to_csv(save_path, index=False)
    print("Data saved in ", save_path)
    return None

def simulationDataToDataframe(simulationData):
    
    t, positionsHistory, orientationsHistory = simulationData[0], simulationData[1], simulationData[2]  
    headers = _headerSimulationData(positionsHistory.shape[1])
    data = np.concatenate((t[:,np.newaxis], positionsHistory.reshape(t.size,-1), orientationsHistory.reshape(t.size,-1)), axis=1)

    df = pd.DataFrame(data, columns=headers.split(','))
    return df

def _headerSimulationData(flockSize):

    """
    Create a header for the CSV file.

    Parameters
    ----------
    flockSize : int
        Number of particles in the flock.

    Returns
    -------
    header : str
        Header for the CSV file.

    """
    header = 't,'
    for i in range(flockSize):
        header += 'x'+str(i)+',y'+str(i)+','
    for i in range(flockSize):
        header += 'theta_x_'+str(i)+',theta_y_'+str(i)+','
    header = header[:-1]
    return header

### Data manipulation - Utils for the clustering data
def clusters_over_time(df, func=periodic_clustering_labels_positions, **kwargs):
    """
    Computes the labels of the clusters over time, using the given function. 

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe containing the simulation data.
    func: function
        The function to use to compute the labels. Must take the dataframe and the iteration as arguments.
    kwargs: dict
        The arguments to pass to the function.

    Returns:
    --------
    df_labels: pd.DataFrame
        The dataframe containing the labels of the clusters over time.
    """
    N = (df.shape[1] - 1) // 4 # x,y,v_x,v_y 
    t = df.shape[0]
    matLabels = np.zeros((t, N), dtype=int)

    for i in range(t):
        # df3 = pd.DataFrame(df.loc[i]).transpose()
        labels = func(df, i, **kwargs)
        matLabels[i] = labels

    # store in dataframe
    df_labels = pd.DataFrame(matLabels) 
    return df_labels

### Data manipulation - Extract columns from the dataframe
def extract_positions_from_dataframe(df):
    """
    Extract positions from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the positions.

    Returns
    -------
    positions : numpy.ndarray
        Positions.

    """
    positions = df.iloc[:,1:1+2*df.shape[1]//4]
    return positions

def extract_orientations_from_dataframe(df):
    """
    Extract orientations from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the orientations.

    Returns
    -------
    orientations : numpy.ndarray
        Orientations.

    """
    orientations = df.iloc[:,1+2*df.shape[1]//4:]
    return orientations 

### Data manipulation - Get positions and orientations from the dataframe
def get_positions(df, i):
    pos = extract_positions_from_dataframe(df)
    list_pos = pos.iloc[i].to_numpy().reshape(-1, 2)
    df_pos = pd.DataFrame(list_pos, columns=['x', 'y'])
    return df_pos

def get_orientations(df, i):
    orient = extract_orientations_from_dataframe(df)
    list_orient = orient.iloc[i].to_numpy().reshape(-1, 2)
    df_orient = pd.DataFrame(list_orient, columns=['theta_x', 'theta_y'])
    return df_orient

def get_positions_and_orientations(df, i):
    pos = get_positions(df, i)
    orient = get_orientations(df, i)
    df_all = pd.DataFrame(np.concatenate((pos, orient), axis=1), columns=['x', 'y', 'theta_x', 'theta_y'])
    return df_all

def get_positions_and_angles(df:pd.DataFrame, i:int)->pd.DataFrame:
    pos = get_positions(df, i)
    orient = get_orientations(df, i)
    angle = np.arctan2(orient['theta_y'], orient['theta_x']) 
    # convert to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    df_all = pd.DataFrame(np.concatenate((pos.to_numpy(), np.array(angle)[:,np.newaxis]), axis=1), columns=['x', 'y', 'angle'])
    return df_all

def get_angles(df:pd.DataFrame, i:int)->pd.DataFrame:
    """
    Get the angles of the birds at a given time step i. The angle is computed as the arctan2 of the y and x components of the orientation
    and is shifted by pi to have the angle in [0, 2*pi].

    Parameters
    ----------
    df : pd.DataFrame
        The simulation data.
    i : int
        The time step.
    
    Returns
    -------
    pd.DataFrame
        The angles of the birds at the time step i.
    """
    orient = get_orientations(df, i)
    # print(f"Orient shape : {orient.shape}")
    angle = np.arctan2(orient['theta_y'], orient['theta_x']) 
    # convert to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    # print(f"Angle shape : {angle.shape}")
    return pd.DataFrame(angle, columns=['angle'])

def get_centroids(df, labels, noiseBool = False):
    """
    Computes the centroids of the clusters in the simulation data at a given time step. 

    Parameters
    ----------
    df : pd.DataFrame
        The simulation data at a given time step.
    labels : np.ndarray
        Labels of the clusters.
    noiseBool : bool, optional
        If True, the noise points are included in the computation of the centroids.

    Returns
    -------
    pd.DataFrame
        The centroids of the clusters.
    """
    df = df.copy()
    df['labels'] = labels
    if not noiseBool:
        df = df[df['labels'] != -1]
    return df.groupby('labels').mean()

def get_bird_number(df):
    return (df.shape[1] - 1) // 4

def getError(df, radius): 
    N = get_bird_number(df)
    errorMatrix = np.zeros((df.shape[0], N))
    for i in range(df.shape[0]):
        positions = get_positions(df, i).to_numpy() # positions at time i
        angles = get_angles(df, i).to_numpy() # angles at time i
        distanceMatrix = distance.cdist(positions, positions, 'euclidean') # distance matrix at time i
        neigh_mask = distanceMatrix < radius
        meanAngles = np.dot(neigh_mask, angles) / np.sum(neigh_mask, axis=1, keepdims=True, dtype=float) # mean angles of neighbours at time i

        error = meanAngles - angles # error at time i = mean angles of neighbours at time i - angles at time i
        errorMatrix[i] = error.reshape(-1)

    dfError = pd.DataFrame(errorMatrix)
    return dfError


def get_flock_orientation(df):
    N, tmax = get_bird_number(df), df.shape[0]
    flock_orientation = np.zeros((tmax, N))
    
    for t in range(tmax):
        # print(t)
        orientation = get_angles(df, t).to_numpy()
        # print(orientation.shape, flock_orientation[t].shape)
        flock_orientation[t] = orientation.reshape(N)
        
    return flock_orientation



# ============================================================================= #
# ========================= Optimal asignment utils =========================== #
# ============================================================================= #

### === Constants === ###
# True if we want to print debug information
DEBUG = False
# Constants for the NoneFlock and DeadFlock
NoneFlock = -2
DeadFlock = -3
# Periodic boundary conditions

def contribution_matrix(labels_before, labels_after):
        ### === Cost matrix computation === ###
    # Counter matrix : number of elements of i in j 
    df_crosstab = pd.crosstab(pd.Series(labels_before), pd.Series(labels_after))
    if DEBUG:
        print("# Count matrix: \n", df_crosstab)

    # Delete noise column and line 
    count_matrix = df_crosstab

    if -1 in df_crosstab.columns:
        count_matrix = count_matrix.drop(-1, axis=1)
    if -1 in df_crosstab.index:
        count_matrix = count_matrix.drop(-1, axis=0)
    labels_after_assigned = count_matrix.columns.to_numpy()
    labels_before_assigned = count_matrix.index.to_numpy()

    if DEBUG: 
        print("# Count matrix after deletion: \n", count_matrix)
        print("Column names: ", labels_after_assigned)
        print("Index names: ", labels_before_assigned)

    cluster_counts_after = Counter(labels_after)
    # List of the number of clusters in the next frame to compute the contribution of each cluster to the next frame
    cluster_sizes_after = [cluster_counts_after[i] for i in labels_after_assigned] # i is a key in the dictionary of the counter /!\
    if DEBUG :
        print("Column count: ", cluster_sizes_after)

    # Normalise the count matrix by computing the contribution of each cluster to the next frame
    # The contribution is the number of elements of i inside the cluster j divided by the number of elements in the cluster j
    # contribution[i,j] = # elements of i in j / # elements in j 
    # The lines represents the clusters in the previous frame, the columns the clusters in the next frame
    count_matrix = count_matrix.div(cluster_sizes_after, axis=1)
    if DEBUG:
        print("# Normalised count matrix: \n", count_matrix)

    # Convert the matrix to a numpy array for the Hungarian algorithm
    cost_matrix = count_matrix.to_numpy() * -1 # We want to find the maximum of the contribution
    # so we multiply by -1 to find the minimum of -1 * contribution (max f = min -f)
    return cost_matrix, labels_before_assigned, labels_after_assigned

def centroid_distance_matrix(labels_before, labels_after, df_pos_before, df_pos_after, L): 
    # Creating centroids 
    gb_centroids_before = get_centroids(df_pos_before, labels_before)
    gb_centroids_after = get_centroids(df_pos_after, labels_after)

    dist = np.zeros((len(gb_centroids_before), len(gb_centroids_after)))
    labels_before = gb_centroids_before.index
    labels_after = gb_centroids_after.index
    for i in range(len(gb_centroids_before)):
        for j in range(len(gb_centroids_after)):
            x_i, y_i = gb_centroids_before.iloc[i].x, gb_centroids_before.iloc[i].y
            x_j, y_j = gb_centroids_after.iloc[j].x, gb_centroids_after.iloc[j].y
            dist[i,j] = distance_periodic(np.array([x_i, y_i]), np.array([x_j, y_j]), np.array([L, L]))

    return dist, np.asarray(labels_before), np.asarray(labels_after)



def to_square_matrix(matrix, noneValue):
    """
    Add imaginary clusters to make the cost matrix square. The
    imaginary clusters are labeled as noneValue. The noise cluster is labeled -1.

    Parameters
    ----------
    matrix : numpy array
        Cost matrix
    noneValue : int
        Value of the imaginary clusters

    Returns
    -------
    matrix : numpy array
        Square matrix with imaginary clusters
    """
    if matrix.shape[0] < matrix.shape[1]:
        diff = matrix.shape[1] - matrix.shape[0]
        matrix = np.concatenate((matrix, np.full((diff, matrix.shape[1]), noneValue)) , axis=0)
    elif matrix.shape[0] > matrix.shape[1]:
        diff = matrix.shape[0] - matrix.shape[1]
        matrix = np.concatenate((matrix, np.full((matrix.shape[0], diff), noneValue)), axis=1)
    return matrix

def unique_labels(labels, noise = -1, remove_noise_bool = True):
    unique_labels = list(set(labels))
    if noise in unique_labels and remove_noise_bool:
        unique_labels.remove(noise)
    return unique_labels

def optimal_assignment(labels_before, labels_after, cost_matrix_func=contribution_matrix, **kwargsCostMatrix):
    """
    Optimal assignment of the clusters from the previous frame to the next frame. If the data is unbalanced, 
    we add imaginary clusters to make the cost matrix square. Then, we use the Hungarian algorithm to find the optimal
    assignment of the clusters from the previous frame to the next frame. 
    
    The imaginary clusters are labeled -2. The noise cluster is labeled -1. 
    
    Parameters
    ----------
    labels_before : numpy array
        Labels of the clusters in the previous frame
    labels_after : numpy array
        Labels of the clusters in the next frame
    cost_matrix_func : function
        Function to compute the cost matrix. The function must return the cost matrix, 
        the labels of the clusters in the previous frame after the optimal assignment, 
        and the labels of the clusters in the next frame after the optimal assignment.
    **kwargsCostMatrix : dict
        Additional arguments for the cost matrix function
        
    Returns  
    -------
    labels_before_assigned : numpy array
        Labels of the clusters in the previous frame after the optimal assignment
    labels_after_assigned : numpy array
        Labels of the clusters in the next frame after the optimal assignment   
    contribution : numpy array
        Contribution of each cluster to the next frame after the optimal assignment
    """
    if DEBUG:
        print("Optimal assignment...")

    ### === Preprocessing === ###
    label_list_before = unique_labels(labels_before)
    label_list_after = unique_labels(labels_after)

    # Count the clusters 
    n_cluster_before = len(label_list_before) 
    n_cluster_after = len(label_list_after)

    ### === Corner cases === ###
    # Corner cases : no cluster in the previous or next frame
    if n_cluster_after == 0:
        # No cluster in the next frame
        if DEBUG:
            print(f"No cluster in the next frame :{Counter(labels_before)} ")
        labels_before_assigned = label_list_before
        labels_after_assigned = np.repeat(NoneFlock, n_cluster_before)
        contribution = np.zeros(len(labels_before))
        return labels_before_assigned, labels_after_assigned, contribution
    
    if n_cluster_before == 0:
        # No cluster in the previous frame
        if DEBUG:
            print(f"No cluster in the previous frame :{Counter(labels_after)} ")
        labels_before_assigned = np.repeat(NoneFlock, n_cluster_after)
        labels_after_assigned = label_list_after
        contribution = np.zeros(len(labels_after))
        return labels_before_assigned, labels_after_assigned, contribution


    if DEBUG:
        print("Labels before: ", label_list_before)
        print("Labels after: ", label_list_after)

    ### === Cost matrix computation === ###
    cost_matrix, labels_before_assigned, labels_after_assigned = cost_matrix_func(labels_before, labels_after, **kwargsCostMatrix)
    if DEBUG:
        print("Cost matrix: ", cost_matrix) 
    cost_matrix = to_square_matrix(cost_matrix, 0)

    # Delete the clusters we added by making the cost matrix square
    diff = np.abs(n_cluster_before - n_cluster_after)
    if DEBUG:
        print("Diff: ", diff)
    if n_cluster_before < n_cluster_after:
        labels_before_assigned = np.concatenate((labels_before_assigned, np.full(diff, NoneFlock))).astype(int)
    elif n_cluster_before > n_cluster_after:
        labels_after_assigned = np.concatenate((labels_after_assigned, np.ones(diff) * NoneFlock)).astype(int)

    ### === Hungarian algorithm for optimal assignment === ###
    # Hungarian algorithm - Optimal assignment of the clusters from the previous frame to the next frame
    assigned_row_indices, assigned_col_indices = linear_sum_assignment(cost_matrix)
    if DEBUG :
        print("Before index: ", assigned_row_indices)
        print("After  index: ", assigned_col_indices)
        print("Before Labels :", labels_before_assigned)
        print("After  Labels :", labels_after_assigned)
        print("Contribution : ", cost_matrix[assigned_row_indices, assigned_col_indices])

    contribution = cost_matrix[assigned_row_indices, assigned_col_indices]


    ### === Postprocessing the labels assigned === ###
    # Reorder the labels after the optimal assignment
    labels_before_assigned = labels_before_assigned[assigned_row_indices]
    labels_after_assigned = labels_after_assigned[assigned_col_indices]
    if DEBUG:
        print("Optimal assignment done.")
        print("Labels before assigned: ", labels_before_assigned)
        print("Labels after assigned: ", labels_after_assigned)
        print("Contribution: ", contribution)
    # Return the labels before and after the optimal assignment, and the contribution of each cluster to the next frame
    return labels_before_assigned, labels_after_assigned, contribution


class Flocks: 
    id_counter = 0

    def __init__(self):
        self.id_list = []
        self.isAlive = []    
    def add_flock(self):
        self.id_list.append(Flocks.id_counter)
        self.isAlive.append(True)
        new_id = Flocks.id_counter
        # print(f"~~~ New flock created: {new_id} ~~~")
        Flocks.id_counter += 1
        return new_id

    def get_last_flock_id(self):
        return Flocks.id_counter - 1
    
    def kill_flock(self, id):
        # print(f"~~~ Flock {id} killed ~~~")
        self.isAlive[id] = False

    def reset_id_counter(self):
        # print("~~~ Resetting id counter... ~~~")
        Flocks.id_counter = 0
    
    def __str__(self) -> str:
        txt = "Flocks: \n"
        txt = f'Id: {self.id_list} \n'
        txt += f'Is alive: {self.isAlive} \n'
        for id in self.id_list:
            if self.isAlive[id]:
                txt += f"Flock {id} ; [ALIVE] "
            else: 
                txt += f"Flock {id} ; [DEAD] "
            txt += "\n"
        
        return txt

def correct_labels_optimal_assignment(df_labels_to_copy, df, option: str, L):
    df_labels = df_labels_to_copy.copy()
    assert "contribution" in option or "centroid" in option, "Option must be either 'contribution' or 'centroid'"
    ### === Initialisation of the algorithm === ###

    # Create a matrix for the new labels. Initiation with noise (-1)
    new_labels_matrix = np.zeros((df_labels.shape[0], df_labels.shape[1]), dtype=int) - 1 

    # Initialise the new matrix
    new_labels_matrix[0,:] = df_labels.iloc[0,:].to_numpy()
    if DEBUG:
        print("New labels matrix counter: ", Counter(new_labels_matrix[0,:]))

    # Create the labels
    labels_before = df_labels.iloc[0,:].to_numpy()  
    labels_after = df_labels.iloc[1,:].to_numpy()

    # Create the positions
    df_pos_before = get_positions(df, 0)
    df_pos_after = get_positions(df, 1)

    # Create the flock objects
    if DEBUG:
        print("Creating the flock objects...")
    flocks = Flocks()
    flocks.reset_id_counter()
    for i in range(len(set(labels_before)) - 1):  
        new = flocks.add_flock()



    for i in range(df_labels.shape[0] - 1):
        ### === Iteration of optimal assignment === ###
        if DEBUG : 
            print(f'####################### Iteration {i} #######################')
            print(f"Labels before counter: {Counter(df_labels.iloc[i,:])}")
            print(f"Labels after counter: {Counter(df_labels.iloc[i+1,:])}")
            print(f"Optimal assignment iteration {i}...")
        # Optimal assignment : computes the best permutation of the labels after the optimal assignment. 
        labels_before = new_labels_matrix[i,:]
        labels_after = df_labels.iloc[i+1,:].to_numpy()
        df_pos_before = get_positions(df, i)
        df_pos_after = get_positions(df, i+1)

        if option == "contribution":
            before, after, contribution = optimal_assignment(labels_before, labels_after, cost_matrix_func=contribution_matrix)
        elif option == "centroid":
            before, after, contribution = optimal_assignment(labels_before, labels_after, cost_matrix_func=centroid_distance_matrix,
                                                              df_pos_before=df_pos_before, df_pos_after=df_pos_after, L=L)
        if DEBUG:
            print("Result of the optimal assignment: ")
            print(f"Before: {before}")
            print(f"After: {after}")
            print(f"Contribution: {contribution}")

        # Build the permutation of the labels after the optimal assignment
        replace_after = after.copy()
        if DEBUG:
            print("#### Renaming clusters..." )
        for id, (id_before, id_after, contrib) in enumerate(zip(before, after, contribution)):
        
            if DEBUG:
                print(f"### {id} : Before: {id_before} ; After: {id_after} ; Contribution: {contrib}")
            # CASES : 
            if id_before == NoneFlock and id_after != NoneFlock: 
                new = flocks.add_flock()
                replace_after[id] = new
                if DEBUG:
                    print("New cluster! : ", new)
            ## Cluster died
            elif id_after == NoneFlock and id_before != NoneFlock:
                if DEBUG:
                    print("Cluster died!")
                flocks.kill_flock(id_before)
                replace_after[id] = DeadFlock # We don't know where the cluster went : it may have fuse with another cluster, or just died
                # We keep track of the cluster that died
            ## Cluster continues : contribution above contributionThreshlod
            else :
                if DEBUG:
                    print("Cluster continues!")
                replace_after[id] = id_before
        if DEBUG:
            print("-> Before permutation: ",after)
            print("-> After  permutation: ",replace_after)
        label_dict = {label: new_label for label, new_label in zip(after, replace_after)}
        # Remove the NoneFlock and DeadFlock from the dictionary 
        # So it does not appear in the new labels matrix
        label_dict = {label: new_label for label, new_label in label_dict.items() if new_label != NoneFlock and new_label != DeadFlock}
        if DEBUG:
            print("-> Label dict: ", label_dict)
        for label in label_dict:
            mask = labels_after == label
            new_labels_matrix[i+1,mask] = label_dict[label]
        if DEBUG:
            print("-> Former labels matrix: ", Counter(df_labels.iloc[i+1,:]))
            print("-> New labels matrix: ", Counter(new_labels_matrix[i+1,:]))

        # new labels does not contain NoneFlock and DeadFlock
        assert NoneFlock not in new_labels_matrix, "Erreur d'assignation des clusters (NoneFlock)"
        assert DeadFlock not in new_labels_matrix, "Erreur d'assignation des clusters (DeadFlock)"
        if DEBUG:
            wait = input("Press enter to continue...")


    # Save the new labels matrix 
    df_new_labels = pd.DataFrame(new_labels_matrix)
    return df_new_labels


