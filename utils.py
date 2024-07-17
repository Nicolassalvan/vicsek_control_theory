# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:11:06 2024
@author: Ian Soede (S4385764), Nicolas Salvan (S6155197) at University of Groningen, Biomimitecs group
"""
import numpy as np
import pandas as pd 
from scipy import integrate
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import matplotlib.pyplot as plt
from numba import jit
from sklearn.neighbors import NearestNeighbors

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

# def order_factor(df):
#     df_orient = extract_orientations_from_dataframe(df)
#     df_orient['order_factor'] = np.sqrt(df_orient.filter(like="theta_x").mean(axis=1)**2 + df_orient.filter(like="theta_y").mean(axis=1)**2)
#     return df_orient['order_factor']

# def stationnary_order_factor(df):
#     df_order = order_factor(df)
#     order = df_order.to_numpy()
#     int_order = integrate.quad(order, df['t'], initial=0)

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


# ============================================================================= #
# =============================  Data clustering  ============================= #
# ============================================================================= #

@jit(nopython=True)
def distance_periodic_scaled(X, Y, periods):
    dims = periods.shape[0]
    n = X.shape[0]
    dist = 0 
    for d in range(dims):
        dist += (np.abs(X[d] - Y[d]) / periods[d])**2
    return np.sqrt(dist)

# Position matrix
@jit(nopython=True)
def distance_periodic(X, Y, periods):
    dims = periods.shape[0]
    n = X.shape[0]
    dist = 0 
    for d in range(dims):
        delta = np.abs(X[d] - Y[d])
        if delta > periods[d] / 2:
            delta = periods[d] - delta
        dist += (delta)**2
    return np.sqrt(dist)

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

# DOESNT WORK
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

# ============================================================================= #
# ============================= Data manipulation ============================= #
# ============================================================================= #

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
    
    t, positionsHistory, orientationsHistory = simulationData
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

def get_positions_and_angles(df, i):
    pos = get_positions(df, i)
    orient = get_orientations(df, i)
    angle = np.arctan2(orient['theta_y'], orient['theta_x']) + np.pi
    df_all = pd.DataFrame(np.concatenate((pos, angle[:,np.newaxis]), axis=1), columns=['x', 'y', 'angle'])
    return df_all

def get_angles(df, i):
    orient = get_orientations(df, i)
    angle = np.arctan2(orient['theta_y'], orient['theta_x']) + np.pi
    return pd.DataFrame(angle, columns=['angle'])