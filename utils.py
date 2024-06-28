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

def naive_clustering_labels_positions(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_pos = get_positions(df, i)
    df_pos_scaled = scaler.fit_transform(df_pos)
    db_pos = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_pos_scaled)
    return db_pos.labels_

def naive_clustering_labels_orientations(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_orient = get_orientations(df, i)
    df_orient_scaled = scaler.fit_transform(df_orient)
    db_orient = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_orient_scaled)
    return db_orient.labels_

def naive_clustering_labels_positions_and_orientations(df, i, threshold=0.3, min_samples=5):
    scaler = StandardScaler()
    df_all = get_positions_and_orientations(df, i)
    df_all_scaled = scaler.fit_transform(df_all)
    db_all = cluster.DBSCAN(eps=threshold, min_samples=min_samples).fit(df_all_scaled)
    return db_all.labels_

def clustering_labels_stats(labels):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return n_clusters_, n_noise

def coloring_clusters(labels, cmap_name='rainbow'):
    n = len(set(labels))
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(labels+1 / max(labels+1))
    return pd.DataFrame(colors)

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