# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:11:06 2024
@author: Ian Soede (S4385764), Nicolas Salvan (S6155197) at University of Groningen, Biomimitecs group
"""
import numpy as np
import pandas as pd 
from scipy import integrate

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

def order_factor(df):
    df_orient = utils.extract_orientations_from_dataframe(df)
    df_orient['order_factor'] = np.sqrt(df_orient.filter(like="theta_x").mean(axis=1)**2 + df_orient.filter(like="theta_y").mean(axis=1)**2)
    return df_orient['order_factor']

def stationnary_order_factor(df):
    df_order = order_factor(df)
    order = df_order.to_numpy()
    int_order = integrate.quad(order, df['t'], initial=0)


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
