import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils as utils

### Single bird visualisation ###

def plot_bird_path(df, i, L=50):

    """
    Plot the travelled path of the bird i in the flock over time.
    """

    # Extract the x and y coordinates of the bird over time.
    bird_x, bird_y = 'x'+str(i),'y'+str(i)
    x, y = df[[bird_x, bird_y]].to_numpy().transpose()

    # Plot the bird path.
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=10, 
             c=np.arange(len(x)), cmap='hot') 
    ax.scatter(x[0], y[0], s=30, 
             c='red', label='start') #startpoint
    ax.scatter(x[-1], y[-1], s=30, 
             c='blue', label='end') #endpoints
    ax.set_title(f'Trajector of the bird {i} in the flock')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect(True)
    ax.plot(x, y, c='k', alpha=0.1)
    ax.legend()
    plt.show()

    return fig, ax

def plot_bird_orientation(df, i):

    """
    Plot the travelled path of the bird.

    Parameters
    ----------
    df : pandas.DataFrame
        The simulation data.
    i : int
        The index of the bird.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    axs : matplotlib.axes.Axes
        The 3 axes containing the plot. 

    """
    bird_x, bird_y = 'theta_x_'+str(i), 'theta_y_'+str(i)
    x, y = df[[bird_x, bird_y]].to_numpy().transpose()
    angle = np.arctan2(y, x)
    fig, axs = plt.subplots(3)
    fig.suptitle('orientation of the bird over time iterations')
    axs[0].plot(x, label="cos_theta_x")
    axs[1].plot(y, label="sin_theta_y")
    axs[0].set(xlabel="time", ylabel="cos_theta")
    axs[1].set(xlabel="time", ylabel="sin_theta")
    axs[2].plot(angle, label="angle theta")
    axs[2].set(xlabel="time", ylabel="angle theta")   

    plt.show()
    return fig, axs

### Flock visualisation ###

def plot_simulation(df, i, L):

    """
    Plot the travelled path of the bird.
    """
    dt = df['t'].iloc[1] - df['t'].iloc[0]
    df_pos = utils.extract_positions_from_dataframe(df).iloc[i]
    df_orient = utils.extract_orientations_from_dataframe(df).iloc[i]

    n_bird = len(df_pos)//2
    
    list_pos = df_pos.to_numpy().reshape((n_bird, 2))
    list_orient = df_orient.to_numpy().reshape((n_bird, 2))

    x, y = list_pos[:,0], list_pos[:,1]
    theta_x, theta_y = list_orient[:,0], list_orient[:,1]

    fig, ax = plt.subplots()
    ax.quiver(x,y,theta_x,theta_y)
    ax.set_title(f'Simulation au temps t = {i*dt:.2f}s')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect(True)
    plt.show()
    return fig, ax

def plot_average_orientation(df):

    # Compute average orientation
    df_orient = utils.extract_orientations_from_dataframe(df)
    theta_x = df_orient.filter(like="theta_x").to_numpy()
    theta_y = df_orient.filter(like="theta_y").to_numpy()
    angle = np.arctan2(theta_y, theta_x)
    angle_mean = np.mean(angle, axis=1)
    angle_start = angle_mean[0]
    theta_x_mean = np.mean(theta_x, axis=1)
    theta_y_mean = np.mean(theta_y, axis=1)
    
    # Plot average orientations
    fig, axs = plt.subplots(3)
    axs[0].plot(theta_x_mean, label="cos_theta")
    axs[1].plot(theta_y_mean, label="sin_theta")
    axs[0].set(xlabel="time", ylabel="cos_theta")
    axs[1].set(xlabel="time", ylabel="sin_theta")
    axs[2].plot(angle_mean, label="angle_theta")
    axs[2].set(xlabel="time", ylabel="angle_theta")
    axs[2].axline([0, angle_start], slope=0, color="red", label="start angle mean")
    # axs[2].legend()
    plt.show()

    return fig, axs

def plot_average_position(df, L=50):
    # Compute the average position of the flock over time.
    df_pos = utils.extract_positions_from_dataframe(df)
    x = df_pos.filter(like="x").to_numpy()
    y = df_pos.filter(like="y").to_numpy()
    x_mean = np.mean(x, axis=1)
    y_mean = np.mean(y, axis=1)

    # Plot the average position of the flock over time.
    fig, ax = plt.subplots()
    ax.plot(x_mean, y_mean)
    ax.set_title('Average position of the flock over time')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    plt.show()

    return fig, ax


def plot_order_factor(df):
    order = utils.order_factor(df)
    fig, ax = plt.subplots()
    ax.plot(order)
    plt.title('Order factor over time')
    plt.xlabel('iterations')
    plt.ylabel('order factor')
    plt.ylim(0, 1)
    plt.show()

    return fig, ax

### Clustering visualisation ###

def plot_clusters(df, i, labels, title=None, L=50,
                    cmap_name='rainbow', kde=True):

    """
    Plot the simulation showing the clusters of birds at iteration i.
    """

    ### EXTRACTING THE DATA ###

    # Extract the positions and orientations of the birds at iteration i.
    df_pos = utils.extract_positions_from_dataframe(df).iloc[i]
    df_orient = utils.extract_orientations_from_dataframe(df).iloc[i]
    # Extract the number of birds 
    n_bird = len(df_pos)//2

    # Convert the dataframes to numpy arrays.
    list_pos = df_pos.to_numpy().reshape((n_bird, 2))
    list_orient = df_orient.to_numpy().reshape((n_bird, 2))
    # Extract the x and y coordinates of the birds, and the theta_x and theta_y components of their orientations.
    x, y = list_pos[:,0], list_pos[:,1]
    theta_x, theta_y = list_orient[:,0], list_orient[:,1]
    
    ### PLOTTING THE SIMULATION ###

    # Coloring the birds according to their cluster.
    colors = utils.coloring_clusters(labels, cmap_name).to_numpy()
    # Setting the title of the plot.
    if title is None:
        title = f'Simulation at the iteration i={i}'
    alpha = 1 # transparency of the arrows if kde is False

    # Plot the simulation.
    fig, ax = plt.subplots()
    if kde : 
        # kde without noise? 
        ax = sns.kdeplot(x=x, y=y, fill=True, hue = labels, alpha=0.5, palette=cmap_name, legend=False)
        # ax = sns.kdeplot(x=x, y=y, fill=True, alpha=0.5, palette=cmap_name, legend=False)
        alpha = 0.5 # transparency of the arrows if kde is True
    ax.quiver(x,y,theta_x,theta_y, color = colors, cmap=cmap_name, alpha=alpha)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect(True)
    plt.show()
    
    return fig, ax