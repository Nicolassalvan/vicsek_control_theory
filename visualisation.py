import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils as utils
from matplotlib.animation import FuncAnimation
import os 


### Animation ### 

def animate_simulation2D_colored(df, labels,  L, plotBool = False, annotateBool = False, fps = 30, saveFolder = None):
    saveInFolderBool = False
    if saveFolder is not None:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)    
        saveInFolderBool = True

    interval = int(1000/fps)
    df_x = df.filter(regex='^x')
    df_y = df.filter(regex='^y')
    df_theta_x = df.filter(regex='^theta_x')
    df_theta_y = df.filter(regex='^theta_y')
    
    time = df['t'].to_numpy()

    x, y = df_x.to_numpy(), df_y.to_numpy()
    theta_x, theta_y = df_theta_x.to_numpy(), df_theta_y.to_numpy()

    labels = labels.to_numpy()  
        # Animation 
    fig, ax = plt.subplots()


    def update_quiver_with_colors(i):
        ax.cla()  # Effacer les anciennes quivers
        arrow_x, arrow_y = x[i], y[i]
        arrow_u, arrow_v = theta_x[i], theta_y[i]
        color = labels[i] / labels.max()
        noise_mask = labels[i] == -1
        color[noise_mask] = -1
        color = color / 2 + 0.5

        ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v, color = plt.cm.viridis(color))
        if annotateBool:
            for j, txt in enumerate(labels[i]):
                ax.annotate(txt, (x[i][j], y[i][j]))

        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_title('$t$=%2.2f' % time[i])
        if saveInFolderBool:
            fig.savefig(saveFolder + f'frame_{i}.png')

    ani = FuncAnimation(fig, update_quiver_with_colors, frames = len(x), repeat=False , interval = interval)
    if plotBool:
        plt.show()
    return ani

def animate_simulation2D(df, L, plotBool = False, fps = 30, saveFolder = None):
    saveInFolderBool = False
    order = utils.order_factor(df)
    if saveFolder is not None:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)    
        saveInFolderBool = True

    interval = int(1000/fps)

    df_x = df.filter(regex='^x')
    df_y = df.filter(regex='^y')
    df_theta_x = df.filter(regex='^theta_x')
    df_theta_y = df.filter(regex='^theta_y')
    
    time = df['t'].to_numpy()

    x, y = df_x.to_numpy(), df_y.to_numpy()
    theta_x, theta_y = df_theta_x.to_numpy(), df_theta_y.to_numpy()


    print("Data created. Now plotting...")
    # Animation 
    fig, ax = plt.subplots()


    def update_quiver(i):
        ax.cla()  # Effacer les anciennes quivers
        arrow_x, arrow_y = x[i], y[i]
        arrow_u, arrow_v = theta_x[i], theta_y[i]

        ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_title(f'$t$={time[i]:.2f}, order factor = {order[i]:.2f}')
        if saveInFolderBool:
            fig.savefig(saveFolder + f'frame_{i}.png')

    ani = FuncAnimation(fig, update_quiver, frames = len(x), repeat=False, interval = interval)
    if plotBool:
        plt.show()
    return ani


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
    # ax.set_aspect(True)
    ax.plot(x, y, c='k', alpha=0.1)
    ax.legend()
    # plt.show()

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

    # plt.show()
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
    # ax.set_aspect(True)
    # plt.show()
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


    return fig, axs

def plot_average_angle(df):
    angles = utils.get_flock_orientation(df)
    fig, ax = plt.subplots()
    ax.plot(angles.mean(axis=1))
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('Time (iteration)')
    ax.set_ylabel('Average orientation (rad)')
    return fig, ax

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
    # plt.show()

    return fig, ax


def plot_order_factor(df):
    order = utils.order_factor(df)
    fig, ax = plt.subplots()
    ax.plot(order)
    plt.title('Order factor over time')
    plt.xlabel('iterations')
    plt.ylabel('order factor')
    plt.ylim(0, 1)
    # plt.show()

    return fig, ax

def plot_series_mean_smoothed(series, smoothing_idx, title, xlabel, ylabel, serieslabel='Original', smoothedlabel='Smoothed'):
    fig, ax = plt.subplots()
    series_smoothed = utils.smoothing(series, smoothing_idx)
    ax.plot(series, label=serieslabel, alpha = 0.7)
    if smoothing_idx > 0:
        ax.plot(series_smoothed, label=smoothedlabel, alpha = 0.7)
    ax.plot(np.ones(len(series))*np.mean(series), label='Mean', linestyle='--')
    ax.legend()
    ax.set_title(title) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_error_over_time(df, radius=1):
    df_error = utils.getError(df, radius)
    mean = df_error.mean(axis=1)
    std = df_error.std(axis=1)
    # PLOTTING
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(mean, label='Mean')
    ax.fill_between(mean.index, mean-std, mean+std, alpha=0.5, label='Standard deviation')
    ax.set_xlabel('Time (iterations)')
    ax.set_ylabel('Error (rad)')
    ax.set_title('Error over time')
    ax.set_ylim(-np.pi, np.pi)
    ax.legend()
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
    # plt.show()
    
    return fig, ax


def plot_cluster_lifespan(df_optimal_labels):
    is_alive = utils.living_clusters(df_optimal_labels)
    max_label = np.max(df_optimal_labels.to_numpy())
    fig, ax = plt.subplots()
    for cluster in range(max_label+1):
        x = np.arange(df_optimal_labels.shape[0])
        x = x[is_alive.T[cluster] == True]
        y = is_alive.T[cluster][is_alive.T[cluster] == True] * (cluster)
        ax.plot(x, y, 'x', label=cluster)
    ax.set(xlabel='Time Iterations', ylabel='Cluster Label',
              title='Life span of clusters',
              yticks=np.arange(max_label+1))
    return fig, ax

def barplot_cluster_lifespan(df_optimal):
    max_label = np.max(df_optimal.to_numpy())
    _, life_span_all = utils.life_span(df_optimal)
    life_span_all = [item for sublist in life_span_all for item in sublist]
    # print(life_span_all)
    fig, ax = plt.subplots()
    ax.bar(range(max_label+1), life_span_all, align='center')
    ax.set(xlabel='Cluster Label', ylabel='Life Span (iterations)',
              title='Life span of clusters',
              xticks=range(max_label+1))
    # ax.set_xticks(range(max_label+1))
    return fig, ax


def plot_life_span_distribution(life_span_all, hist_bins = 20):
    fig, ax = plt.subplots()
    n, bins, _ = ax.hist(life_span_all, bins=hist_bins, align='mid', alpha=0.75, facecolor='b')
    ax.set(
        xlabel='Life Span (iterations)',
        ylabel='Frequency',
        title='Histogram of life span of clusters',
    )
    ax.set_xticks(bins, minor=True)
    return fig, ax  

def plot_Kmeans_inertia(life_span, K_max=10):
    # Determine the optimal number of clusters using the elbow method
    K_max = min(K_max, len(np.array(life_span)))
    inertias = utils.cluster_inertia(life_span, K_max)
    K = range(1, K_max+1)

    # Plot the elbow method
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    ax.plot(K, inertias, 'bx-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method showing the optimal k')
    return fig, ax

def plot_categorize_flock(life_span, optimal_k):
    clustering_stats = utils.categorize_flock(life_span, optimal_k)
    # plot of mean, std and values 
    fig, ax = plt.subplots()
    # ax.bar(range(optimal_k), clustering_stats[:, 0], align='center', label="Size")
    ax.bar(range(optimal_k), clustering_stats[:, 1], align='center', alpha=0.05)
    ax.errorbar(range(optimal_k), clustering_stats[:, 1], yerr=clustering_stats[:, 2], fmt='o', label="Mean +/- std", color='red')
    ax.set_xticks(range(optimal_k))
    ax.set_title(f"Clustering of life span (k = {optimal_k})")
    ax.set_xlabel("Category of cluster")
    ax.set_xticks([])
    ax.set_ylabel("Life span")
    ax.legend()
    return fig, ax

def barplot_bird_time_in_cluster(df_optimal, cluster):
    birds_in_cluster = (df_optimal == cluster)
    bird_cluster_span = birds_in_cluster.astype(int).sum(axis=0).to_numpy()
    fig, ax = plt.subplots()
    ax.bar(range(bird_cluster_span.shape[0]), bird_cluster_span)
    ax.set_xlabel("Bird ID")
    ax.set_xticks(range(bird_cluster_span.shape[0]), minor=True)
    ax.set_ylabel("Time in cluster (iterations)")
    ax.set_ylim(0, df_optimal.shape[0])
    ax.set_title(f"Time of birds lasting in cluster {cluster}")
    return fig, ax

def plot_bird_time_in_cluster(df_optimal, cluster):
    """
    Plot the time of birds lasting in a cluster over time. On the y-axis, the bird ID is displayed. 
    """
    birds_in_cluster = (df_optimal == cluster)
    # bird_cluster_span = birds_in_cluster.astype(int).sum(axis=0).to_numpy()

    fig, ax = plt.subplots()
    for bird in range(df_optimal.shape[1]):
        mask = birds_in_cluster.iloc[:, bird].to_numpy().astype(bool)
        x = np.arange(df_optimal.shape[0])[mask]
        y = np.ones_like(x) * bird
        ax.plot(x, y, 'x')
    ax.set(
        title = f"Time of birds lasting in cluster {cluster}",
        xlabel = "Time (Iterations)",
        ylabel = "Bird ID"
    )
    ax.set_yticks(range(df_optimal.shape[1]), minor=True)
    return fig, ax

def plot_cluster_orientation(df, df_labels, cluster):
    mean_orientation = utils.mean_orientation_cluster(df, df_labels, cluster)
    fig, ax = plt.subplots()
    ax.plot(mean_orientation)
    ax.set(
        title = f"Mean orientation of cluster {cluster}",
        xlabel = "Time (Iterations)",
        ylabel = "Mean orientation", 
        ylim = (-np.pi, np.pi)
    )
    return fig, ax

def plot_cluster_position(df, df_labels, cluster):
    mean_x, mean_y = utils.mean_position_cluster(df, df_labels, cluster)
    fig, ax = plt.subplots(2)
    ax[0].plot(mean_x)
    ax[0].set_xlabel("Time (iterations)")
    ax[0].set_ylabel("Mean x position")
    ax[1].plot(mean_y)
    ax[1].set_xlabel("Time (iterations)")
    ax[1].set_ylabel("Mean y position")
    fig.suptitle(f"Mean position of cluster {cluster}")
    return fig, ax


def plot_bird_correlation_with_mean(df):

    correlation_with_mean = utils.bird_correlation_with_mean(df)

    fig, ax = plt.subplots()
    ax.bar(range(correlation_with_mean.shape[0]), correlation_with_mean)
    ax.set(
        title = "Bird correlation with mean orientation",
        xlabel = "Bird ID",
        ylabel = "Correlation with mean orientation",
        ylim = (-1, 1)
    )
    ax.set_xticks(range(correlation_with_mean.shape[0]), minor=True)
    return fig, ax

def plot_cross_correlation(df, df_labels, cluster, bird, max_lag):
    mean_orientation = utils.mean_orientation_cluster(df, df_labels, cluster)
    flock_orientation = utils.get_flock_orientation(df)
    lags, corr = utils.cross_correlation(mean_orientation, flock_orientation[:, bird], max_lag)
    fig, ax = plt.subplots()
    ax.plot(lags, corr)
    ax.set(
        xlabel='Lag',
        ylabel='Correlation',
        title=f'Cross-Correlation of bird {bird} with cluster {cluster}'
    )
    return fig, ax


def plot_cluster_lag_on_mean(df, df_labels, cluster, max_lag=300):
    lag_estimation = utils.compute_lag_estimation_on_mean(df, df_labels, cluster, max_lag)
    fig, ax = plt.subplots()
    ax.hist(lag_estimation)
    ax.set(
        xlabel='Lag',
        ylabel='Number of birds',
        title=f'Lag estimation of birds in cluster {cluster}'
    )
    return fig, ax


def heatmap_granger_causality(df, max_lag=5):
    result_matrix = utils.granger_causality_matrix_flock(df, max_lag)
    fig, ax = plt.subplots()
    ax = sns.heatmap(result_matrix, annot=False, cmap='coolwarm', cbar=True, ax=ax)
    ax.set(title='Granger Causality Matrix')
    return fig, ax

def heatmap_granger_causality_significant(df, max_lag=5, significance_level=0.05):
    causal_relations = utils.granger_causality_matrix_significant(df, max_lag, significance_level)
    fig, ax = plt.subplots()
    ax = sns.heatmap(causal_relations, annot=False, cmap='coolwarm', cbar=True, ax=ax)
    ax.set(title='Significant Causal Relations')
    return fig, ax

def plot_most_influent(df, max_lag=5):
    print("Calculating Granger Causality Matrix...")
    causal_relations = utils.granger_causality_matrix_significant(df, max_lag=max_lag)
    influence_scores = causal_relations.sum(axis=0)
    most_influential_series = influence_scores.idxmax()
    print(f'The bird that has the most influence : {most_influential_series}')
    print(f'It influences {influence_scores.max()} other series.')
    # influence on others (p-values)
    result_matrix = utils.granger_causality_matrix_flock(df)
    p_values_most_influential = result_matrix[most_influential_series]

    print("Plotting...")
    fig, ax = plt.subplots()
    ax.plot(p_values_most_influential, 'x')
    ax.plot([0.05]*len(p_values_most_influential), linestyle='--')
    ax.set(
        title=f'Influence of bird {most_influential_series}',
        xlabel='Bird ID',
        ylabel='p-value'
    )
    ax.set_xticks(range(len(p_values_most_influential)), minor=True)
    return fig, ax


def plot_most_influent_bird(df, df_label, cluster):
    # COMPUTATION
    causal_relations = utils.granger_causality_matrix_significant(df)
    influence_scores = causal_relations.sum(axis=0)

    most_influential_bird = influence_scores.idxmax()
    mean_orientation = utils.mean_orientation_cluster(df, df_label, cluster)
    flock_orientation = utils.get_flock_orientation(df)
    bird_orientation = flock_orientation[:, most_influential_bird]
    # PLOTTING
    fig, ax = plt.subplots()
    ax.plot(mean_orientation, label='Mean orientation', alpha=0.8)
    ax.plot(bird_orientation, label='Bird orientation', alpha=0.7)
    ax.set(
        xlabel='Time (iterations)',
        ylabel='Orientation (radians)',
        title=f'Influence of bird {most_influential_bird} on cluster {cluster}'
    )
    ax.legend()
    return fig, ax


def plot_cross_correlation_most_influential_bird(df, df_labels, cluster, max_lag=50):
    # COMPUTATION
    causal_relations = utils.granger_causality_matrix_significant(df)
    influence_scores = causal_relations.sum(axis=0)
    most_influential_bird = influence_scores.idxmax()
    # PLOTTING
    fig, ax = plot_cross_correlation(df, df_labels, cluster, most_influential_bird, max_lag)
    return fig, ax

def plot_bird_difference_with_cluster(df, df_labels, cluster):
    causal_relations = utils.granger_causality_matrix_significant(df)
    influence_scores = causal_relations.sum(axis=0)
    most_influential_bird = influence_scores.idxmax()
    mean_orientation = utils.mean_orientation_cluster(df, df_labels, cluster)
    flock_orientation = utils.get_flock_orientation(df)
    bird_orientation = flock_orientation[:, most_influential_bird]
    fig, ax = plt.subplots()
    ax.plot(np.abs(mean_orientation - bird_orientation), label = 'Absolute difference')
    ax.set(
        xlabel='Time (iterations)',
        ylabel='Absolute difference',
        title=f'Absolute difference between bird {most_influential_bird} and cluster {cluster}'
    )
    return fig, ax


def plot_bird_causality_with_mean(df):
    flock_orientation = utils.get_flock_orientation(df)

    causal_relation_with_mean = utils.granger_causality_mean(pd.DataFrame(np.nan_to_num(flock_orientation)), max_lag=10)
    significant_relations = causal_relation_with_mean < 0.05 
    significant_birds = np.arange(len(causal_relation_with_mean))[significant_relations]
    least_significant_birds = np.arange(len(causal_relation_with_mean))[~significant_relations]
    print(f'Significant birds: {significant_birds}')
    print(f'Least significant birds: {least_significant_birds}')
    influential_bird_percentage = len(significant_birds) / len(causal_relation_with_mean) * 100
    print(f'{influential_bird_percentage}% of birds have a significant influence on the mean orientation.')
    fig, ax = plt.subplots()
    ax.scatter(significant_birds, causal_relation_with_mean[significant_birds], color='red', label='Significant')
    ax.scatter(least_significant_birds, causal_relation_with_mean[least_significant_birds], color='blue', label='Least significant')
    ax.set(
        xlabel='Bird ID',
        ylabel='p-value',
        title='Granger Causality of birds on mean orientation'
    )
    ax.legend()
    ax.plot([0.05]*len(causal_relation_with_mean), linestyle='--')
    return fig, ax

def heatmap_lag_matrix(df, max_lag=5):
    lagMat = utils.compute_lag_matrix(df, max_lag)
    fig, ax = plt.subplots()
    sns.heatmap(lagMat, annot=False, cmap='coolwarm', cbar=True, ax=ax)
    ax.set(
        title='Lag Matrix',
        xlabel='Bird ID',
        ylabel='Bird ID'
    )
    return fig, ax

def hist_lag_matrix(df):
    lag_matrix = utils.compute_lag_matrix(df)
    df_lag = pd.DataFrame(lag_matrix)
    fig, ax = plt.subplots()
    ax.hist(df_lag.mean().to_numpy())
    ax.set(
        title='Mean lag distribution',
        xlabel='Lag',
        ylabel='Frequency'
    )
    return fig, ax

def plot_cluster_size_over_time(df_labels, smoothness = 20):
    # COMPUTATION
    eff, noise = utils.cluster_effectif(df_labels)
    max_cluster = np.max(df_labels.to_numpy())
    # PLOTTING 
    fig, ax = plt.subplots()
    for i in range(max_cluster+1):
        ax.plot(utils.smoothing(eff[:, i], smoothness), label=f'Cluster {i}')
    ax.plot(utils.smoothing(noise, smoothness), label='Noise', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of particles')
    ax.set_title('Number of particles in each cluster over time')
    return fig, ax 