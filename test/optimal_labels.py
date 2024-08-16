# General imports
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from IPython.display import Video, display
from collections import Counter
from scipy.optimize import linear_sum_assignment
import time

# Module imports
import models.vicsek as vicsek
import visualisation as visualisation
import utils


### === Constants === ###
# True if we want to print debug information
DEBUG = False
# Constants for the NoneFlock and DeadFlock
NoneFlock = -2
DeadFlock = -3
# Periodic boundary conditions
L = 10


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

def centroid_distance_matrix(labels_before, labels_after, df_pos_before, df_pos_after, L = L): 
    # Creating centroids 
    gb_centroids_before = utils.get_centroids(df_pos_before, labels_before)
    gb_centroids_after = utils.get_centroids(df_pos_after, labels_after)

    dist = np.zeros((len(gb_centroids_before), len(gb_centroids_after)))
    labels_before = gb_centroids_before.index
    labels_after = gb_centroids_after.index
    for i in range(len(gb_centroids_before)):
        for j in range(len(gb_centroids_after)):
            x_i, y_i = gb_centroids_before.iloc[i].x, gb_centroids_before.iloc[i].y
            x_j, y_j = gb_centroids_after.iloc[j].x, gb_centroids_after.iloc[j].y
            dist[i,j] = utils.distance_periodic(np.array([x_i, y_i]), np.array([x_j, y_j]), np.array([L, L]))

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

def correct_labels_optimal_assignment(df_labels_to_copy, df, option: str):
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
    df_pos_before = utils.get_positions(df, 0)
    df_pos_after = utils.get_positions(df, 1)

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
        df_pos_before = utils.get_positions(df, i)
        df_pos_after = utils.get_positions(df, i+1)

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
    

# TESTS 
if __name__ == "__main__":
    # ### === Importing the data === ###
    # Get the data from the .csv files 
    print("Importing data...")
    df = pd.read_csv("data_test/vicsektest.csv")
    df_labels = pd.read_csv("data_test/vicsek_labels_test.csv")
    print("Data imported.")

    # Save the new labels matrix 
    print("Computing the new labels matrix with optimal assignment...")
    df_new_labels = correct_labels_optimal_assignment(df_labels, df, option="contribution")
    df_new_labels.to_csv("data/new_labels_matrix.csv", index=False)
    df_new_labels.to_csv("data_test/new_labels_matrix.csv", index=False)
    print("New labels matrix saved.")