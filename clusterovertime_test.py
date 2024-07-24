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
import animation.Animator2D as Animator2D
import animation.MatplotlibAnimator as MatplotlibAnimator

# True if we want to print debug information
DEBUG = False

NoneFlock = -2




def optimal_assignment(labels_before, labels_after):
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
    label_list_before = list(set(labels_before))
    label_list_after = list(set(labels_after))
    if -1 in label_list_before:
        label_list_before.remove(-1)
    if -1 in label_list_after:
        label_list_after.remove(-1)

    # Corner cases : no cluster in the previous or next frame
    if len(label_list_after) == 0:
        # No cluster in the next frame
        return labels_before, np.repeat(NoneFlock, len(labels_before)), np.zeros(len(labels_before))
    
    if len(label_list_before) == 0:
        # No cluster in the previous frame
        return np.repeat(NoneFlock, len(labels_after)), labels_after, np.zeros(len(labels_after))
    # label_list_before.remove(-1) # Remove the noise label
    # label_list_after.remove(-1) # Remove the noise label

    if DEBUG:
        print("Labels before: ", label_list_before)
        print("Labels after: ", label_list_after)

    # Count the clusters 
    n_cluster_before = len(set(labels_before)) - 1
    n_cluster_after = len(set(labels_after)) - 1

    # we use counters because the labels are keys to the dictionary of the counter
    cluster_counts_after = Counter(labels_after)
    if DEBUG: 
        print("# Cluster sizes after: \n", cluster_counts_after)
        print("Noise count after", cluster_counts_after[-1])

    # Count the number of clusters
    df_crosstab = pd.crosstab(pd.Series(labels_before), pd.Series(labels_after))
    if DEBUG:
        print("# Count matrix: \n", df_crosstab)

    # Delete noise column and line 
    count_matrix = df_crosstab.drop(-1, axis=0)
    count_matrix = count_matrix.drop(-1, axis=1)
    labels_after_assigned = count_matrix.columns.to_numpy()
    labels_before_assigned = count_matrix.index.to_numpy()
    if DEBUG: 
        print("# Count matrix after deletion: \n", count_matrix)
        print("Column names: ", labels_after_assigned)
        print("Index names: ", labels_before_assigned)

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
    cost_matrix = count_matrix.to_numpy() 
    if DEBUG:
        print("Cost matrix: ", cost_matrix) 

    # Convert the matrix into a square matrix 
    if cost_matrix.shape[0] < cost_matrix.shape[1]: # more clusters in the next frame
        diff = cost_matrix.shape[1] - cost_matrix.shape[0] # More columns than rows
        cost_matrix = np.concatenate((cost_matrix, np.zeros((diff, cost_matrix.shape[1]))), axis=0)
    elif cost_matrix.shape[0] > cost_matrix.shape[1]: # more clusters in the previous frame
        diff = cost_matrix.shape[0] - cost_matrix.shape[1] # More rows than columns
        cost_matrix = np.concatenate((cost_matrix, np.zeros((cost_matrix.shape[0], diff))), axis=1)

    # Hungarian algorithm - Optimal assignment of the clusters from the previous frame to the next frame
    assigned_row_indices, assigned_col_indices = linear_sum_assignment(cost_matrix, maximize=True)
    if DEBUG :
        print("Before index: ", assigned_row_indices)
        print("After  index: ", assigned_col_indices)
        print("Before Labels :", labels_before_assigned)
        print("After  Labels :", labels_after_assigned)
        print("Cost matrix: ", cost_matrix[assigned_row_indices, assigned_col_indices])
    contribution = cost_matrix[assigned_row_indices, assigned_col_indices]

    
    # Delete the clusters we added by making the cost matrix square
    if n_cluster_before < n_cluster_after:
        labels_before_assigned = np.concatenate((labels_before_assigned, np.zeros(n_cluster_after - n_cluster_before) - 2)).astype(int)
    elif n_cluster_before > n_cluster_after:
        labels_after_assigned = np.concatenate((labels_after_assigned, np.zeros(n_cluster_before - n_cluster_after) - 2)).astype(int)

    if DEBUG:
        print("Optimal assignment done.")
    
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
        Flocks.id_counter += 1
        return new_id

    def get_last_flock_id(self):
        return Flocks.id_counter - 1
    
    def kill_flock(self, id):
        self.isAlive[id] = False

    def reset_id_counter(self):
        Flocks.id_counter = 0
    
    def __str__(self) -> str:
        txt = "Flocks: \n"
        for id in self.id_list:
            if self.isAlive[id]:
                txt += f"Flock {id} ; [ALIVE] "
            else: 
                txt += f"Flock {id} ; [DEAD] "
            txt += "\n"
        
        return txt

    def print_tab(self):
        txt = f'Flocks: {self.id_list} \n'
        txt += f'Is alive: {self.isAlive} \n'
        return txt


    


if __name__ == "__main__":
    ### === Importing the data === ###
    # Get the data from the .csv files 
    print("Importing data...")
    df = pd.read_csv('data/vicsek.csv')
    df_labels = pd.read_csv('data/vicsek_labels.csv')

    if DEBUG:
        # Test if we extracted the data correctly
        print("Dataframe shape: ", df.shape)
        print("Dataframe labels shape: ", df_labels.shape)

        print("Dataframe head: ", df.head())
        print("Dataframe labels head: ", df_labels.head())


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

    # Create the flock objects
    print("Creating the flock objects...")
    flocks = Flocks()
    flocks.reset_id_counter()
    for i in range(len(set(labels_before)) - 1):  
        new = flocks.add_flock()

    # print("Flock objects created.", flocks) 

    for i in range(df_labels.shape[0] - 1):
        ### === Iteration of optimal assignment === ###
        print(f'####################### Iteration {i} #######################')
        print(f"Labels after counter: {Counter(df_labels.iloc[i+1,:])}")
        print(f"Optimal assignment iteration {i}...")
        # # Optimal assignment : computes the best permutation of the labels after the optimal assignment. 
        labels_before = new_labels_matrix[i,:]
        labels_after = df_labels.iloc[i+1,:].to_numpy()
        before, after, contribution = optimal_assignment(labels_before, labels_after)
        # Build the permutation of the labels after the optimal assignment
        threshold = 0.
        replace_after = after.copy()
        DEBUG = True
        for id, (id_before, id_after, contrib) in enumerate(zip(before, after, contribution)):
        
            if DEBUG:
                print(f"{id} : Before: {id_before} ; After: {id_after} ; Contribution: {contrib}")
            print(flocks)
            print(flocks.print_tab())
            # CASES : 
            ## Contribution is below the threshold
            if id_before == NoneFlock and id_after != NoneFlock: 
                new = flocks.add_flock()
                replace_after[id_after] = new
                if DEBUG:
                    print("New cluster! : ", new)
            ## Cluster died
            elif id_after == NoneFlock and id_before != NoneFlock:
                if DEBUG:
                    print("Cluster died!")
                flocks.kill_flock(id_before)
                replace_after[id_after] = -1 # the flock is dead at next iteration
                # flocks.set_label_after(id_before, None)
            ## Cluster dies, another birth
            elif contrib <= threshold:
                flocks.kill_flock(id_before)
                new = flocks.add_flock()
                replace_after[id] = new
                if DEBUG:
                    print("Cluster died!, New cluster to be created.", new) 
            ## Cluster continues : contribution above threshold
            if contrib > threshold:
                if DEBUG:
                    print("Cluster continues!")
                replace_after[id] = id_before
                # flocks.set_label_after(id_before, id_after)
        print("-> Before permutation: ",after)
        print("-> After  permutation: ",replace_after)
        # label_dict = {label: new_label for label, new_label in zip(after, replace_after)}
        # t_start = time.time()
        # for label in label_dict:
        #     mask = labels_after == label
        #     new_labels_matrix[i+1,mask] = label_dict[label]
        # t_end = time.time()
        # print(f"Time taken (vector): {(t_end - t_start)*1000:.2f}ms")
        # print("-> Former labels matrix: ", Counter(df_labels.iloc[i+1,:]))
        # print("-> New labels matrix: ", Counter(new_labels_matrix[i+1,:]))
        # DEBUG = False
        if i >= 8:
            wait = input("Press enter to continue...")

