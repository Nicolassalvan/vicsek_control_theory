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

# Module imports
import models.vicsek as vicsek
import visualisation as visualisation
import utils
import animation.Animator2D as Animator2D
import animation.MatplotlibAnimator as MatplotlibAnimator

global DEBUG
DEBUG = False 





def optimal_assignment(labels_before, labels_after):
    print("Optimal assignment...")
    label_list_before = list(set(labels_before))
    label_list_after = list(set(labels_after))
    label_list_before.remove(-1) # Remove the noise label
    label_list_after.remove(-1) # Remove the noise label
    print("Labels before: ", label_list_before)
    print("Labels after: ", label_list_after)

    # # Check if they have the same size 
    # if len(label_list_before) > len(label_list_after):
    #     diff = len(label_list_before) - len(label_list_after)
    #     # We add the missing labels to the list of labels after
    #     for label in label_list_before:
    #         if label not in label_list_after and diff > 0:
    #             label_list_after.append(label)
    #             diff -= 1
    # elif len(label_list_before) < len(label_list_after):
    #     # We add the missing labels to the list of labels before
    #     diff = len(label_list_after) - len(label_list_before)
    #     for label in label_list_after:
    #         if label not in label_list_before and diff > 0: 
    #             label_list_before.append(label)
    #             diff -= 1

    # We now have the same number of labels in both lists
    print("Labels before: ", label_list_before)
    print("Labels after: ", label_list_after)

    # # Count the clusters 
    n_cluster_before = len(set(labels_before)) - 1
    # print("# Number of clusters before: \n", n_cluster_before)
    # assert n_cluster_before == 1+max(set(new_labels_matrix[0,:])), f'Error in counting the clusters. Cluster before: {n_cluster_before}, max: {max(set(new_labels_matrix[0,:]))}'
    n_cluster_after = len(set(labels_after)) - 1
    # print("# Number of clusters after: \n", n_cluster_after)

    # Cluster sizes 
    # counter_before = Counter(labels_before)
    # print("# Cluster sizes before: \n", counter_before)
    # print("Noise count before", counter_before[-1])

    counter_after = Counter(labels_after)
    print("# Cluster sizes after: \n", counter_after)
    print("Noise count after", counter_after[-1])

    # Count the number of clusters
    df_crosstab = pd.crosstab(pd.Series(labels_before), pd.Series(labels_after))
    print("# Count matrix: \n", df_crosstab)

    # Delete noise column and line 
    df_count = df_crosstab.drop(-1, axis=0)
    df_count = df_count.drop(-1, axis=1)
    df_count_after = df_count.columns.to_numpy()
    df_count_before = df_count.index.to_numpy()

    print("# Count matrix after deletion: \n", df_count)
    print("Column names: ", df_count_after)
    print("Index names: ", df_count_before)

    # List of the number of clusters in the next frame to compute the contribution of each cluster to the next frame
    col_count = [counter_after[i] for i in df_count_after] # i is a key in the dictionary of the counter /!\
    print("Column count: ", col_count)

    # Normalise the count matrix by computing the contribution of each cluster to the next frame
    df_count = df_count.div(col_count, axis=1)
    print("# Normalised count matrix: \n", df_count)

    # Convert the matrix to a numpy array for the Hungarian algorithm
    cost_matrix = df_count.to_numpy() 
    if DEBUG:
        print("Cost matrix: ", cost_matrix) 

    # Convert the matrix into a square matrix 
    if cost_matrix.shape[0] < cost_matrix.shape[1]: # more clusters in the next frame
        diff = cost_matrix.shape[1] - cost_matrix.shape[0]
        cost_matrix = np.concatenate((cost_matrix, np.zeros((diff, cost_matrix.shape[1]))), axis=0)
        # add new labels to df_count_c
    elif cost_matrix.shape[0] > cost_matrix.shape[1]: # more clusters in the previous frame
        diff = cost_matrix.shape[0] - cost_matrix.shape[1]
        cost_matrix = np.concatenate((cost_matrix, np.zeros((cost_matrix.shape[0], diff))), axis=1)


    # Hungarian algorithm - Optimal assignment of the clusters from the previous frame to the next frame
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    print("Before index: ", row_ind)
    print("After  index: ", col_ind)
    print("Before Labels :", df_count_before)
    print("After  Labels :", df_count_after)
    print("Cost matrix: ", cost_matrix[row_ind, col_ind])
    contribution = cost_matrix[row_ind, col_ind]


    # df_count = df_count.apply(  )
    # print("# Normalised count matrix: \n", df_count)
    
    # Delete the clusters we added by making the cost matrix square
    if n_cluster_before < n_cluster_after:
        df_count_before = np.concatenate((df_count_before, np.zeros(n_cluster_after - n_cluster_before) - 2)).astype(int)
    elif n_cluster_before > n_cluster_after:
        df_count_after = np.concatenate((df_count_after, np.zeros(n_cluster_before - n_cluster_after) - 2)).astype(int)
    
    # Final labels 
    print("Before Labels :", df_count_before)
    print("After  Labels :", df_count_after)
    # print(df_crosstab)
    print("Optimal assignment done.")
    # Return the labels before and after the optimal assignment, and the contribution of each cluster to the next frame
    return df_count_before, df_count_after, contribution

class Flocks: 
    id_counter = 0
    def __init__(self):
        self.id_list = []
        self.isAlive = []
        self.label_after = []
        self.label_before = []
    
    def add_flock(self):
        self.id_list.append(Flocks.id_counter)
        self.isAlive.append(True)
        self.label_after.append(None) 
        self.label_before.append(None)

        Flocks.id_counter += 1
        return Flocks.id_counter - 1

    def get_flock_id_from_label(self, label):
        if label in self.label_after:
            return self.label_after.index(label)
        else:
            index = self.label_before.index(label)
            return self.label_after[index]

    def get_last_flock_id(self):
        return Flocks.id_counter - 1
    
    def kill_flock(self, id):
        self.isAlive[id] = False
    
    def set_label_after(self, id, label):
        self.label_after[id] = label

    def set_label_before(self, id, label):
        self.label_before[id] = label

    def reset_id_counter(self):
        Flocks.id_counter = 0
    
    def get_alive_flocks(self):
        return [id for id, isAlive in enumerate(self.isAlive) if isAlive]
    
    def get_dead_flocks(self):
        return [id for id, isAlive in enumerate(self.isAlive) if not isAlive]
    
    def __str__(self) -> str:
        txt = "Flocks: \n"
        for id in self.id_list:
            if self.isAlive[id]:
                txt += f"Flock {id} ; [ALIVE] ; "
            else: 
                txt += f"Flock {id} ; [DEAD] ; "
            txt += f"Label before: {self.label_before[id]} ; "
            txt += f"Label after: {self.label_after[id]} \n"

        return txt

    def set_next_labels(self, id, new_label):
        self.label_before[id] = self.label_after[id]
        self.label_after[id] = new_label

    


if __name__ == "__main__":
    ### === Importing the data === ###
    # Get the data from the .csv files 
    print("Importing data...")
    df = pd.read_csv('data/vicsek.csv')
    df_labels = pd.read_csv('data/vicsek_labels.csv')

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
    print("New labels matrix counter: ", Counter(new_labels_matrix[0,:]))

    # Create the labels
    labels_before = df_labels.iloc[0,:].to_numpy()  
    labels_after = df_labels.iloc[1,:].to_numpy()

    # Create the flock objects
    print("Creating the flock objects...")
    flocks = Flocks()
    for i in range(len(set(labels_before)) - 1):  
        new = flocks.add_flock()
        flocks.set_label_before(new, new)
    print("Flock objects created.", flocks) 


    # # Optimal assignment
    before, after, contribution = optimal_assignment(labels_before, labels_after)
    replace_after = after.copy()
    for id, (id_before, id_after, contrib) in enumerate(zip(before, after, contribution)):
        print(f"Before: {id_before} ; After: {id_after} ; Contribution: {contrib}")
        
        
        if id_before == -2:
            new = flocks.add_flock()
            replace_after[id_after] = new
            # flocks.set_label_after(flocks.get_last_flock_id(), flocks.get_last_flock_id())
            print("New cluster! : ", new)

        elif id_after == -2:
            print("Cluster died!")
            flocks.kill_flock(id_before)
            replace_after[id_after] = None # the flock is dead at next iteration
            # flocks.set_label_after(id_before, None)
        elif contrib == 0.:
            flocks.kill_flock(id_before)
            new = flocks.add_flock()
            replace_after[id] = new
            # flocks.set_label_after(id_before, new)
            # flocks.set_label_before(id_before, id_before)

            # flocks.set_label_after(new, new)
            print("Cluster died!, New cluster to be created.", new) 
            
        if contrib > 0:
            print("Cluster continues!")
            replace_after[id] = id_before
            # flocks.set_label_after(id_before, id_after)
    print("Results : ")
    print(after)
    print(replace_after)
    # # Permute the labels after 
    txt = ""
    for id, lab in enumerate(labels_after):
        txt += f"Label: {lab} ; "
        if lab != -1:
            idx = np.where(after == lab)[0][0]
            new_labels_matrix[1,id] = replace_after[idx]
            txt += f"New label: {new_labels_matrix[1,id]}"
        else:
            txt += f"New label: -1"
        txt += "\n"
    # save text
    with open("results.txt", "w") as f:
        f.write(txt)
