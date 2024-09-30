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
import os
os.environ["OMP_NUM_THREADS"] = '1'

# Module imports
sys.path.append('..')
import models.vicsek as vicsek
import models.pid as pid
import visualisation as visualisation
import utils


################################################################################
# =============================== Utils ====================================== #
################################################################################
def progress(percent=0, width=30):
    # Progress bar animation 
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)
 
################################################################################
# ============================= Binder Cumulant ============================== #
################################################################################

# Study one parameter of the simulation, one simulation per parameters tuple 
def plot_binder_cumulant():
    """
    Here is an example of how to plot some float coefficient depending on one parameter. You can modify the parameters to study, 
    and also the parameters that are fixed. To delete the progress bar, you can delete the line 'progress(i/constNoiseNumberOfPoints*100)'.
    You can also customize the figure with the 'ax.set' function.
    It is possible to save the figure with the 'fig.savefig' function, and to modify the image description with the 'imageDescription' variable. 
    It will save the figure in the current directory, and you can change the name of the file by changing the string in the 'fig.savefig' function.
    """
    ### PARAMETERS ###
    N = 50
    L = 10
    radius = 1 
    dt = 0.5
    tmax = 500
    n_simulations = 20 # Nombre de simulations pour chaque couple de paramètres
    ### PARAMETERS TO STUDY ###
    constNoiseMin, constNoiseMax = 0, 2
    constNoiseNumberOfPoints = 100
    noiseArr = np.linspace(constNoiseMin, constNoiseMax, constNoiseNumberOfPoints)
    binderArr = np.zeros((n_simulations, constNoiseNumberOfPoints))

    ### COMPUTATION ###
    # Compute binder cumulant 
    print(f"Computing {n_simulations * constNoiseNumberOfPoints} simulations...")
    for i, noiseAmplitude in enumerate(noiseArr) : # for each parameter tuple
        for sample in range(n_simulations):
            # Create the model 
            model = vicsek.Vicsek(numberOfParticles=N, domainSize=(L, L), radius=radius, noiseAmplitude=noiseAmplitude, speed=0.3)
            # Simulate the model
            simulationData = model.simulate(dt=dt, tmax=tmax)
            df = utils.simulationDataToDataframe(simulationData)
            # Compute the Order Factor
            binderArr[sample][i] = utils.stationnary_order_factor(df)
        percentage = (i/constNoiseNumberOfPoints)*100
        progress(int(percentage))
    print("Done B)")
    # look for a critical point 
    binderMean = np.mean(binderArr, axis=0)
    binderSTD = np.std(binderArr, axis=0)


    ### PLOT ###
    fig, ax = plt.subplots()
    ax.plot(noiseArr, binderMean, label = 'Binder Cumulant') 
    ax.errorbar(noiseArr, binderMean, yerr = binderSTD)

    # Customizing the plot 
    ax.set(
        xlabel='Noise Amplitude',
        ylabel='Binder Cumulant',
        title='Binder Cumulant as a function of Noise Amplitude'
    )
    # ax.scatter(criticalNoise, criticalBinder, color='red', label=f'Crit: {criticalNoise:.2f}, {criticalBinder:.2f}')
    ax.legend()


    ### SAVING PLOT ###
    # ADD DESCRIPTION : Parameters, etc.
    imageDescription = 'Binder Cumulant as a function of Noise Amplitude \n'
    imageDescription += 'N = ' + str(N) + ', L = ' + str(L) + ', radius = ' + str(radius) + ', dt = ' + str(dt) + ', tmax = ' + str(tmax)
    imageDescription += 'rho = ' + str(N/(L*L)) + '\n'

    fig.text(0, 0, imageDescription, va='top')
    fig.tight_layout()
    fig.savefig('binder_cumulant.png', bbox_inches='tight')
    plt.show(block = False)

    ### SAVING IN .CSV ### 
    # df_res = pd.DataFrame({'Noise Amplitude': noiseArr, 'Binder Cumulant': binderArr})
    # df_res.to_csv('binder_cumulant.csv', index=False)

################################################################################
# ========================= Stationnary Order Factor ========================= #
################################################################################
# Study one parameter of the simulation, n_simulations simulations for each parameter tuple 
def plot_stationnary_order():
    """
    Here is an example of how to plot some float coefficient depending on one parameter. You can modify the parameters to study, 
    and also the parameters that are fixed. To delete the progress bar, you can delete the line 'progress(i/constNoiseNumberOfPoints*100)'.
    You can also customize the figure with the 'ax.set' function.
    It is possible to save the figure with the 'fig.savefig' function, and to modify the image description with the 'imageDescription' variable. 
    It will save the figure in the current directory, and you can change the name of the file by changing the string in the 'fig.savefig' function.
    """
    ### PARAMETERS ###
    N = 50
    L = 10
    radius = 1 
    dt = 0.5
    tmax = 10
    n_simulations = 20 # Nombre de simulations pour chaque couple de paramètres 
    ### PARAMETERS TO STUDY ###
    constNoiseMin, constNoiseMax = 0, 2
    constNoiseNumberOfPoints = 100
    noiseArr = np.linspace(constNoiseMin, constNoiseMax, constNoiseNumberOfPoints)
    orderArr = np.zeros((n_simulations, constNoiseNumberOfPoints))
    
    ### COMPUTATION ###
    # Compute binder cumulant 
    print(f"Computing {n_simulations * constNoiseNumberOfPoints} simulations...")
    for i, noiseAmplitude in enumerate(noiseArr) : # for each parameter tuple
        for sample in range(n_simulations):
            # Create the model 
            model = vicsek.Vicsek(numberOfParticles=N, domainSize=(L, L), radius=radius, noiseAmplitude=noiseAmplitude, speed=0.3)
            # Simulate the model
            simulationData = model.simulate(dt=dt, tmax=tmax)
            df = utils.simulationDataToDataframe(simulationData)
            # Compute the Order Factor
            orderArr[sample][i] = utils.stationnary_order_factor(df)
        percentage = (i/constNoiseNumberOfPoints)*100
        progress(int(percentage))
    print("Done B)")
    print(orderArr)
    orderArrMean = np.mean(orderArr, axis=0)
    orderArrSTD = np.std(orderArr, axis=0) 
    print(orderArrMean.shape)
    print(orderArrSTD.shape)
    print(noiseArr.shape)

    # # look for a critical point for each simulation 
    orderMeanDerivative = np.gradient(orderArrMean)
    criticalIndex = np.argmax(orderMeanDerivative * orderMeanDerivative)
    criticalNoise = noiseArr[criticalIndex]
    criticalOrder = orderArrMean[criticalIndex]
    print(f'Critical point : {criticalNoise} {criticalOrder}')

    # ### PLOT ###
    fig, ax = plt.subplots()
    ax.plot(noiseArr, orderArrMean) 
    ax.errorbar(noiseArr, orderArrMean, yerr = orderArrSTD)
    # Customizing the plot 
    ax.set(
        xlabel='Noise Amplitude',
        ylabel='Stationnary Order Factor',
        title='Stationnary order factor as a function of Noise Amplitude, many simulations'
    )
    ax.scatter(criticalNoise, criticalOrder, color='red', label=f'Crit: {criticalNoise:.2f}, {criticalOrder:.2f}')
    ax.legend()


    # ### SAVING PLOT ###
    # # ADD DESCRIPTION : Parameters, etc.
    imageDescription = 'Stationnary Order Factor as a function of Noise Amplitude \n'
    imageDescription += 'N = ' + str(N) + ', L = ' + str(L) + ', radius = ' + str(radius) + ', dt = ' + str(dt) + ', tmax = ' + str(tmax)
    imageDescription += 'rho = ' + str(N/(L*L)) + '\n'
    imageDescription += 'n_simulations = ' + str(n_simulations)

    fig.text(0, 0, imageDescription, va='top')
    fig.tight_layout()
    fig.savefig('order.png', bbox_inches='tight')
    plt.show(block = False)

    # ### SAVING IN .CSV ###   
    # resMat = [[noiseArr, orderArr[:, n]] for n in range(n_simulations)]    
    noiseMat = np.zeros((n_simulations, constNoiseNumberOfPoints))
    for sim in range(n_simulations): 
        noiseMat[sim] = noiseArr
    df_res = pd.DataFrame(
        {
            'Noise' : noiseMat.reshape(n_simulations * constNoiseNumberOfPoints), 
            'Stationnary_order_factor' : orderArr.reshape(n_simulations * constNoiseNumberOfPoints)
        }
    ) 

    df_res.to_csv('order.csv', index=False)


def heatmap_stationnary_order_factor():
    ### PARAMETERS ###
    L = 10
    radius = 1 
    dt = 1.
    tmax = 1000
    ### PARAMETERS TO STUDY ###
    constNoiseMin, constNoiseMax = 0, 3
    constNoiseNumberOfPoints = 35
    noiseArr = np.linspace(constNoiseMin, constNoiseMax, constNoiseNumberOfPoints)
    constNmin, constNmax = 10, 300
    constNnumberOfPoints = 35
    NArr = np.linspace(constNmin, constNmax, constNnumberOfPoints, dtype=int)

    ### COMPUTATION ###
    print('Computing...')
    stationnaryOrderArr = np.zeros((constNnumberOfPoints, constNoiseNumberOfPoints))
    for i, N in enumerate(NArr) : 
        for j, noiseAmplitude in enumerate(noiseArr) : 
            # # Create the model 
            model = vicsek.Vicsek(numberOfParticles=N, domainSize=(L, L), radius=radius, noiseAmplitude=noiseAmplitude, speed=0.03)
            # # Simulate the model
            simulationData = model.simulate(dt=dt, tmax=tmax)
            df = utils.simulationDataToDataframe(simulationData)
            # Compute the Binder Cumulant
            stationnaryOrderArr[i, j] = utils.stationnary_order_factor(df)
            # print(f'({i}, {j}) : {stationnaryOrderArr[i, j]} \r', end='')
            percentage = (i*constNoiseNumberOfPoints + j)/(constNnumberOfPoints*constNoiseNumberOfPoints)*100
            progress(int(percentage))
    print('Plotting...')
    ### PLOT ###
    fig, ax = plt.subplots() 
    sns.heatmap(stationnaryOrderArr, ax=ax, xticklabels=noiseArr, yticklabels=NArr/(L*L), cmap='viridis')
    # Customizing the plot
    ax.set(
        xlabel='Noise Amplitude',
        ylabel='Number of particles (proportional to density)',
        title='Stationnary Order Factor as a function of Noise Amplitude and Number of Particles in Vicsek Model',
        xticks = [],
        yticks = []
    )

    ### SAVING PLOT ###
    print('Saving plot...')
    # ADD DESCRIPTION : Parameters, etc.
    imageDescription = 'Stationnary Order Factor as a function of Noise Amplitude and Number of Particles \n'
    imageDescription += 'L = ' + str(L) + ', radius = ' + str(radius) + ', dt = ' + str(dt) + ', tmax = ' + str(tmax) + '\n'
    imageDescription += 'N' + str(constNmin) + ' to ' + str(constNmax) + ', noise ' + str(constNoiseMin) + ' to ' + str(constNoiseMax) + '\n'
    fig.text(0, 0, imageDescription, va='top')
    fig.tight_layout()
    fig.savefig('stationnary_order.png', bbox_inches='tight')
    plt.show(block = False)

    ### SAVING IN .CSV ###
    print('Saving in .csv...')
    df_res = pd.DataFrame(stationnaryOrderArr, columns=noiseArr, index=NArr/(L*L))
    df_res.to_csv('stationnary_order.csv', index=True)



if __name__ == '__main__':
    # heatmap_stationnary_order_factor() 
    print("Starting...")
    # plot_binder_cumulant()
    heatmap_stationnary_order_factor()
    # plot_stationnary_order()