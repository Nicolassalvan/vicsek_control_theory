import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd


# Import models 
import models.vicsek as vicsek


from utils import *


if __name__ == "__main__":
    # Initialize the Vicsek model.
    simulator = vicsek.Vicsek(domainSize=(50,50,50), numberOfParticles=50)
    # Simulate the Vicsek model.
    simulationData = simulator.simulate()
    print(np.shape(simulationData[0]), np.shape(simulationData[1]), np.shape(simulationData[2]))
    # Animate the simulation.
