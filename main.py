import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd

# Import models 
import models.vicsek as vicsek


from utils import *


import animation.Animator2D as Animator2D
import animation.MatplotlibAnimator as MatplotlibAnimator



if __name__ == "__main__":
    # Initialize the Vicsek model.
    simulator = vicsek.Vicsek(domainSize=(50,50), numberOfParticles=50)
    # Simulate the Vicsek model.
    simulationData = simulator.simulate()

    # Extract the simulation data.
    time_sim, positions_sim, orientations_sim = simulationData[0], simulationData[1], simulationData[2]
    
    # Save the simulation data to a CSV file.
    save_path = 'data/vicsek_simulation.csv'
    simulationDataToCSV(simulationData, save_path)

    # Initialize the Matplotanimator and feed the simulation data and domain size.
    animator = MatplotlibAnimator.MatplotlibAnimator(simulationData, (50,50))
    # Prepare the animator for a 2D representation.
    preparedAnimator = animator.prepare(Animator2D.Animator2D())
    # Execute and save the animation as an mp4. This requires ffmpeg to be installed.
    preparedAnimator.saveAnimation('data/vicsek2.mp4')

