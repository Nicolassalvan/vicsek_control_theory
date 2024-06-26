import numpy as np 
import matplotlib.pyplot as plt

import pandas as pd
import vicsek

from utils import *

import Animator2D
import MatplotlibAnimator



if __name__ == "__main__":
    simulator = vicsek.Vicsek(domainSize=(50,50), numberOfParticles=1000)
    simulationData = simulator.simulate()


    # print(simulationData[0].shape)
    # print(simulationData[1])
    # print(simulationData[1].shape)
    # print(simulationData[2].shape)

    save_path = 'vicsek_simulation.csv'
    simulationDataToCSV(simulationData, save_path)

    """
    Initialize the Matplotanimator and feed the simulation data and domain size.
    """
    animator = MatplotlibAnimator.MatplotlibAnimator(simulationData, (50,50))

    """
    Prepare the animator for a 2D representation.
    """
    preparedAnimator = animator.prepare(Animator2D.Animator2D())

    """
    Execute and save the animation as an mp4.

    Optionally: (this requires ffmpeg)
    preparedAnimator.saveAnimation('vicsek2.mp4')
    """

    """
    After saving the animation we can also show it directly.
    """
    preparedAnimator.saveAnimation('vicsek2.mp4')

