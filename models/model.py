import pandas as pd
import numpy as np

# Base class for the models that we will implement

class model():
    def __init__(self, speed=1, numberOfParticles=150, domainSize=(100,100)):
        self.speed = speed
        self.numberOfParticles = numberOfParticles
        self.domainSize = np.asarray(domainSize)
        # parameters of the model
        pass

    def simulate(self, initialState=(None, None), dt=None, tmax=None):
        pass
    #   return dt*np.arange(nt), positionsHistory, orientationsHistory # others can be added
    # I advice you to stock the return in a res variable, and then separate it into t, positionsHistory, orientationsHistory 
    # and others 


