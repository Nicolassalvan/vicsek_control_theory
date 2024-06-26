# -*- coding: utf-8 -*-


import numpy as np
import models.model as model


class Vicsek(model.model):
    """
    Vicsek model class

    Parameters
    ----------
    speed : float, optional
        Speed of the particles. The default is 1.
    radius : float, optional
        Interaction radius. The default is 2.
    noiseAmplitude : float, optional
        Noise amplitude. The default is 0.3.
    numberOfParticles : int, optional   
        Number of particles. The default is 150.
    domainSize : tuple, optional
        Size of the domain. The default is (100,100).

    Methods
    -------
    calculateMeanOrientations(positions, orientations)
        Calculate the mean orientations of the particles.

    generateNoise()
        Generate noise for the particles.

    simulate(initialState=(None, None), dt=None, tmax=None)
        Simulate the Vicsek model.
    """

    def __init__(self, speed=1, numberOfParticles=150, domainSize=(100,100),radius=2, noiseAmplitude=0.3):
        super().__init__(speed, numberOfParticles, domainSize)
        self.speed = speed
        self.radius = radius
        self.noiseAmplitude = noiseAmplitude
        self.numberOfParticles = numberOfParticles
        self.domainSize = np.asarray(domainSize)

    def __normalizeOrientations(self,orientations):
        return orientations/(np.sqrt(np.sum(orientations**2,axis=1))[:,np.newaxis])

    def __initializeState(self, domainSize, numberOfParticles):
        positions = domainSize*np.random.rand(numberOfParticles,len(domainSize))
        orientations = self.__normalizeOrientations(np.random.rand(numberOfParticles, len(domainSize))-0.5)
        
        return positions, orientations

    def calculateMeanOrientations(self, positions, orientations):
        rij=positions[:,np.newaxis,:]-positions
        #rij=rij[~np.eye(rij.shape[0],dtype=bool),:].reshape(rij.shape[0],rij.shape[0]-1,-1) #remove i<>i interaction
    
        rij = rij - self.domainSize*np.rint(rij/self.domainSize) #minimum image convention

        rij2 = np.sum(rij**2,axis=2)
        neighbours = (rij2 <= self.radius**2)
        summedOrientations = np.sum(neighbours[:,:,np.newaxis]*orientations[np.newaxis,:,:],axis=1)
        return self.__normalizeOrientations(summedOrientations)

    def generateNoise(self):
        return np.random.normal(scale=self.noiseAmplitude, size=(self.numberOfParticles, len(self.domainSize)))

    def simulate(self, initialState=(None, None), dt=None, tmax=None):
        """
        Launch a simulation of the Vicsek model.

        Parameters
        ----------
        initialState : tuple, optional
            Initial state of the particles. The default is (None, None).
        dt : float, optional
            Time step. The default is None.
        tmax : float, optional
            Maximum time. The default is None.

        Returns
        -------
        t : numpy.ndarray
            Time array.
        positionsHistory : numpy.ndarray
            Positions history.
        orientationsHistory : numpy.ndarray
            Orientations history.
            
        """
        positions, orientations = initialState
        
        if None in initialState:
            positions, orientations = self.__initializeState(self.domainSize, self.numberOfParticles);
            
        if dt is None:
            dt = 10**(-2)*(np.max(self.domainSize)/self.speed)
        
        if tmax is None:
            tmax = (10**3)*dt

        t=0
        nt=int(tmax/dt+1)
        
        positionsHistory = np.zeros((nt,self.numberOfParticles,len(self.domainSize)))
        orientationsHistory = np.zeros((nt,self.numberOfParticles,len(self.domainSize)))
        
        positionsHistory[0,:,:]=positions
        orientationsHistory[0,:,:]=orientations
        
        for it in range(nt):

            positions += dt*(self.speed*orientations)
            positions += -self.domainSize*np.floor(positions/self.domainSize)

            orientations = self.calculateMeanOrientations(positions, orientations)
            orientations = self.__normalizeOrientations(orientations+self.generateNoise())

            positionsHistory[it,:,:]=positions
            orientationsHistory[it,:,:]=orientations

            t+=dt

        return dt*np.arange(nt), positionsHistory, orientationsHistory
    



# TESTS 

if __name__ == "__main__":
    simulator = Vicsek(domainSize=(50,50), numberOfParticles=3)
    simulationData = simulator.simulate()


    print(simulationData[0].shape)
    print(simulationData[1])
    print(simulationData[1].shape)
    print(simulationData[2].shape)
    # dump in csv
