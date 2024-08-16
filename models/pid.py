# -*- coding: utf-8 -*-


import numpy as np
# import model 
import models.model as model
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd 

class PID_Flock(model.model):
    """
    Flock of birds, indivuals are controlled by a PID controller. 

    
    Parameters
    ----------
    speed : float, optional
        Speed of the particles. The default is 1.
    radius : float, optional
        Interaction radius. The default is 2.
    Kp : float, optional
        Proportional gain. The default is 0.1.
    Ki : float, optional
        Integral gain. The default is 0.01.
    Kd : float, optional
        Derivative gain. The default is 0.1.
    numberOfParticles : int, optional   
        Number of particles. The default is 150.
    domainSize : tuple, optional
        Size of the domain. The default is (100,100). Must be a tuple of integers, maximum 2 dimensions.

    Methods
    -------
    calculateMeanOrientations(positions, orientations)
        Calculate the mean orientations of the particles.

    generateNoise()
        Generate noise for the particles.

    simulate(initialState=(None, None), dt=None, tmax=None)
        Simulate the Vicsek model.
    """

    def __init__(self, speed=1, numberOfParticles=150, domainSize=(100,100),radius=2, Ki=0.01, Kp=0.1, Kd=0.1, noiseAmplitude=0.3):
        super().__init__(speed, numberOfParticles, domainSize) # a bit useless, but it's better to keep it
        self.speed = speed
        self.radius = radius
        self.numberOfParticles = numberOfParticles
        self.domainSize = np.asarray(domainSize)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.noiseAmplitude = noiseAmplitude

    def dict_params(self):
        dict = {
            "speed": self.speed,
            "radius": self.radius,
            "domainSize": self.domainSize,
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
            "noiseAmplitude": self.noiseAmplitude,
            "numberOfParticles": self.numberOfParticles, 
            "density": self.numberOfParticles / (self.domainSize[0]*self.domainSize[1])
        }
        return dict
    
    # === Initialisation Methods === # 
    def initializePosition(self):
        """
        Initialize the position of the particles. 
        """
        position = self.domainSize*np.random.rand(self.numberOfParticles,len(self.domainSize))
        return position     
    
    def initializeAngles(self):
        """
        Initialize the angles of the particles. 
        """

        angles = 2 * np.pi * np.random.rand(self.numberOfParticles) - np.pi # 2D case, angle between -pi and pi
        return angles       

    
    def getNeighbours(self, positions):
        """
        Get the neighbours of the particles. 

        Parameters
        ----------
        positions : numpy.ndarray
            Positions of the particles. Dimension : (numberOfParticles, len(domainSize))
        
        Returns
        -------
        neighbours : numpy.ndarray
            Neighbours of the particles. Dimension : (numberOfParticles, numberOfParticles)

        """
        distanceMatrix = distance.cdist(positions, positions, 'euclidean')
        neighbours = distanceMatrix < self.radius
        return neighbours
    
    def getMeanAngles(self, positions, angles):
        """
        Get the mean angles of the particles. 

        Parameters
        ----------
        positions : numpy.ndarray
            Positions of the particles. Dimension : (numberOfParticles, len(domainSize))
        angles : numpy.ndarray
            Orientations of the particles. Dimension : (numberOfParticles)
        
        Returns
        -------
        meanAngles : numpy.ndarray
            Mean angles of the particles. Dimension : (numberOfParticles)

        """
        # print("Computing mean angles...")
        # print(positions.shape)
        # print(angles.shape)
        neighbours = self.getNeighbours(positions)

        # print(f"Neighbours : {neighbours.shape}")
        meanAngles = np.zeros((self.numberOfParticles))
        for i in range(self.numberOfParticles):
            meanAngles[i] = np.mean(angles[neighbours[i]])
        # print(f"Mean angles computed : {meanAngles.shape}")
        return meanAngles
    
    def computeErrorWithNoise(self, positions, angles):
        """
        Compute the error between the angles and the goal angles. 

        Parameters
        ----------
        angles : numpy.ndarray
            Orientations of the particles. Dimension : (numberOfParticles, len(domainSize)-1)
        anglesGoal : numpy.ndarray
            Goal orientations of the particles. Dimension : (numberOfParticles, len(domainSize)-1)
        
        Returns
        -------
        error : numpy.ndarray
            Error between the angles and the goal angles. Dimension : (numberOfParticles, len(domainSize)-1)

        """
        anglesGoal = self.getMeanAngles(positions, angles)
        # print("Angles goal : ", anglesGoal.shape)
        error = anglesGoal - angles

        # print("Error : ", error.shape)
        assert anglesGoal.shape == error.shape, "Error : shapes are not the same"
        assert anglesGoal.shape == angles.shape, "Error : shapes are not the same" 

        # Add noise
        noise = self.noiseAmplitude * np.random.uniform(-np.pi, np.pi, size=(self.numberOfParticles))
        error = error + noise 
        error = (error + np.pi) % (2 * np.pi) - np.pi # project in [-pi, pi]
        return error
    
    def computeError(self, positions, angles):
        """
        Compute the error between the angles and the goal angles. 

        Parameters
        ----------
        angles : numpy.ndarray
            Orientations of the particles. Dimension : (numberOfParticles, len(domainSize)-1)
        anglesGoal : numpy.ndarray
            Goal orientations of the particles. Dimension : (numberOfParticles, len(domainSize)-1)
        
        Returns
        -------
        error : numpy.ndarray
            Error between the angles and the goal angles. Dimension : (numberOfParticles, len(domainSize)-1)

        """
        anglesGoal = self.getMeanAngles(positions, angles)
        # print("Angles goal : ", anglesGoal.shape)
        error = anglesGoal - angles

        # print("Error : ", error.shape)
        assert anglesGoal.shape == error.shape, "Error : shapes are not the same"
        assert anglesGoal.shape == angles.shape, "Error : shapes are not the same" 

        return error

    def computeBankingAngle(self, errorHistory, it, dt):
        """
        Return the banking angle of the particles at time it + dt. 
        The banking angle is computed using a PID controller. 
        """
        # print("Computation of banking angle")
        # print(f"Error history : {errorHistory.shape}")
        error_plus_dt = errorHistory[it]
        if it == 0 or it == 1:
            # Cannot compute the derivative of the error : return 0
            bankingAngles = np.zeros(errorHistory[it].shape)
            # print(f"Banking angles : {bankingAngles.shape}")
            return bankingAngles
        else :
            error = errorHistory[it - 1]
            error_minus_dt = errorHistory[it - 2]
            # Numerical derivative
            integrate_term = dt * self.Ki * error_plus_dt
            derivative_term = self.Kd * (error_plus_dt - 2 * error + error_minus_dt) / dt
            proportional_term = self.Kp * (error_plus_dt - error)
            bankingAngles = integrate_term + derivative_term + proportional_term
            # print(f"Banking angles : {bankingAngles.shape}")
            return bankingAngles

    def computeAngles(self, angles, bankingAngles, dt):
        """
        Compute the new angles of the particles. 
        """
        # print(f"Computation of angles: {angles} and banking angles : {bankingAngles}")
        angles = angles + bankingAngles * dt
        # print(f"New angles : {angles.shape}")
        return angles
    
    def computePositions(self, positions, angles, dt):
        """
        Compute the new positions of the particles. 
        """
        # print(f"Computation of pos {positions.shape}")
        # print(f"Angles : {angles.shape}")

        pos_x = positions[:,0] + self.speed * np.cos(angles) * dt
        # periodic boundary conditions
        pos_x = pos_x % self.domainSize[0]
        pos_y = positions[:, 1] + self.speed * np.sin(angles) * dt
        # periodic boundary conditions
        pos_y = pos_y % self.domainSize[1]
        pos = np.array([pos_x, pos_y]).T
        # print(f"New positions : {pos.shape}")
        return pos

    def simulate(self, initialState=(None, None), dt=None, tmax=None):
        # TODO : Compute initial state
        positions, angles = initialState
        if None in initialState:
            positions, angles = self.initializePosition(), self.initializeAngles()
        
        if dt is None:
            dt = 10**(-2)*(np.max(self.domainSize)/self.speed)
        
        if tmax is None:
            tmax = (10**3)*dt

        nt=int(tmax/dt+1)

        # Create the history
        positionsHistory = np.zeros((nt,self.numberOfParticles,len(self.domainSize))) # Position history : (t, particle, (x, y, z))
        anglesHistory = np.zeros((nt,self.numberOfParticles)) # Orientation history : (t, particle, theta)
        errorHistory = np.zeros((nt, self.numberOfParticles)) # Error orientation history : (t, particle), initialized to 0
        bankingAnglesHistory = np.zeros((nt, self.numberOfParticles)) # Banking angle history : (t, particle), initialized to 0
        positionsHistory[0,:,:] = positions
        anglesHistory[0,:] = angles.reshape(self.numberOfParticles)

        # print("Shapes of history\n")
        # print(f"Positions : {positionsHistory.shape}")
        # print(f"Angles : {anglesHistory.shape}")
        # print(f"Error : {errorHistory.shape}")
        # print(f"Banking angles : {bankingAnglesHistory.shape}")
                                  

        # Simulation loop        
        for it in range(1, nt):
            # TODO : Compute mean angles
            meanAngles = self.getMeanAngles(positions, angles)
            # TODO : Compute error
            error = self.computeErrorWithNoise(positions, angles)
            error = np.reshape(error, (self.numberOfParticles))
            # print(error.shape)
            # print(errorHistory[it,:].shape)
            errorHistory[it] = error
            # TODO : Compute banking angle
            bankingAngles = self.computeBankingAngle(errorHistory, it, dt)
            # TODO : Compute new angles
            angles = self.computeAngles(angles, bankingAngles, dt)
            # TODO : Compute new positions
            positions = self.computePositions(positions, angles, dt)
            # TODO : Add to history 
            positionsHistory[it,:] = positions
            anglesHistory[it,:] = angles
        
        # Build orientation history 
        orientationHistory = np.zeros((nt,self.numberOfParticles, 2))
        orientationHistory[:,:,0] = np.cos(anglesHistory)
        orientationHistory[:,:,1] = np.sin(anglesHistory)

        ret = list([
            np.arange(nt) * dt,
            positionsHistory,
            orientationHistory,
            anglesHistory,
            errorHistory,
            bankingAnglesHistory
            ])
        return ret



    


def saturation(value, min, max):
    return np.max([np.repeat(min, len(value)), np.min([np.repeat(max, len(value)), value], axis=0)], axis=0)

# TESTS 

if __name__ == "__main__":
    simulator = PID_Flock(domainSize=(50,50), numberOfParticles=3)
    simulationData = simulator.simulate()

    print(simulationData[0].shape)
    print(simulationData[1].shape)
    print(simulationData[2].shape)
    # dump in csv
