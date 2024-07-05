import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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

L = 20
_domainSize = (L, L, L)
# Exemple de données
simulator = vicsek.Vicsek(domainSize=_domainSize, numberOfParticles=50)
# Simulate the Vicsek model.
simulationData = simulator.simulate()
_positions, _orientations, _time = simulationData[1], simulationData[2], simulationData[0]

# Animation 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax = plt.figure().add_subplot(projection='3d')
def update_quiver(i):


    ax.cla()  # Effacer les anciennes quivers
    x, y, z = _positions[i, :, 0], _positions[i, :, 1], _positions[i, :, 2]
    u, v, w = _orientations[i, :, 0], _orientations[i, :, 1], _orientations[i, :, 2]
    ax.quiver(x, y, z, u, v, w, length=1, normalize=True)
    ax.set_xlim(0, _domainSize[0])
    ax.set_ylim(0, _domainSize[1])
    ax.set_zlim(0, _domainSize[2])
    ax.view_init(elev=35, azim=45)
    plt.title('$t$=%2.2f' % _time[i])

ani = animation.FuncAnimation(fig, update_quiver, frames=20, repeat=False)
ani.save('data/vicsek3D.mp4', writer='ffmpeg', fps=10)
plt.show()
