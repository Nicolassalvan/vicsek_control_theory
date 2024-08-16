import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 
import numpy as np 
import matplotlib.pyplot as plt

# Import models 
import models.vicsek as vicsek
import models.pid as pid
import utils as utils


import utils as utils
import visualisation as visualisation




# def animate_simulation2D_colored(df, labels,  L, plotBool = False, annotateBool = False, fps = 30):
#     interval = int(1000/fps)
#     df_x = df.filter(regex='^x')
#     df_y = df.filter(regex='^y')
#     df_theta_x = df.filter(regex='^theta_x')
#     df_theta_y = df.filter(regex='^theta_y')
    
#     time = df['t'].to_numpy()

#     x, y = df_x.to_numpy(), df_y.to_numpy()
#     theta_x, theta_y = df_theta_x.to_numpy(), df_theta_y.to_numpy()

#     labels = labels.to_numpy()  
#         # Animation 
#     fig, ax = plt.subplots()


#     def update_quiver_with_colors(i):
#         ax.cla()  # Effacer les anciennes quivers
#         arrow_x, arrow_y = x[i], y[i]
#         arrow_u, arrow_v = theta_x[i], theta_y[i]
#         color = labels[i] / labels.max()
#         noise_mask = labels[i] == -1
#         color[noise_mask] = -1
#         color = color / 2 + 0.5

#         ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v, color = plt.cm.viridis(color))
#         if annotateBool:
#             for j, txt in enumerate(labels[i]):
#                 ax.annotate(txt, (x[i][j], y[i][j]))

#         ax.set_xlim(0, L)
#         ax.set_ylim(0, L)
#         plt.title('$t$=%2.2f' % time[i])

#     ani = FuncAnimation(fig, update_quiver_with_colors, frames = len(x), repeat=False , interval = interval)
#     if plotBool:
#         plt.show()
#     return ani

# def animate_simulation2D(df, L, plotBool = False, fps = 30):
#     interval = int(1000/fps)

#     df_x = df.filter(regex='^x')
#     df_y = df.filter(regex='^y')
#     df_theta_x = df.filter(regex='^theta_x')
#     df_theta_y = df.filter(regex='^theta_y')
    
#     time = df['t'].to_numpy()

#     x, y = df_x.to_numpy(), df_y.to_numpy()
#     theta_x, theta_y = df_theta_x.to_numpy(), df_theta_y.to_numpy()


#     print("Data created. Now plotting...")
#     # Animation 
#     fig, ax = plt.subplots()


#     def update_quiver(i):
#         ax.cla()  # Effacer les anciennes quivers
#         arrow_x, arrow_y = x[i], y[i]
#         arrow_u, arrow_v = theta_x[i], theta_y[i]

#         ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v)
#         ax.set_xlim(0, L)
#         ax.set_ylim(0, L)
#         plt.title('$t$=%2.2f' % time[i])

#     ani = FuncAnimation(fig, update_quiver, frames = len(x), repeat=False, interval = interval)
#     if plotBool:
#         plt.show()
#     return ani

if __name__ == "__main__":
    # Creating simulation data 
    print("Creating the data...")
    L = 50

    # Exemple de données
    # simulator = vicsek.Vicsek(domainSize=_domainSize, numberOfParticles=50)
    simulator = pid.PID_Flock(domainSize=(L, L), radius=5, numberOfParticles=50, Kp = 0.9812, Kd = 0, Ki = 0.1929)
    # Simulate the Vicsek model.
    simulationData = simulator.simulate()
    df = utils.simulationDataToDataframe(simulationData)
    df_labels = utils.clusters_over_time(df, k_coef = 2, L = L)
    
    animation = visualisation.animate_simulation2D(df, L)
    animation.save('test.mp4', writer='ffmpeg', fps=30)

    animation2 = visualisation.animate_simulation2D_colored(df, df_labels, L)
    animation2.save('test_colored.mp4', writer='ffmpeg', fps=30)