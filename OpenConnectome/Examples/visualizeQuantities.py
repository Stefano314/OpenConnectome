import networkx as nx
from openconnectome import Connectome
import numpy as np

# Load the numpy arrays containing the solutions.
f = np.load('f_sol.npy')
u1 = np.load('u1_sol.npy')
u2 = np.load('u2_sol.npy')
u3 = np.load('u3_sol.npy')
w = np.load('w_sol.npy')
time = np.load('time.npy')

# Read the connectome as a networkx graph ...
G = nx.read_graphml("path/to/connectome.graphml")
G = Connectome(G, quantities = [f,u1,u2,u3,w], time=time)

# ... Or pass directly the path to the graphml file
G = Connectome(G="path/to/connectome.graphml", quantities = [f,u1,u2,u3,w], time=time)


# ----- VISUALIZATION PROCEDURE -----

# Plot the average value of all the quantities over time.
G.visualizeAvgQuantities(title=f'Avg. Quantities')

# Plot f as a surface
G.draw_f(title=f'f solution')

# Evaluate the atrophy considering "all" regions, not a specific one.
atrophy = G.getAtrophy(region='all')

# Plot the value of the atrophy on all the nodes
G.drawConnectomeQuantity(atrophy, visualization='linear', links=False, title=f"atrophy")

# Plot the value of the atrophy over time of all the regions.
import matplotlib.pyplot as plt
for node in G.G.nodes():
    plt.plot(G.time, atrophy[node,:])

plt.title(f'Atrophy, all regions', fontweight='bold', fontsize=16)
plt.xlabel('time, t', fontsize=14)
plt.ylabel('atrophy, A', fontsize=14)
plt.grid(alpha=0.4)
plt.show()

# Plot each single quantity on all the nodes.
for i,val in enumerate(['u1','u2','u3','w']):
    G.drawConnectomeQuantity(G.quantities[i+1], visualization='linear', links=False, title=f"{val}")
