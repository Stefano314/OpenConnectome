# Packages
import numpy as np
import time as tm
import networkx as nx

from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid


# ////// Load Generic 1015 Connectome from Budapest Reference Connectome repository //////
# This will be used to generate the "proximity connectome", since the average one does not have the positions.
from openconnectome import Connectome
G1 = Connectome(G="path/to/connectome.graphml")
N = len(G1.G.nodes())
# --------------------------------


# ////// Define integration time, a range and proximity radius //////
t0 = 0 # Initial time
t_f = 100 # Final time
t_eval = np.linspace(0, t_f, 300) # Evaluation points
M = 14 # a variable points without a0
M_a0 = M+1 # adding a0
a_range = np.linspace(0,1,M_a0)
h_a = 1./M_a0
r_prox = 12.
# -------------------------------

# ////// Define model parameters //////
CG = 0.1
CS = 0.01
CW = 0.01
CF = 10.
mu0 = 0.01
U_bar = 0.001
d1 = 0.1 
d2 = 0.1 
dw = 1.

a = np.zeros(shape=(3,3))
b = [2,3,4]
for i,val1 in enumerate(b):
    for j,val2 in enumerate(b):
        a[i,j] = np.sqrt(val1*val2)
del b

s1 = 0.1
s2 = 0.1
s3 = 0.1
Cw = 10.
uw = 0.001
aa = 1.
s4 = 1.1
eps = 1.
source_val = 1.
# --------------------------

# ///// Initial conditions, boundaries and tools /////

# Boundary conditions
mask = np.ones(N*M_a0, bool)
mask[::M_a0] = False
C = np.zeros(N*M_a0)
C[mask] = 0. # f(a0, t) = 0, for all t

# Declare variables
f = np.zeros(N*M)
u1 = np.zeros(N)
u2 = np.zeros(N)
u3 = np.zeros(N)
w = np.zeros(N)

# Select the nodes that will contain the sources of w
source_nodes = list(G1.get_nodes_in_region('lh.entorhinal'))
source_nodes.extend(G1.get_nodes_in_region('rh.entorhinal'))

# Fix seed for replicability
np.random.seed(10)

# Initial conditions
u1_0 = np.random.normal(loc=0.5, scale=0.1, size=N)
f[::M] = 1. # f(a1,t0)
u1[:] = u1_0
u2[:] = 0.
u3[:] = 0.
w[source_nodes] = 0.

z0 = np.concatenate([f, u1, u2, u3, w])

# ///// Model Definition /////
# - L_tau: Laplacian matrix for w
# - L_abeta: Laplacian matrix for u1, u2, u3
# - C: Matrix containing ALL the values, even the initial ones. It is used in the integrals.
# !! All the parameters are defined globally (since they are way to many). It's not a good habit, but it is more clear to see !!

def alzheimerModel(t, y, L_tau, L_abeta, C):

    dydt = np.zeros(N*M+4*N) # Derivative variables
    w_source = np.zeros(N)
    w_source[source_nodes] = source_val

    C[mask] = y[:N*M]

    fv = [C[k*M_a0:(k+1)*M_a0]*(CG * np.array([trapezoid(np.maximum(a_range-a_i, 0) * C[k*M_a0:(k+1)*M_a0], a_range) for a_i in a_range]) +\
        CS*(1-a_range)*np.maximum(y[N*M+N:N*M+2*N][k]-U_bar,0) + \
        CW*(1-a_range)*y[N*M+3*N:N*M+4*N][k]) for k in range(N)]

    for k in range(N): # Slide all nodes to assign the M values of a
        dydt[k*M:(k+1)*M] = -(np.diff(fv[k]))/(h_a)


    dydt[N*M:N*M+N] = ( -d1*L_abeta @ y[N*M:N*M+N] - y[N*M:N*M+N]*(a[0,0]*y[N*M:N*M+N]+a[0,1]*y[N*M+N:N*M+2*N]+a[0,2]*y[N*M+2*N:N*M+3*N]) +\
                        CF*np.array([trapezoid((mu0+a_range)*(1-a_range)*C[k*M_a0:(k+1)*M_a0], a_range) for k in range(N)]) -\
                        s1*y[N*M:N*M+N] )

    dydt[N*M+N:N*M+2*N] = (-d2*L_abeta @ y[N*M+N:N*M+2*N] + 0.5*a[0,0]*y[N*M:N*M+N]*y[N*M:N*M+N]-y[N*M+N:N*M+2*N]*(a[1,0]*y[N*M:N*M+N]+a[1,1]*y[N*M+N:N*M+2*N]+a[1,2]*y[N*M+2*N:N*M+3*N]) - \
                        s2*y[N*M+N:N*M+2*N])
    
    dydt[N*M+2*N:N*M+3*N] = (0.5*(a[0,1]*y[N*M:N*M+N]*y[N*M+N:N*M+2*N]+a[0,2]*y[N*M:N*M+N]*y[N*M+2*N:N*M+3*N]+\
                                  a[1,1]*y[N*M+N:N*M+2*N]*y[N*M+N:N*M+2*N]+a[1,2]*y[N*M+N:N*M+2*N]*y[N*M+2*N:N*M+3*N]) -\
                            s3*y[N*M+2*N:N*M+3*N])

    dydt[N*M+3*N:N*M+4*N] = (Cw*np.maximum(y[N*M+N:N*M+2*N]-uw,0) + \
                            -dw*L_tau @ y[N*M+3*N:N*M+4*N] - \
                            (s4-1)*y[N*M+3*N:N*M+4*N] + w_source)

    return dydt
# -------------------------------------------

# ///// Laplacians and Adjacencies definitions /////

# -------------- Load 1015 Nodes Connectome from Budapest Reference Connectome ------------
# Load both number of fibers and fiber length, then create a "weight" attribute defined as the ratio
G2 = nx.read_graphml('data/budapest_connectome_3.0_5_0_median_length.graphml')
G2_ = nx.read_graphml('data/budapest_connectome_3.0_5_0_median_fibers.graphml')

# Add a new attribute to each edge
for u, v, attr in G2.edges(data=True):
    attr['weight'] = G2_[u][v]['number_of_fiber_per_fiber_length_mean'] / attr['number_of_fiber_per_fiber_length_mean']

del G2_
# ---------------------------------

# Load the budapest reference connectome to get the "connection links"
G2 = Connectome(G2)
L_conn = nx.laplacian_matrix(G2.G).toarray()
print("Connession Laplacian max value:", np.max(L_conn))
print("Connession Laplacian min value:", np.min(L_conn), '\n')
# ----------------------------------------------------

# Generate unitary proximity connectome
G1.getProximityConnectome(r_max=r_prox)
L_prox = nx.laplacian_matrix(G1.G_prox).toarray()
# ----------------------------------------------------

# Print and show some generalities (not required)
from utils.utilitytools import getNetworkComponents, drawComponentsSizes
components = getNetworkComponents(G1.G_prox)

graph_components = []
graph_components.append(components)

print("Proximity Graph connected components:", len(components))
print("Proximity Graph Links:", len(G1.G_prox.edges()))
degs = [v for _,v in nx.degree(G1.G_prox)]
print(f"Average Degree: {np.mean(degs)}, min: {np.min(degs)}, max: {np.max(degs)}") 

drawComponentsSizes(graph_components)
# -------------------------------------------------


# ===================== Integration =====================
save = True
if save:

    # ///// Integrate model /////
    import time as tm
    t1 = tm.perf_counter()
    sol = solve_ivp(alzheimerModel, t_span=[t0,t_f], t_eval=t_eval, y0=z0, method='RK23', args=(L_conn, L_prox, C))
    elapsed_time = tm.perf_counter()-t1
    print("Elapsed Time:", tm.perf_counter()-t1, 's')
    # ----------------------------

    # ///// Integrator solutions /////
    f_sol = sol.y[:N*M,:]
    u1_sol = sol.y[N*M:N*M+N,:]
    u2_sol = sol.y[N*M+N:N*M+2*N,:]
    u3_sol = sol.y[N*M+2*N:N*M+3*N,:]
    w_sol = sol.y[N*M+3*N:N*M+4*N,:]

    # Integrator time steps
    time = sol.t

    # Free space
    del sol

    # Add boundary conditions
    f_sol = np.array([np.insert(f_sol[:,t_ind], slice(0,len(C),M), C[(np.invert(mask))]) for t_ind,_ in enumerate(time)]).T

    # Save results
    sim = f'TEST_SIMULATION'
    np.save(f'time_{sim}', time)
    np.save(f'f_sol_{sim}', f_sol)
    np.save(f'u1_sol_{sim}', u1_sol)
    np.save(f'u2_sol_{sim}', u2_sol)
    np.save(f'u3_sol_{sim}', u3_sol)
    np.save(f'w_sol_{sim}', w_sol)
