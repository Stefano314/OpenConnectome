import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_all_regions_ = [
    'Right-Hippocampus', 'rh.cuneus', 'rh.paracentral', 'lh.supramarginal', 'lh.isthmuscingulate', 
    'rh.postcentral', 'lh.lateralorbitofrontal', 'rh.lateralorbitofrontal', 'lh.rostralanteriorcingulate', 
    'lh.lateraloccipital', 'lh.fusiform', 'rh.fusiform', 'rh.middletemporal', 'lh.lingual', 'lh.parsopercularis', 
    'lh.caudalmiddlefrontal', 'rh.entorhinal', 'Right-Putamen', 'lh.caudalanteriorcingulate', 'lh.superiortemporal', 
    'rh.medialorbitofrontal', 'Left-Putamen', 'lh.parsorbitalis', 'rh.isthmuscingulate', 'Left-Caudate', 
    'Right-Thalamus-Proper', 'rh.posteriorcingulate', 'lh.parstriangularis', 'lh.paracentral', 'lh.superiorparietal', 
    'lh.medialorbitofrontal', 'rh.rostralanteriorcingulate', 'rh.superiorparietal', 'rh.inferiorparietal', 
    'lh.rostralmiddlefrontal', 'rh.parahippocampal', 'lh.middletemporal', 'rh.caudalmiddlefrontal', 'lh.superiorfrontal', 
    'lh.temporalpole', 'rh.parsopercularis', 'rh.superiortemporal', 'rh.caudalanteriorcingulate', 'lh.transversetemporal', 
    'rh.transversetemporal', 'rh.precentral', 'rh.frontalpole', 'lh.entorhinal', 'lh.postcentral', 'lh.insula', 
    'rh.lateraloccipital', 'Brain-Stem', 'rh.precuneus', 'Right-Amygdala', 'Right-Caudate', 'rh.parstriangularis', 
    'Left-Hippocampus', 'rh.supramarginal', 'Left-Thalamus-Proper', 'Left-Accumbens-area', 'rh.rostralmiddlefrontal', 
    'Right-Accumbens-area', 'lh.inferiorparietal', 'lh.frontalpole', 'rh.temporalpole', 'lh.posteriorcingulate', 
    'lh.parahippocampal', 'rh.inferiortemporal', 'lh.inferiortemporal', 'Left-Amygdala', 'lh.bankssts', 'lh.precuneus', 
    'rh.insula', 'rh.lingual', 'rh.parsorbitalis', 'rh.pericalcarine', 'rh.superiorfrontal', 'lh.pericalcarine', 
    'lh.precentral', 'lh.cuneus', 'Right-Pallidum', 'Left-Pallidum', 'rh.bankssts'
    ]

_colors_ = {
# "lh.unknown" :                      (25 , 5  , 25 , 0),
"lh.bankssts" :                     (25 , 100, 40 , 255),
"lh.caudalanteriorcingulate" :      (125, 100, 160, 255),
"lh.caudalmiddlefrontal" :          (100, 25 , 0  , 255),
# "lh.corpuscallosum" :               (120, 70 , 50 , 0),
"lh.cuneus" :                       (220, 20 , 100, 255),
"lh.entorhinal" :                   (220, 20 , 10 , 255),
"lh.fusiform" :                     (180, 220, 140, 255),
"lh.inferiorparietal" :             (220, 60 , 220, 255),
"lh.inferiortemporal" :             (180, 40 , 120, 255),
"lh.isthmuscingulate" :             (140, 20 , 140, 255),
"lh.lateraloccipital" :             (20 , 30 , 140, 255),
"lh.lateralorbitofrontal" :         (35 , 75 , 50 , 255),
"lh.lingual" :                      (225, 140, 140, 255),
"lh.medialorbitofrontal" :          (200, 35 , 75 , 255),
"lh.middletemporal" :               (160, 100, 50 , 255),
"lh.parahippocampal" :              (20 , 220, 60 , 255),
"lh.paracentral" :                  (60 , 220, 60 , 255),
"lh.parsopercularis" :              (220, 180, 140, 255),
"lh.parsorbitalis" :                (20 , 100, 50 , 255),
"lh.parstriangularis" :             (220, 60 , 20 , 255),
"lh.pericalcarine" :                (120, 100, 60 , 255),
"lh.postcentral" :                  (220, 20 , 20 , 255),
"lh.posteriorcingulate" :           (220, 180, 220, 255),
"lh.precentral" :                   (60 , 20 , 220, 255),
"lh.precuneus" :                    (160, 140, 180, 255),
"lh.rostralanteriorcingulate" :     (80 , 20 , 140, 255),
"lh.rostralmiddlefrontal" :         (75 , 50 , 125, 255),
"lh.superiorfrontal" :              (20 , 220, 160, 255),
"lh.superiorparietal" :             (20 , 180, 140, 255),
"lh.superiortemporal" :             (140, 220, 220, 255),
"lh.supramarginal" :                (80 , 160, 20 , 255),
"lh.frontalpole" :                  (100, 0  , 100, 255),
"lh.temporalpole" :                 (70 , 70 , 70 , 255),
"lh.transversetemporal" :           (150, 150, 200, 255),
"lh.insula" :                       (255, 192, 32 , 255),
"Left-Pallidum" :                   (13,  48,  255, 255),
"Left-Accumbens-area" :             (255, 165, 0,   255),
"Left-Amygdala" :                   (103, 255, 255, 255),
"Left-Thalamus-Proper" :            (0,   118, 14,  255),
"Left-Hippocampus" :                (220, 216, 20,  255),
"Left-Caudate" :                    (122, 186, 220, 255),
"Left-Putamen" :                    (236, 13,  176, 255),

# "rh.unknown" :                      (25 , 5  , 25 , 0),
"rh.bankssts" :                     (25 , 100, 40 , 255),
"rh.caudalanteriorcingulate" :      (125, 100, 160, 255),
"rh.caudalmiddlefrontal" :          (100, 25 , 0  , 255),
# "rh.corpuscallosum" :               (120, 70 , 50 , 0),
"rh.cuneus" :                       (220, 20 , 100, 255),
"rh.entorhinal" :                   (220, 20 , 10 , 255),
"rh.fusiform" :                     (180, 220, 140, 255),
"rh.inferiorparietal" :             (220, 60 , 220, 255),
"rh.inferiortemporal" :             (180, 40 , 120, 255),
"rh.isthmuscingulate" :             (140, 20 , 140, 255),
"rh.lateraloccipital" :             (20 , 30 , 140, 255),
"rh.lateralorbitofrontal" :         (35 , 75 , 50 , 255),
"rh.lingual" :                      (225, 140, 140, 255),
"rh.medialorbitofrontal" :          (200, 35 , 75 , 255),
"rh.middletemporal" :               (160, 100, 50 , 255),
"rh.parahippocampal" :              (20 , 220, 60 , 255),
"rh.paracentral" :                  (60 , 220, 60 , 255),
"rh.parsopercularis" :              (220, 180, 140, 255),
"rh.parsorbitalis" :                (20 , 100, 50 , 255),
"rh.parstriangularis" :             (220, 60 , 20 , 255),
"rh.pericalcarine" :                (120, 100, 60 , 255),
"rh.postcentral" :                  (220, 20 , 20 , 255),
"rh.posteriorcingulate" :           (220, 180, 220, 255),
"rh.precentral" :                   (60 , 20 , 220, 255),
"rh.precuneus" :                    (160, 140, 180, 255),
"rh.rostralanteriorcingulate" :     (80 , 20 , 140, 255),
"rh.rostralmiddlefrontal" :         (75 , 50 , 125, 255),
"rh.superiorfrontal" :              (20 , 220, 160, 255),
"rh.superiorparietal" :             (20 , 180, 140, 255),
"rh.superiortemporal" :             (140, 220, 220, 255),
"rh.supramarginal" :                (80 , 160, 20 , 255),
"rh.frontalpole" :                  (100, 0  , 100, 255),
"rh.temporalpole" :                 (70 , 70 , 70 , 255),
"rh.transversetemporal" :           (150, 150, 200, 255),
"rh.insula" :                       (255, 192, 32 , 255),
"Right-Pallidum" :                  (13,  48,  255, 255),
"Right-Accumbens-area" :            (255, 165, 0,   255),
"Right-Amygdala" :                  (103, 255, 255, 255),
"Right-Thalamus-Proper" :           (0,   118, 14,  255),
"Right-Hippocampus" :               (220, 216, 20,  255),
"Right-Caudate" :                   (122, 186, 220, 255),
"Right-Putamen" :                   (236, 13,  176, 255),

"Brain-Stem" :                      (119, 159, 176, 255)
}

# Normalize colors
_colors_ = { key : (R/255., G/255., B/255., A/255.) for key,[R,G,B,A] in _colors_.items() }


class Connectome:
    
    def __init__(self, G, quantities = None, time = None):
        
        if isinstance(G, str):
            self.G = nx.read_graphml(G)
        else:
            self.G = G
            
        self.nodes_info = self._relabel_nodes()

        # ONLY FOR LOG
        self._elapsed_time = 0
        self._source_region = []
        self._min_max_tau = [0.,0.]
        self._min_max_amyloid = [0.,0.]
        # ----------------------

        self.quantities = quantities
        self.time = time
        self.a = self._get_a_range()
        self.proximity_df = None
        self.G_prox = None
        self.all_regions = list(set([dict(self.G.nodes(data=True))[n]['dn_name'].split("_")[0] for n in self.G.nodes()]))

        
        # ////// Define model parameters //////
        self.r_prox = 12.
        self.CG = 0.1
        self.CS = 0.01
        self.CW = 0.01
        self.CF = 10.
        self.mu0 = 0.01 # in F
        self.U_bar = 0.001 # for u2 in v
        self.d1 = 0.1 
        self.d2 = 0.1 
        self.dw = 1.

        self.alpha = np.ones(shape=(3,3))*10

        self.s1 = 0.1
        self.s2 = 0.1
        self.s3 = 0.1
        self.Cw = 10.
        self.uw = 0.001
        self.s4 = 1.1

        self.Xi = 10.
        self.Lambda = 25.

        self.source_val = 1.

    @property
    def nodesPosition(self):
        positions = {k: [self.nodes_info[k][v] for v in ['dn_position_x','dn_position_y','dn_position_z']] \
                            for k in self.nodes_info.keys()}
        return np.vstack(list(positions.values()))
    
    @property
    def proximityConnectome(self):
        if self.G_prox is None:
            self.getProximityConnectome(r_max=self.r_prox)
            return self.G_prox
        else:
            return self.G_prox

    def printAllParameters(self):
        output = f""" *--------- Model Parameters ---------*

 - Proximity radius: {self.r_prox} mm
 - Integration time: [{self.time[0]}, {self.time[-1]}] s
 - a range: [{self.a[0]}, {self.a[-1]}]

 - CG: {self.CG}
 - CS: {self.CS}
 - CW: {self.CW}
 - CF: {self.CF}
 - mu0: {self.mu0}
 - U_bar: {self.U_bar}
 - d1: {self.d1}
 - d2: {self.d2}
 - dw: {self.dw}
 - s1: {self.s1}
 - s2: {self.s2}
 - s3: {self.s3}
 - Cw: {self.Cw}
 - uw: {self.uw}
 - s4: {self.s4}
 - Xi: {self.Xi}
 - Lambda: {self.Lambda}

 - sources: {self.source_val} # If 'H', then it uses Xi*exp(-t/Lambda)
 - alpha: \n{self.alpha}

*-----------------------------------------------------*
"""
        print(output)

    def saveSimulation(self, path='', suffix='', log = None):

        # PrintSimulationInfo([sim, alpha, Cw, elapsed_time])

        np.save(f'{path}time_{suffix}', self.time)
        np.save(f'{path}f_sol_{suffix}', self.quantities[0])
        np.save(f'{path}u1_sol_{suffix}', self.quantities[1])
        np.save(f'{path}u2_sol_{suffix}', self.quantities[2])
        np.save(f'{path}u3_sol_{suffix}', self.quantities[3])
        np.save(f'{path}w_sol_{suffix}', self.quantities[4])

        if log is not None:
            with open(f'{path}log_simulation_{log}.txt', 'w') as f:
                f.write(f"####### Simulation {log} ########\n")
                f.write(f""" - GENERALITIES:
 - Brain Graph Nodes: {len(self.G.nodes())}
 - Simulation Elapsed Time: {self._elapsed_time} s
 - Source Nodes Region: {self._source_region}
 - Tau Laplacian Min/Max Values: {self._min_max_tau[0]}/{self._min_max_tau[1]}
 - Amyloid Laplacian Min/Max Values: {self._min_max_amyloid[0]}/{self._min_max_amyloid[1]}

 - NETWORK COMPONENTS:
 - Tau connectome components: {len(self.getNetworkComponents(graph='tau'))}
 - Amyloid connectome components: {len(self.getNetworkComponents(graph='amyloid'))}
CONTINUE..
 
 - Proximity radius: {self.r_prox} mm
 - Integration time: [{self.time[0]}, {self.time[-1]}] s
 - Integration time steps: {len(self.time)}
 - a range: [{self.a[0]}, {self.a[-1]}]
 - a range steps (M): {len(self.a)}

 - CG: {self.CG}
 - CS: {self.CS}
 - CW: {self.CW}
 - CF: {self.CF}
 - mu0: {self.mu0}
 - U_bar: {self.U_bar}
 - d1: {self.d1}
 - d2: {self.d2}
 - dw: {self.dw}
 - s1: {self.s1}
 - s2: {self.s2}
 - s3: {self.s3}
 - Cw: {self.Cw}
 - uw: {self.uw}
 - s4: {self.s4}
 - Xi: {self.Xi}
 - Lambda: {self.Lambda}

 - sources: {self.source_val} # If 'H', then it uses Xi*exp(-t/Lambda)
 - alpha: \n{self.alpha}""")
    

    def getNetworkComponents(self, graph):
        
        if graph == 'tau':
            G = self.G
        elif graph == 'amyloid':
            G = self.G_prox
        else:
            ValueError(f"No graph found with name '{graph}'")
            
        components = []
        components.append(list(nx.node_connected_component(G, list(G.nodes())[0])))

        for n in list(G.nodes())[1:]:
            
            found = False
            for component in components:
                if n in component:
                    found = True

            if not found:
                components.append(list(nx.node_connected_component(G,n)))

        return components
    

    def alzheimerModelSimulation(self, G_prox = None, G_conn = None, seed=10):

        from scipy.integrate import solve_ivp
        from scipy.integrate import trapezoid

        np.random.seed(seed)

        if G_prox is None:
            self.getProximityConnectome(self.r_prox)
            G_prox = self.G_prox

        if G_conn is None:
            G_conn = self.G

        # Not that beautiful to see. 
        if len(self.G.nodes()) == 83:
            # ---------------------- 83 ------------------------
            # Add a new attribute to each edge
            for _, _, attr in G_conn.edges(data=True):
                attr['weight'] = attr['number_of_fibers'] / attr['fiber_length_mean']**2

            # # Specify edge weight
            # nx.set_edge_attributes(G_conn, {(u, v): attrs["N_Lsqr_weights"] for u, v, attrs in G_conn.edges(data=True)}, "weight")
            # ----------------------------------------------------
        elif len(self.G.nodes()) == 1015:
            # -------------- 1015 ------------
            # Load both number of fibers and fiber length, then create a "weight" attribute defined as the ratio
            G_conn = nx.read_graphml('data/budapest_connectome_3.0_5_0_median_length.graphml')
            G2_ = nx.read_graphml('data/budapest_connectome_3.0_5_0_median_fibers.graphml')

            # Add a new attribute to each edge
            for u, v, attr in G_conn.edges(data=True):
                attr['weight'] = G2_[u][v]['number_of_fiber_per_fiber_length_mean'] / attr['number_of_fiber_per_fiber_length_mean']

            del G2_
            # ---------------------------------

        L_prox = None
        L_conn = None
        L_prox = nx.laplacian_matrix(G_prox).toarray()
        L_conn = nx.laplacian_matrix(G_conn).toarray()

        del G_prox, G_conn

        N = len(self.G.nodes())
        M_a0 = len(self.a) # adding a0
        M = M_a0-1 # a variable points without a0
        h_a = 1./M_a0

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

        # Initial conditions
        u1_0 = np.random.normal(loc=0.5, scale=0.1, size=N)
        f[::M] = 1. # f(a1,t0)
        u1[:] = u1_0
        u2[:] = 0.
        u3[:] = 0.

        source_nodes = list(self.getNodesInRegion('lh.entorhinal'))
        source_nodes.extend(self.getNodesInRegion('rh.entorhinal'))

        # ------- LOG --------
        self._min_max_tau[0], self._min_max_tau[1] = np.min(L_conn), np.max(L_conn)
        self._min_max_amyloid[0], self._min_max_amyloid[1] = np.min(L_prox), np.max(L_prox)
        self._source_region = 'entorhinal'
        # --------------------

        w[source_nodes] = 0.
        z0 = np.concatenate([f, u1, u2, u3, w])
        del f, u1, u2, u3, w


        def H(t, Xi, Lambda):
            return Xi*np.exp(-t/Lambda)

        def alzheimerModel(t, y, L_tau, L_abeta, C):

            dydt = np.zeros(N*M+4*N) # Derivative variables
            w_source = np.zeros(N)

            if self.source_val == "H":
                w_source[source_nodes] = H(t, Xi=self.Xi, Lambda=25)
            else:
                w_source[source_nodes] = self.source_val

            C[mask] = y[:N*M]

            fv = [C[k*M_a0:(k+1)*M_a0]*(self.CG * np.array([trapezoid(np.maximum(self.a-a_i, 0) * C[k*M_a0:(k+1)*M_a0], self.a) for a_i in self.a]) +\
                self.CS*(1-self.a)*np.maximum(y[N*M+N:N*M+2*N][k]-self.U_bar,0) + \
                self.CW*(1-self.a)*y[N*M+3*N:N*M+4*N][k]) for k in range(N)]

            for k in range(N): # Slide all nodes to assign the M values of a
                dydt[k*M:(k+1)*M] = -(np.diff(fv[k]))/(h_a)


            dydt[N*M:N*M+N] = -self.d1*L_abeta @ y[N*M:N*M+N] - y[N*M:N*M+N]*(self.alpha[0,0]*y[N*M:N*M+N]+self.alpha[0,1]*y[N*M+N:N*M+2*N]+self.alpha[0,2]*y[N*M+2*N:N*M+3*N]) +\
                              self.CF*np.array([trapezoid((self.mu0+self.a)*(1-self.a)*C[k*M_a0:(k+1)*M_a0], self.a) for k in range(N)]) -\
                              self.s1*y[N*M:N*M+N]

            dydt[N*M+N:N*M+2*N] = -self.d2*L_abeta @ y[N*M+N:N*M+2*N] + 0.5*self.alpha[0,0]*y[N*M:N*M+N]*y[N*M:N*M+N]-y[N*M+N:N*M+2*N]*\
                                   (self.alpha[1,0]*y[N*M:N*M+N]+self.alpha[1,1]*y[N*M+N:N*M+2*N]+self.alpha[1,2]*y[N*M+2*N:N*M+3*N]) - \
                                   self.s2*y[N*M+N:N*M+2*N]
            
            dydt[N*M+2*N:N*M+3*N] = 0.5*(self.alpha[0,1]*y[N*M:N*M+N]*y[N*M+N:N*M+2*N]+self.alpha[0,2]*y[N*M:N*M+N]*y[N*M+2*N:N*M+3*N]+\
                                          self.alpha[1,1]*y[N*M+N:N*M+2*N]*y[N*M+N:N*M+2*N]+self.alpha[1,2]*y[N*M+N:N*M+2*N]*y[N*M+2*N:N*M+3*N]) -\
                                    self.s3*y[N*M+2*N:N*M+3*N]


            # THIS HAS THE LINK WEIGHT INSTEAD OF ONES!
            dydt[N*M+3*N:N*M+4*N] = (self.Cw*np.maximum(y[N*M+N:N*M+2*N]-self.uw,0) + \
                                    -self.dw*L_tau @ y[N*M+3*N:N*M+4*N] - \
                                    (self.s4-1)*y[N*M+3*N:N*M+4*N] + w_source)
            
            return dydt
            
            
        import time as tm
        t1 = tm.perf_counter()
        sol = solve_ivp(alzheimerModel, t_span=[self.time[0],self.time[-1]], t_eval=self.time, y0=z0, method='RK23', args=(L_conn, L_prox, C))
        self._elapsed_time = tm.perf_counter()-t1

        print("\nSimulation compleated.\n")
        # ///// Integrator solutions /////
        self.quantities = [sol.y[:N*M,:], sol.y[N*M:N*M+N,:], sol.y[N*M+N:N*M+2*N,:], sol.y[N*M+2*N:N*M+3*N,:], sol.y[N*M+3*N:N*M+4*N,:]]

        del sol
    


    def setIntegrationTime(self, time):
        """
        Description
        -----------
        Specify the time integration range for the simulation.
        """

        self.time = time

    
    def setA_Range(self, a_range):
        """
        Description
        -----------
        Sepcify the range of a variable. It must be between 0 and 1.
        """

        self.a = a_range


    def insertQuantities(self, quantities, time, a_range):
        """
        Description
        -----------
        Give the quantities to the nodes in the connectome.

        """
        
        self.quantities = quantities
        self.time = time
        self.a = a_range


    def _get_a_range(self)->np.ndarray:
        """
        Description
        -----------
        Since we know exactly what is the structure of our data, we can determine what is the a range.

        """

        if self.quantities is not None:
            # since solution shape is (val_dim, time), and f_sol is (N*M, time)
            return np.linspace(0,1,int(self.quantities[0].shape[0]/len(self.G.nodes())))
        
        else: return None
    

    def _relabel_nodes(self) -> dict:
        """
        Description
        -----------
        Relabel the nodes as integers from 0 to N-1, so that we always have a corrispondence between
        any node in the network and its position in the adjacency matrix.

        """

        old_keys = list(dict(self.G.nodes(data=True)).keys())

        relabel_map = dict()
        for old_key in old_keys:

            relabel_map[old_key] = int(old_key)-1
        
        # Relabel nodes to integer keeping 1 to 1 corresponence (with -1, so node '22' is associated to 21)
        nx.relabel_nodes(self.G, relabel_map, copy=False)

        # Create auxiliary network so that we can reorder nodes
        H = nx.Graph()
        H.add_nodes_from(sorted(self.G.nodes(data=True)))
        H.add_edges_from(self.G.edges(data=True))

        self.G = H
        del H

        return dict(self.G.nodes(data=True))
    

    def getNodesInRegion(self, region : str) -> list:
        """
        Description
        -----------
        Get the integer ID of the nodes in the specified region.

        """
        
        nodes_locations = {k: self.nodes_info[k]['dn_name'] for k in self.nodes_info.keys()}

        # This is consistent because the nodes dictionary is ORDERED, strting from 0 to N-1!
        
        nodes_in_region = np.where(np.array([a.split("_")[0] for a in nodes_locations.values()])==region)[0]

        # At the moment, it can be both int or list. It doesn't bother numpy anyway, so we leave it like that
        # if isinstance(nodes_in_region, int):
        #     nodes_in_region = [nodes_in_region]

        return nodes_in_region
    

    def getProximityConnectome(self, r_max, weighted=False):
        """
        Description
        -----------
        Generate the proximity connectome of the current graph. This is possible only if the graph has the positions.

        """
        
        if self.proximity_df is None:
            N = len(self.G.nodes())
            nodes_position = {k: [self.nodes_info[k][v] for v in ['dn_position_x','dn_position_y','dn_position_z']] \
                            for k in self.nodes_info.keys()}

            from scipy.spatial import distance

            p = np.vstack(list(nodes_position.values()))
            distances = distance.cdist(p,p, metric='euclidean').flatten()
            nodes_id = list(nodes_position.keys())
            self.proximity_df = pd.DataFrame({'node1':np.repeat(nodes_id,N), 'node2': np.tile(nodes_id,N), 'weight':distances})
            self.proximity_df = self.proximity_df[self.proximity_df['node1']!=self.proximity_df['node2']]

        # Use copy so we can perform this procedure multiple times in the same run
        df_copy = self.proximity_df.copy()
        df_copy = df_copy[df_copy['weight']<r_max]
        df_copy = df_copy[df_copy['node1']!=df_copy['node2']]

        # Initialize self.G_prox all the times we run this function, cause if we change threshold we want it to change too.
        self.G_prox = nx.Graph()
        self.G_prox.add_nodes_from(self.G.nodes(data=True))

        if weighted:
            self.G_prox.add_weighted_edges_from(list(df_copy.itertuples(index=False, name=None)))
        else:
            df_copy['weight'] = 1
            self.G_prox.add_weighted_edges_from(list(df_copy.itertuples(index=False, name=None)))

        if not nx.is_connected(self.G_prox):
            import warnings
            warnings.warn("The created proximity network is disconnected. You might want to increase the sphere radius.", RuntimeWarning)


    def resetProximityDf(self):
        """
        Description
        -----------
        Free space required for proximity csv. It like a cache.

        """

        self.proximity_df = None


    def getAtrophy(self, region = None) -> np.ndarray:
        """
        Description
        -----------
        Evaluate the atrophy once we have the function 'f' as defined in Tesi's paper.
        We can calculate only in specific regions or globally.

        """
        
        from scipy.integrate import trapezoid

        N = len(self.G.nodes())
        M = len(self.a)

        # Rho is the normalization factor at the node k
        # rho = np.zeros(N)

        # Atrophy: this tests that the dimensions are correct, should be Nxt
        atrophy = np.zeros(shape=(int(self.quantities[0].shape[0]/M), self.quantities[0].shape[1]))

        for t in range(len(self.time)):
            
            # rho[:] = [trapezoid(self.quantities[0][k*M:(k+1)*M, t], self.a) for k in range(N)]
            atrophy[:,t] = [trapezoid(self.a * self.quantities[0][k*M:(k+1)*M, t], self.a) for k in range(N)]
            # atrophy[:,t] = atrophy[:,t]/rho # Evaluate atrophy during whole evolution for all nodes

        if region == 'all':

            for reg in self.all_regions:

                nodes_in_region = self.getNodesInRegion(region=reg)
                reg_size = len(nodes_in_region)
                for t,_ in enumerate(self.time):
                    atrophy[nodes_in_region, t] = np.sum(atrophy[nodes_in_region, t])/reg_size

        elif region is not None:
            nodes_in_region = self.getNodesInRegion(region=region)
            atrophy = np.sum(atrophy[nodes_in_region, -1])/len(nodes_in_region) # Atrophy of those nodes at last time

        else: pass

        return atrophy


    def drawConnectome(self, highlight_nodes = None, region = None, links = False,
                        normal_size=40, highlight_size=40, link_size = 0.1, title = 'Connectome'):
        """
        Description
        -----------
        Connectome visualization function. Is it also possible to highlight specific nodes/regions.

        """

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=10, azim=-90)

        for node in self.G.nodes():      
            ax.scatter(self.nodes_info[node]['dn_position_x'],
                       self.nodes_info[node]['dn_position_y'],
                       self.nodes_info[node]['dn_position_z'],
                       marker='o', s=normal_size, color=[(1.,1.,1.,1.)], edgecolors='black')
        if links:
            for n1,n2 in self.G.edges():
                ax.plot([self.nodes_info[n1]['dn_position_x'],self.nodes_info[n2]['dn_position_x']],
                        [self.nodes_info[n1]['dn_position_y'],self.nodes_info[n2]['dn_position_y']],
                        [self.nodes_info[n1]['dn_position_z'],self.nodes_info[n2]['dn_position_z']], 'black', linestyle='-', linewidth=link_size)        
            
        if highlight_nodes is not None:

            if isinstance(highlight_nodes, int):
                highlight_nodes = [highlight_nodes]
            
            plt.title(f'{title}, nodes: {[self.nodes_info[i]["dn_name"] for i in highlight_nodes]}', fontsize=16, fontweight='bold')

            for node in highlight_nodes:
                ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                            marker='o', s=highlight_size, color=[(1.,0.,0.,1.)], edgecolors='black')
           
        elif region is not None:
            highlight_nodes = self.getNodesInRegion(region=region)
            plt.title(f'{title}, Region: {region}', fontsize=16, fontweight='bold')

            for node in highlight_nodes:
                ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                            marker='o', s=highlight_size, color=[(1.,0.,0.,1.)], edgecolors='black')

        else: plt.title(f'{title}', fontsize=16, fontweight='bold')

        ax.set_xlabel('x position', fontsize=12)
        ax.set_ylabel('y position', fontsize=12)
        ax.set_zlabel('z position', fontsize=12)
        plt.grid(alpha=0.4)
        plt.show()
        

    def drawRegions(self, regions, links=True, normal_size = 50, region_size = 50, link_size = 0.1, title = "Connectome"):
        """
        Description
        -----------
        Connectome visualization function for regions. The colors are the ones given by FreeSurfer
        """

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=10, azim=-90)

        if links:
            for n1,n2 in self.G.edges():
                ax.plot([self.nodes_info[n1]['dn_position_x'],self.nodes_info[n2]['dn_position_x']],
                        [self.nodes_info[n1]['dn_position_y'],self.nodes_info[n2]['dn_position_y']],
                        [self.nodes_info[n1]['dn_position_z'],self.nodes_info[n2]['dn_position_z']], 'black', linestyle='-', linewidth=link_size)        
            

        if regions == 'all':
            regions = _all_regions_
            colors = _colors_
        
        colors = [_colors_[reg] for reg in regions] 

        highlighted_nodes = [self.getNodesInRegion(region) for region in regions]
       
        flat = [n for region in highlighted_nodes for n in region]

        for node in [n for n in self.G.nodes() if n not in flat]:
            ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                        marker='o', s=normal_size, color=[(1.,1.,1.,1.)], edgecolors='black')
            
        for ind, highlighted_nodes_sub in enumerate(highlighted_nodes):
            for ind2,node in enumerate(highlighted_nodes_sub):
                
                if ind2==0: # Label only on the first one
                    ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                                marker='o', s=region_size, c=list(colors[ind]), label=regions[ind], edgecolors='black')
                else:
                    ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                                marker='o', s=region_size, c=list(colors[ind]), edgecolors='black')

        plt.legend()
        ax.set_xlabel('x position', fontsize=12)
        ax.set_ylabel('y position', fontsize=12)
        ax.set_zlabel('z position', fontsize=12)
        plt.title(f"{title}", fontsize=16, fontweight='bold')
        plt.grid(alpha=0.4)
        plt.show()


    def drawConnectomeQuantity(self, quantity, visualization = 'spatial', links=True, title='Connectome quantity evaluation'):
        """
        Description
        -----------
        Quantities dynamical visualization on the connectome. The time evolution is the one given to the constructor and has obiously
        to match the simulation points.

        """

        from matplotlib.widgets import Slider
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Reds') 

        time = [i for i in range(quantity.shape[1])]

        if visualization == 'spatial':
            max_val = np.max(quantity)
            for t_ind in range(len(time)):
                quantity[:,t_ind] = quantity[:,t_ind]/max_val

            cmap = plt.cm.get_cmap('Reds')

            del max_val

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=10, azim=-90)

            scatter_plots = []

            if links:
                for n1,n2 in self.G.edges():

                    ax.plot([self.nodes_info[n1]['dn_position_x'],self.nodes_info[n2]['dn_position_x']],
                            [self.nodes_info[n1]['dn_position_y'],self.nodes_info[n2]['dn_position_y']],
                            [self.nodes_info[n1]['dn_position_z'],self.nodes_info[n2]['dn_position_z']], 'b', linestyle='-', linewidth=0.3)
          
            def update_color(t):

                t = int(t)
                for node, sp in enumerate(scatter_plots):
                    sp.set_color(cmap(quantity[node,t]))


            for node in self.G.nodes():
                    
                sp = ax.scatter(self.nodes_info[node]['dn_position_x'],
                                self.nodes_info[node]['dn_position_y'],
                                self.nodes_info[node]['dn_position_z'],
                                marker='o', s=100)
                
                scatter_plots.append(sp)

            # Create a slider widget to control the value of val
            t_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider = Slider(t_slider, 'Time [ind]', 0, len(time)-1, 0)
            slider.on_changed(update_color)

            ax.set_xlabel('x position', fontsize=12)
            ax.set_ylabel('y position', fontsize=12)
            ax.set_zlabel('z position', fontsize=12)

            plt.grid(alpha=0.4)
            plt.title(f'{title}, time [{self.time[0], self.time[len(time)-1]}]', fontsize=16, fontweight='bold')
            plt.show()

        elif visualization == 'linear':
            
            N = len(self.G.nodes())
            max_val = np.max(quantity)
            sp = plt.scatter([i for i in range(N)], [j for j in quantity[:,0]], vmin=0, vmax=max_val,
                                            c=[j for j in quantity[:,0]], s=[20]*N, marker='o', cmap=cmap)

            def update_color(t):

                t = int(t)
                sp.set_offsets(np.column_stack(([i for i in range(N)], [j for j in quantity[:,t]])))
                sp.set_color([cmap(quantity[node,t]/max_val) for node in range(N)])

            plt.colorbar(sp) # Draw color bar on the right

            plt.ylim([0, max_val+0.05*max_val]) # Y axis limit, to keep the initial scale, otherwise it resizes everytime

            plt.title(f'{title}, time [{self.time[0], self.time[len(time)-1]}]', fontsize=16, fontweight='bold')
            plt.xlabel('node id', fontsize=12)
            plt.ylabel('quantity', fontsize=12)
            plt.grid(alpha=0.7)
             # Create a slider widget to control the value of val
            t_slider = plt.axes([0.15, 0.02, 0.65, 0.03])
            slider = Slider(t_slider, 'Time [ind]', 0, len(time)-1, 0)
            slider.on_changed(update_color)
            plt.show()


    def draw_f(self, title = "f solution"):
        """
        Description
        -----------
        f function 3D visualization
        """
        import matplotlib.pyplot as plt

        N = len(self.G.nodes())
        M = len(self.a)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(self.a, self.time)

        f_surf = np.array([np.mean(self.quantities[0][:,t_ind].reshape(N, M), axis=0) for t_ind, _ in enumerate(self.time)])

        from matplotlib import cm
                    
        plt.title(f'{title}', fontsize=16, fontweight='bold')

        ax.plot_surface(X, Y, f_surf, cmap=cm.coolwarm)

        ax.set_xlabel('a')
        ax.set_ylabel('t')
        ax.set_zlabel('f')
        plt.show()


    def visualizeAvgQuantities(self, title = 'Average quantites evaluation'):
        """
        Description
        -----------
        Visualize the average concentration of all the quantities over time in the connectome.

        """

        plt.plot(self.time, np.mean(self.quantities[1], axis=0), '-', label=r'avg $u1$')
        plt.plot(self.time, np.mean(self.quantities[2], axis=0), '-', label=r'avg $u2$')
        plt.plot(self.time, np.mean(self.quantities[3], axis=0), '-', label=r'avg $u3$')
        plt.plot(self.time, np.mean(self.quantities[4], axis=0), '-', label=r'avg $w$')
        
        plt.xlabel('Time, a.u.', fontsize=15)
        plt.ylabel(r'$\vec{\varphi}(t)$', fontsize=15)

        plt.title(f'{title}', fontweight='bold', fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        plt.plot([],[], ' ', 
                 label = r'$(u1_{0},u2_{0},u3_{0}, w_{0})$'+f'={np.round([self.quantities[1][0,0],self.quantities[2][0,0],self.quantities[3][0,0],self.quantities[4][0,0]],4)}')
        
        plt.plot([],[], ' ', 
                 label = r'$(u1_{f},u2_{f},u3_{f}, w_{f})$'+f'={np.round([self.quantities[1][0,-1],self.quantities[2][0,-1],self.quantities[3][0,-1],self.quantities[4][0,-1]],4)}')
        plt.legend()
        plt.show()

