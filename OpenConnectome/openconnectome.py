import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Connectome:
    
    def __init__(self, G, quantities = None, time = None):
        
        self.G = G
        self.nodes_info = self._relabel_nodes()

        self.quantities = quantities
        self.time = time
        self.a = self._get_a_range()
        self.proximity_df = None
        self.G_prox = None
        self.all_regions = list(set([dict(self.G.nodes(data=True))[n]['dn_name'].split("_")[0] for n in self.G.nodes()]))

    @property
    def nodes_position(self):
        positions = {k: [self.nodes_info[k][v] for v in ['dn_position_x','dn_position_y','dn_position_z']] \
                            for k in self.nodes_info.keys()}
        return np.vstack(list(positions.values()))

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


    def drawConnectome(self, highlight_nodes = None, region = None, links = False, normal_size=40, highlight_size=40):
        """
        Description
        -----------
        Connectome visualization function. Is it also possible to highlight specific nodes/regions.

        """

        link_size = 0.

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=10, azim=-90)

        for node in self.G.nodes():      
            ax.scatter(self.nodes_info[node]['dn_position_x'],
                       self.nodes_info[node]['dn_position_y'],
                       self.nodes_info[node]['dn_position_z'],
                       marker='o', s=normal_size, color='blue')
        if links:
            for n1,n2 in self.G.edges():
                ax.plot([self.nodes_info[n1]['dn_position_x'],self.nodes_info[n2]['dn_position_x']],
                        [self.nodes_info[n1]['dn_position_y'],self.nodes_info[n2]['dn_position_y']],
                        [self.nodes_info[n1]['dn_position_z'],self.nodes_info[n2]['dn_position_z']], 'b', linestyle='-', linewidth=link_size)        
            
        if highlight_nodes is not None:

            if isinstance(highlight_nodes, int):
                highlight_nodes = [highlight_nodes]
            
            plt.title(f'Connectome, nodes: {[self.nodes_info[i]["dn_name"] for i in highlight_nodes]}', fontsize=16, fontweight='bold')

            for node in highlight_nodes:
                ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                            marker='o', s=highlight_size, color='red')
           
        elif region is not None:
            highlight_nodes = self.getNodesInRegion(region=region)
            plt.title(f'Connectome, Region: {region}', fontsize=16, fontweight='bold')

            for node in highlight_nodes:
                ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                            marker='o', s=highlight_size, color='red')

        else: plt.title('Connectome', fontsize=16, fontweight='bold')

        ax.set_xlabel('x position', fontsize=12)
        ax.set_ylabel('y position', fontsize=12)
        ax.set_zlabel('z position', fontsize=12)
        plt.grid(alpha=0.4)
        plt.show()
        

    def drawRegions(self, regions = None):
        """
        Description
        -----------
        Connectome visualization function for regions. Needs to be improved.
        """
        colors = ['red', 'red', 'black','black','green','green', 'yellow', 'yellow', 'orange', 'grey']
        highlight_nodes_size = 20
        normal_size = 15

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=10, azim=-90)

        highlighted_nodes = [self.getNodesInRegion(region) for region in regions]

        flat = [n for region in highlighted_nodes for n in region]

        for node in [n for n in self.G.nodes() if n not in flat]:
            ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                        marker='o', s=normal_size, color='blue')
            
        for ind, highlighted_nodes_sub in enumerate(highlighted_nodes):
            for ind2,node in enumerate(highlighted_nodes_sub):
                if ind2==0:
                    ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                                marker='o', s=highlight_nodes_size, color=colors[ind], label=regions[ind])
                else:
                    ax.scatter(self.nodes_info[node]['dn_position_x'], self.nodes_info[node]['dn_position_y'], self.nodes_info[node]['dn_position_z'],
                                marker='o', s=highlight_nodes_size, color=colors[ind])

        plt.legend()
        ax.set_xlabel('x position', fontsize=12)
        ax.set_ylabel('y position', fontsize=12)
        ax.set_zlabel('z position', fontsize=12)
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

