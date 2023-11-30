# OpenConnectome Introduction

OpenConnectome is a Python project aimed at manipulating **brain networks**, generally called *Connectomes*. It is designed as a tool for simulating partial differential equations on brains, with features for saving, visualizing, and tracking results in each region of the brain. OpenConnectome requires a specific data structure based on the Budapest Reference Connectome, strongly connected to the FreeSurfer nomenclature standard.

## Connectome Structure

### Budapest Reference Connectome

The ideal functioning of **OpenConnectome** is achieved using the connectome structure proposed by the Budapest Reference Connectome. These brain graphs come in different resolutions ($83$, $129$, $234$, $463$, and $1015$ nodes), providing varying levels of region precision. The Budapest Reference Connectome also allows users to download a directed brain graph, but it's still in development.

Connectomes are mathematical undirected graphs $G(V, E)$ with nodes ($N$) and edges ($M$). Budapest Reference Connectome proposes brain graphs with different numbers of nodes, ranging from $83$ to $1015$. The graphs provide a detailed web of functional connections between regions and sub-regions in the brain.

### Key points
- Average connectomes available at: [Budapest Connectome](https://pitgroup.org/connectome/).
- Single brain graphs with different resolutions available at: [Brain Graphs](https://braingraph.org/cms/download-pit-group-connectomes/).
- Connectomes can be created by analyzing fMRI images using the Connectome Mapper Toolkit: [CMTK](http://www.cmtk.org/).
- Connectomes used are undirected, and node positions are available only for single brain graphs.
- The number of nodes in the brain can range from $83$ to $1015$. However, the regions will remain $83$, with additional subdivisions in case of higher node counts.

### OpenConnectome File Format

To create a Connectome object, a brain graph provided as a `.graphml` file is required. Additionally, a brain graph with positions is needed for creating the proximity connectome. Connectome objects can be saved and loaded using the `.pkl` (pickle) format. The structure of a Connectome object is schematized in the image below.

![ConnectomeStructure](https://github.com/Stefano314/CImage/assets/79590448/980642eb-7bc8-4bba-9dc2-e02d3255aebf)


### OpenConnectome Installation

OpenConnectome is a Python package, and installation only requires Python and the following terminal command:

```bash
pip install openconnectome
```

# OpenConnectome Documentation

In this section, the complete documentation of OpenConnectome is presented. The Connectome object can be easily **instantiated** as:

```python
# Import package
from OpenConnectome.openconnectome import Connectome

# Instantiate object
G = Connectome(G_conn="path/to/connectionGraph.graphml", G_prox=("path/to/singleSubjectGraph.graphml", 5))
```
where $5$ is the radius of the proximity graph we want to create. We can also give directly the graphs if we already have them instantiated:

```python
# Import packages
import networkx as nx
from OpenConnectome.openconnectome import Connectome

# Instantiate objects
G_conn = nx.read_graphml("path/to/connectionGraph.graphml")
G_prox = nx.read_graphml("path/to/proximityGraph.graphml")
G = Connectome(G_conn=G_conn, G_prox=G_prox)
```
We can also save and load the Connectome directly, without having to specify anything else:

```python
# Import packages
from OpenConnectome.openconnectome import Connectome

# Assuming one created a connectome with everything..
# ...
# Save connectome
G.saveConnectome(path='path/to/dir/filename')

# Load connectome
G = Connectome.loadConnectome(path='path/to/dir/filename')
```
The connectome from ```loadConnectome()``` will have all the variables and results of the saved one.

## **Functions and Methods**

### *Connectome Graphs*
This is the series of network types that are available within the Connectome object. ```G_prox``` and ```G_cont``` can also be created once the Connectome object has been instantiated.

- `G_conn` : Connection graph, only functional links.
- `G_prox` : Proximity graph, only morphological links.
- `G_cont` : Contagion graph.

### *Connectome Variables*

The Connectome variables are where we can store values, and thus, affect the results of the simulations. Once we instantiate the object, as shown previously, we can use them as follows:

```python
# Import packages
from OpenConnectome.openconnectome import Connectome
G = Connectome(G_conn=G_conn, G_prox=G_prox)

# Call the variables
G.varName1 = 10.
G.varName2 = 'some value'
```
- `quantities` : List of variables resulting from integration, in order $[f, u_1, u_2, u_3, w]$.
- `time` : Integration time range.
- `a` : Integration range for $a$.
- `r_prox` : Proximity radius. This sets the maximum threshold for creating a link in the proximity graph.
- `d1` : Diffusion coefficient for $u_1$.
- `d2` : Diffusion coefficient for $u_2$.
- `dw` : Diffusion coefficient for $w$.
- `alpha` : Aggregation coefficient matrix for $u_1, u_2, u_3$.
- `s1` : Clearance for $u_1$.
- `s2` : Clearance for $u_2$.
- `s3` : Clearance for $u_3$.
- `s4` : Clearance for $w$.
- `source_region` : List containing the region with the $w$ source.
- `source_val` : Source value for $w$. If it is a number, then the source will be constant in the specified region. If it is `H`, then it is the exponential source $H(x,t)=\xi \exp{(-t/\lambda)}$.
- Other Variables : `CG`, `CS`, `CW`, `CF`, `mu0`, `U_bar`, `Ck`, `Cw`, `uw`, `Xi`, `Lambda`.

### *Connectome Log Variables*

These are the variables that appear *only* in the simulation log, and we don't need to do anything with them. Of course, they are not the only ones that will enter in the log.

- `_elapsed_time` : Time required to complete the integration.
- `_min_max_tau` : Maximum and minimum value of the $w$ laplacian.
- `_min_max_amyloid` : Maximum and minimum value of the $u$ laplacian.

### *Connectome Properties*

Properties are simply **attributes** of the object that do not require any arguments. They are simply called as:

```python
# Import packages
from OpenConnectome.openconnectome import Connectome
G = Connectome(G_conn=G_conn, G_prox=G_prox)

# Call the wanted property
nodes = G.nodesPosition
```

- `nodesPosition` : Array of all the ordered nodes' positions of the Connectome. Its shape is $N\times3$, with $N$ being the number of nodes.
- `proximityConnectome` : This returns the proximity graph with the specific `r_prox` stored.
- `getAllRegionsWithNodes` : Dictionary of all the $83$ regions with their respective nodes.

### *Connectome Methods (Calculation)*

This is clearly the most complex portion of OpenConnectome. Here, we only focus on the methods that perform **actions**; then, we will see the ones producing plots. The methods perform any numerical operation available in the package, so they will require arguments of different types. The usage depends on the method, but in general, they are called as follows:

```python
# Import packages
from OpenConnectome.openconnectome import Connectome
G = Connectome(G_conn=G_conn, G_prox=G_prox)

# Method that returns something
someValue = G.firstMethod(arg1=firstArg, arg2='stringArg')

# Methods that do not return anything
G.secondMethod(arg1=anotherArg)
G.saveSomething(path='path/to/dir')
```
- `printAllParameters()` : Print on the terminal all the parameters currently stored in the Connectome object.
- `saveSimulation(path, suffix, log)` : Once the simulation has ended, we can call this method to save all the results and logs in the required locations with the specified names.
- `saveConnectome(path)` : Save the Connectome object in the specified location.
- `loadConnectome(path)` : Load any saved Connectome object in the specified location.
- `getNetworkComponents(graph)` : Get all the components associated with the required graph, which can be `amyloid` or `tau`.
- `alzheimerModelSimulation(seed, G_cont)` : Perform the Alzheimer simulation with all the stored parameters.
- `setIntegrationTime(time)` : Set the time range as a discretized series of numbers in a range.
- `setA_Range(a_range)` : Set the $a$ range as a discretized series of numbers in a range.
- `insertQuantities(quantities, time, a_range)` : Set all the simulation quantities in case needed.
- `getNodesInRegion(region)` : Return the list of node indexes found in the specified region.
- `getProximityConnectome(r_max, G_temp, weighted)` : Generate and store the proximity connectome.
- `getAtrophy(regions, numpy_out)` : Evaluate and return the model atrophy in the specified regions. This requires a solution to the partial differential equations system.
- `getPathDistance(path)` : Evaluate the distance of a path given a list of nodes. This requires the nodes' positions.
- `getContagionGraph(steps, pathDistance, neighborhood, save, path)` : Produce the contagion graph with all the specified options. This requires a neighborhood.
- `BFS(G, u, steps)` : Breadth-first search algorithm that allows finding the nodes that are 'steps' steps distant from the current node 'u' given a generic network 'G'.
- `getNeighborhoods(G, steps, save, path)` : Get the steps-order neighborhood of all the nodes in the network as a dictionary of paths.

### *Connectome Methods (Plots)*

- `drawConnectome(highlight_nodes, region, links, normal_size, highlight_size, link_size, title)` : 3D visualization of the connectome, with the possibility to modify nodes and links.
- `drawRegions(regions, links, normal_size, region_size, link_size, title)` : 3D visualization of the connectome, with emphasis on the regions only. The colors used are the ones in the FreeSurfer standard.
- `drawConnectomeQuantity(quantity, visualization, links, title)` : Visualization of a given quantity on the nodes in the network, that can be viewed as 3D or linear. The time evolution is available through a slide.
- `drawQuantityInRegions(quantitiesDict, timeEvolution, links, normal_size, highlight_size, title)` : 3D Visualization of a quantity over time with emphasis on the specified regions. The colors used are the ones in the FreeSurfer standard.
- `drawQuantityOverTime(quantity, title, save_path, legend)` : Linear plot of a given quantity in function of time. It can be a dictionary with regions as keys so that the plot will only represent the specified regions.
- `draw_f(title, save_path)` : Plot of the surface of the function $f$.
- `drawAvgQuantities(title, save_path)` : Linear plot of the $4$ quantities averaged over all the nodes $u_1,u_2,u_3,w$.

## **Simulations, Save, and Load**

The simulation requires a model definition in the method `alzheimerModelSimulation()`, which by default is the system described by the provided set of partial differential equations:
$
\begin{align}
	&\frac{\partial f(x,a,t)}{\partial t} + \frac{\partial \big((f\,v[f])(x,a,t)\big)}{\partial a} = 0 \nonumber \\[10pt]
	&\frac{\partial u_1(x,t)}{\partial t} - d_1\nabla^2 u_1(x,t) = u_1(x,t) \sum_{j=1}^3\alpha_{1,j}\, u_j(x,t)+ F[f] - \sigma_1 u_1(x,t) \nonumber \\[10pt]
	&\frac{\partial u_2(x,t)}{\partial t} - d_2\nabla^2 u_2(x,t) = \frac{\alpha_{1,1}}{2} u_1^2(x,t) - u_2 \sum_{j=1}^3\alpha_{2,j}\, u_j(x,t) - \sigma_2 u_2(x,t) \nonumber \\[10pt]
	&\frac{\partial u_3(x,t)}{\partial t} = \frac{1}{2} \sum_{3 \leq j+k < 6}\alpha_{j,k}\, u_j(x,t) u_k(x,t) - \sigma_3u_3(x,t) \nonumber \\[10pt]
	&\frac{\partial w(x,t)}{\partial t} = C_w(u_2(x,t)-U_w)^+ + d_w\int_{\Omega}dy\, h_w(|y-x|)w(y,t)-\sigma_4 w(x,t) + H(x,t) \nonumber
\end{align}
$
which has been discretized on the network nodes.
In order to perform the simulation, we simply need to call the method as:

```python
# Import packages
import numpy as np
from OpenConnectome.openconnectome import Connectome
G = Connectome(G_conn=G_conn, G_prox=G_prox)

# Set the integration generalities
timeRange = np.linspace(0,20,150)
aRange = np.linspace(0,1,15)
G.setIntegrationTime(timeRange)
G.setA_Range(aRange)

# Change some parameters if needed
G.source_val = "H"
G.Xi = 100

# Perform the simulation
G.alzheimerModelSimulation()

# Save the results if needed
G.saveSimulation(path="path/to/dir/", suffix='someSuffix', log="path/to/dir/"+'simName'+".txt")
```
In case we also want to evaluate the **atrophy**, we simply call the method:

```python
# Load connectome in case it has been saved
from OpenConnectome.openconnectome import Connectome
G = Connectome.loadConnectome(path='path/to/dir/filename')

# Evaluate atrophy
atrophy = G.getAtrophy()

# Save atrophy if needed
import pickle as pl
f = open('path/to/dir/filename.pkl', "wb")
pl.dump(atrophy, f)
f.close()
```
If we already have some results and we simply want to load them into the Connectome, assuming that we saved the files using the method `saveSimulation()`, we do as follows:

```python
# Import packages
import numpy as np
from OpenConnectome.openconnectome import Connectome
G = Connectome(G_conn=G_conn, G_prox=G_prox)

# Load all the variables
simName = 'simulationNameUsed'
for var in ["f_sol", "u1_sol", "u2_sol", "u3_sol", "w_sol"]:
    G.quantities.append(np.load(f"path/to/dir/{var}_{simName}.npy"))

# Load time and a
G.setIntegrationTime(np.load(f"path/to/dir/time_{simName}.npy"))
G.setA_range(np.load(f"path/to/dir/a_{simName}.npy"))
```
Once everything has been created, we can show the results using the plot functions:

```python
# Assuming all the quantities are loaded
simName = 'simulationNameUsed'

# u1, u2, u3, w plots
G.drawAvgQuantities(title=f"Avg. quantities, {simName}")

# f surface
G.draw_f(title=f"f Avg. Solution, {simName}")

# Evaluate the atrophy in a few regions and plot it over time
atrophy = G.getAtrophy(regions=['Right-Hippocampus', 'Left-Hippocampus', 'Brain-Stem'])
G.drawQuantityOverTime(atrophy, title=f"Atrophy, {simName}")
```

## **Citing**
If **OpenConectome** was utilized in your research, we kindly request that you consider citing us in the following manner:
```
@software{OpenConnectome,
  author       = {S. Bianchi, G. Landi, M. C. Tesi, C. Testa},
  title        = {OpenConnectome},
  year         = 2023,
  url          = {https://github.com/Stefano314/OpenConnectome}
}
```

## **Bibliography**
- M. Bertscha, B. Franchi, V. Meschini, M. C. Tesi, A. Tosin. *"A sensitivity analysis of a mathematical model for the synergistic interplay
of amyloid beta and tau on the dynamics of Alzheimer’s disease"*, Brain Multiphysics Volume 2, 2021, 100020.