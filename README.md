# **OpenConnectome**

[![PyPI](https://img.shields.io/pypi/v/openconnectome.svg)](https://pypi.org/project/openconnectome/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Stefano314/openconnectome/blob/main/LICENSE)

## **Description**

### *OpenConnectome* is a Python package for working with connectome data. It provides tools for loading, processing, analyzing, and visualizing connectomes in a user-friendly manner. The data structure used is the one proposed by the [Budapest Reference Connectome](https://pitgroup.org/connectome/) and has to be strictly followed. For now, it is not designed to work with any other network except that **brain graphs**.
</b></b>

### The package provides the implementation of a mathematical model describing the development of the *Alzheimer's Disease*, which is its the main application.
</b></b>

## **Images and Results**
</b></b>
<img src="https://github.com/Stefano314/OpenConnectome/assets/79590448/4f159b0e-86cc-40d4-84c5-d13400d96e7d" width="300" height="300" /> <img src="https://github.com/Stefano314/OpenConnectome/assets/79590448/0dbc5614-1b14-472b-8a35-0db563cc023f" width="300" height="300" />
<img src="https://github.com/Stefano314/OpenConnectome/assets/79590448/dd921474-477d-49fa-a632-2edad015f34e" width="600" height="300" />
<img src="https://github.com/Stefano314/OpenConnectome/assets/79590448/488cfb05-f196-4897-89b0-79f3be291703" width="600" height="300" />


## **Installation**

(TODO) You can install OpenConnectome using `pip`:

```shell
pip install openconnectome
```

## **Generalities and Usage**
OpenConnectome is designed as a tool for simulating Alzheimer models, offering users a highly accessible way to visualize the resulting data.
It is divided into two main classes: ```openconnectome``` and ```grid3D```: the first handles the visualization and analysis of the brain graph, whereas the latter incorporates an adaptive space-grid division technique that aids in estimating specific connectome parameters, such as the *proximity radius*.


The package is structured in a way that is possible to quickly run articulated commands, such as the whole simulation of an implemented model, but it also provides users with the flexibility to work with custom scripts and aesthetics.

### **OpenConnectome Methods**
- **```setIntegrationTime``` :** Specify the time steps in which we want the simulation to be evaluated. It should be a one dimensional ```numpy.ndarray```; it will be stored into Connectome ```time``` attribute.

- **```setA_Range``` :** Specify the *a* variable range. It should be a one dimensional ```numpy.ndarray```; it will be stored into Connectome ```a``` attribute.

- **```getNodesInRegion``` :** Returns the list of nodes *IDs* in the requested region.

- **```getProximityConnectome``` :** Generates the *proximity connectome* using euclidean metrics. It creates links between nodes that are distant at most *r_max*. The proximity connectome is stored into Connectome ```G_prox``` attribute. It can also be accessed directly using the class property ```proximityConnectome```.

- **```getAtrophy``` :** Calculate the atrophy across all the nodes and regions in the brain graph according to the relation described in the PAPER ...

- **```drawConnectome``` :** Simple 3D visualization of the given brain graph.

- **```drawRegions``` :** 3D brain graph visualization highlighting the specified regions. The colors used for the regions is the **FreeSurfer** colormap.

- **```drawConnectomeQuantity``` :** Plot the specified quantity on the nodes in the brain graph over the time range stored into Connectome ```time``` attribute. It provides a *slider* to go through all the time steps.

- **```draw_f```** : Visualization of the surface *f*. More details in the PAPER.. 

- **```visualizeAvgQuantities``` :** Plot the time evolution of the biological quantities **u1**, **u2**, **u3** and **w**, averaged over all the nodes.

- **```alzheimerModelSimulation``` :** Performs the complete analysis using the new analytical model describing the development of the Alzheimer's Diseases and saves the result into the instantiated **Connectome** object.


### **Grid3D Methods**
- **```FindPointInGrid``` :** Give one or more points to be found in the grid and it returns ***cellIndexes***, ***gridCoordinates***, ***cellsWithElements***. The first one is the ordered list of the cells indexes (i,j,k) containing the points, the second is the list of (x,y,z) coordinates of the cells and the last one is a dictionary with *cells indexes* as **keys** and *point IDs* contained in that cell as **values**.

- **```ShowPointsInGrid``` :** 3D visualization of the grid and the points in it. The cells containing the points are displayed in *red* with increasing alpha as the number of points inside increases, and in *green* the empty cells.


## **Citing**
If **OpenConectome** was utilized in your research, we kindly request that you consider citing us in the following manner:
```
@software{OpenConnectome,
  author       = {S. Bianchi, G. Landi, M. C. Tesi, C. Testa},
  title        = {OpenConnectome, a Brain Graph Analysis tool.},
  year         = 2023,
  url          = {https://github.com/Stefano314/OpenConnectome}
}
```
## **Bibliography**
- M. Bertscha, B. Franchi, V. Meschini, M. C. Tesi, A. Tosin. *"A sensitivity analysis of a mathematical model for the synergistic interplay
of amyloid beta and tau on the dynamics of Alzheimer’s disease"*, Brain Multiphysics Volume 2, 2021, 100020.