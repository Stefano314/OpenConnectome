import numpy as np
import matplotlib.pyplot as plt

class Grid3D:

    def __init__(self, pointsPosition, boxSubdivision):

        self._minCorner, self._maxCorner = self._GetCorners(pointsPosition)
        self.cellSize = [0,0,0]
        self.xPoints = None
        self.yPoints = None
        self.zPoints = None

        # boxSubdivision can be both a float or a list of integers
        self.boxSubdivision = boxSubdivision

        # If boxSubdivision is a number, the cell sizes are all the same
        if not isinstance(boxSubdivision, list):
            self.cellSize = [boxSubdivision, boxSubdivision, boxSubdivision] 

        self._CreateGrid()

    @property
    def grid(self):
        return np.array([[xi, yi, zi] for xi in self.xPoints for yi in self.yPoints for zi in self.zPoints])
    
    @property
    def estimatedRadius(self):
        return np.sqrt(self.cellSize[0]**2 + self.cellSize[1]**2 + self.cellSize[2]**2)/2.
    
    def _GetCorners(self, pointsPosition):
        maxX = np.max(pointsPosition[:,0])
        maxY = np.max(pointsPosition[:,1])
        maxZ = np.max(pointsPosition[:,2])

        minX = np.min(pointsPosition[:,0])
        minY = np.min(pointsPosition[:,1])
        minZ = np.min(pointsPosition[:,2])

        smallestCorner = np.array([minX, minY, minZ])
        largestCorner = np.array([maxX, maxY, maxZ])

        return smallestCorner, largestCorner
    
    def _CreateGrid(self):

        # Use evenly spaced jumps or fixed number of intervals to create the grid based on boxSubdivision value.
        if not isinstance(self.boxSubdivision, list):
            self.xPoints = np.arange(self._minCorner[0], self._maxCorner[0]+self.boxSubdivision, self.cellSize[0])
            self.yPoints = np.arange(self._minCorner[1], self._maxCorner[1]+self.boxSubdivision, self.cellSize[1])
            self.zPoints = np.arange(self._minCorner[2], self._maxCorner[2]+self.boxSubdivision, self.cellSize[2])
        else:
            self.xPoints = np.linspace(self._minCorner[0], self._maxCorner[0], self.boxSubdivision[0])
            self.yPoints = np.linspace(self._minCorner[1], self._maxCorner[1], self.boxSubdivision[1])
            self.zPoints = np.linspace(self._minCorner[2], self._maxCorner[2], self.boxSubdivision[2])
            self.cellSize[0] = self.xPoints[1]-self.xPoints[0]
            self.cellSize[1] = self.yPoints[1]-self.yPoints[0]
            self.cellSize[2] = self.zPoints[1]-self.zPoints[0]

        self._maxCorner = np.array([self.xPoints[-1], self.yPoints[-1], self.zPoints[-1]])

        # Print some generalities
        print(f"Grid Shape: ({len(self.xPoints)}, {len(self.yPoints)}, {len(self.zPoints)})")
        print(f"Grid Size: ({self._maxCorner[0]-self._minCorner[0]}, {self._maxCorner[1]-self._minCorner[1]}, {self._maxCorner[2]-self._minCorner[2]})")
        print(f"Cell Size: ({self.cellSize[0]}, {self.cellSize[1]}, {self.cellSize[2]})")
        print(f"Grid Proximity Radius: {np.sqrt(self.cellSize[0]**2 + self.cellSize[1]**2 + self.cellSize[2]**2)/2.}")

    
    def _cuboid_data(self, pos, size=(1,1,1)):
        """
        Description
        -----------
        Build the cuboid mesh at position 'pos' and resize it with 'size'.

        """
        
        X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
             [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
        
        X = np.array(X).astype(float)

        for i in range(3):
            X[:,:,i] *= size[i]

        X += np.array(pos)

        return X


    def _plotCubeAt(self, positions, bias, sizes=None, colors=None, **kwargs):
        """
        Description
        -----------
        Draw a cube as a mesh at 'position', with custom size and color.

        """
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from mpl_toolkits.mplot3d import Axes3D

        if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
        if not isinstance(sizes,(list,np.ndarray)): sizes=[(bias[0],bias[1],bias[2])]*len(positions)

        g = []
        bias = np.array(bias) # Assure it is an array so that we can perform division and difference with another array

        for p,s,c in zip(positions-bias/2., sizes, colors):
            g.append( self._cuboid_data(p, size=s) )

        return Poly3DCollection(np.concatenate(g), facecolors=np.repeat(colors, 6, axis=0), **kwargs)


    def FindPointInGrid(self, searchPoints):
        """
        Description
        -----------
        Search in which cell are the specified points. It finds them as a ratio of their normalized position over the max value in the grid.

        """

        if searchPoints.shape == (3,):
            searchPoints = np.array([searchPoints])

        searchPoints_copy = searchPoints.copy()
        searchPoints_copy = np.array(searchPoints_copy)-self._minCorner
        maxGridPoint = self._maxCorner - self._minCorner
        pointPositionRatio = np.abs(searchPoints_copy/maxGridPoint)

        # Use a small bias to avoid rounding errors
        if np.any(searchPoints_copy<-0.0001) or np.any(pointPositionRatio > 1.0001):
            print("Point outside the grid")
            return None, None

        else:
            gridCoordinates = np.zeros(shape=(len(searchPoints_copy),3))
            cellIndexes = []
            cellsWithElements = dict()

            for ind,_ in enumerate(searchPoints_copy):

                cellIndex = [int(np.rint(pointPositionRatio[ind][0] * (len(self.xPoints) - 1))),
                             int(np.rint(pointPositionRatio[ind][1] * (len(self.yPoints) - 1))),
                             int(np.rint(pointPositionRatio[ind][2] * (len(self.zPoints) - 1)))]

                cellIndexes.append(cellIndex)

                if tuple(cellIndex) in cellsWithElements:
                    cellsWithElements[tuple(cellIndex)].append(ind)
                else:
                    cellsWithElements[tuple(cellIndex)] = [ind]

                
                gridCoordinates[ind] = [self.xPoints[cellIndex[0]], self.yPoints[cellIndex[1]], self.zPoints[cellIndex[2]]]

        return cellIndexes, gridCoordinates, cellsWithElements


    def ShowPointsInGrid(self, searchPoints, remaining = False, margins=1, overlaps = False):
        """
        Description
        -----------
        Show the points in the grid highlighting the cells in which they are.

        """

        gridIndexes, gridCoordinates, _ = self.FindPointInGrid(searchPoints)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(searchPoints[:,0], searchPoints[:,1], searchPoints[:,2], marker='o', color='black', s=100)

        if overlaps:

            _, indx, counts = np.unique(gridIndexes, axis=0, return_counts=True, return_index=True)

            for count in np.unique(counts):

                newCords = gridCoordinates[indx[np.where(counts==count)[0]]]
                nCubes = len(newCords)
                colors = np.array([count/np.max(counts),0,0]*nCubes).reshape(nCubes,3)
                sizes=[(self.cellSize[0],self.cellSize[1],self.cellSize[2])]*nCubes
                pc = self._plotCubeAt(newCords, bias=self.cellSize, colors=colors, alpha=0.5*(count/np.max(counts)), 
                                      edgecolors='k', linewidths=0.05, sizes = sizes)
                ax.add_collection3d(pc)
        else:

            nCubes = len(searchPoints)
            colors = np.array([1,0,0]*nCubes).reshape(nCubes,3)
            pc = self._plotCubeAt(gridCoordinates, bias=self.cellSize, colors=colors, alpha=0.05, edgecolors='k', linewidths=0.05)
            ax.add_collection3d(pc)

        if remaining:
            
            fullGrid = self.grid
            remainingGrid = []
            for row in fullGrid:
                if np.any(np.all(np.isclose(gridCoordinates, row), axis=1)):
                    pass
                else:
                    remainingGrid.append(row)

            remainingGrid = np.array(remainingGrid)

        if remaining:
            nCubes = len(remainingGrid)
            colors = np.array([0,1,0]*nCubes).reshape(nCubes,3)
            pc = self._plotCubeAt(remainingGrid, bias=self.cellSize, colors=colors, alpha=0.02, edgecolors='k', linewidths=0.05)
            ax.add_collection3d(pc)

        ax.set_xlim([self._minCorner[0]-margins,self._maxCorner[0]+margins])
        ax.set_ylim([self._minCorner[1]-margins,self._maxCorner[1]+margins])
        ax.set_zlim([self._minCorner[2]-margins,self._maxCorner[2]+margins])

        ax.set_title("Grid Point Search", fontweight='bold', fontsize=16)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        plt.show()