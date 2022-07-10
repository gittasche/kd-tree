import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from .base import KDTreeBase

class KDTree(KDTreeBase):
    """
    Numpy based kd-tree.

    Parameters
    ----------
    points : array-like of shape (N, D)
        Array of points to build tree
    """
    def __init__(self, points: ArrayLike) -> None:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        self.points = np.atleast_2d(points)
        super().__init__(np.copy(self.points))
    
    def add_point(self, point: ArrayLike) -> None:
        """
        Add a new point to the tree.

        Parameters
        ----------
        point : array-like of shape (D,)
            New point
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        
        self.points = np.append(self.points, np.atleast_2d(point), axis=0)
        super()._add_point(point)
    
    def query_knn(
        self,
        point: ArrayLike,
        k: int,
        sort: bool = False,
        return_distances: bool = False,
        metric: int = 2
    ) -> np.ndarray:
        """
        k nearest neighbors query.

        Parameters
        ----------
        point : array-like of shape (D,)
            get neighbors of this point
        k : int
            number of neighbors
        sort : bool (default: False)
            sort neighbors by distance to point
        return_distances : bool (default: False)
            return distances between neighbors and point
        metric : {1, 2, numpy.inf}
            metric of distance for numpy.linalg.norm function.
            1 : l1 norm, sum(abs(x[i]))
            2 : l2 norm, sqrt(sum(x[i]**2))
            numpy.inf : l-inf norm, max(abs(x[i]))

        Returns
        -------
        if return_distances:
            np.ndarray of tuples(distance, neighbor points)
        else:
            np.ndarray of tuples(neighbor points)
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
            
        if k < 1:
            raise ValueError("k must be greater than 0.")
        
        results = []
        super()._query_knn(point, k, metric, results)
        if sort:
            results = sorted(results)[::-1]
        if return_distances:
            return np.array([(-result[0], result[2]) for result in results], dtype=object)
        else:
            return np.array([(result[2]) for result in results], dtype=object)
    
    def visualize(self):
        """
        Visualize 2d-tree.

        Returns
        -------
        ax : matplotlib.axes.Axes object
            for further use
        """
        if super().get_dim != 2:
            raise ValueError("Must be 2 dimensions to visualize.")
        
        x_min, x_max = np.min([self.points[:, 0]]), np.max([self.points[:, 0]])
        y_min, y_max = np.min([self.points[:, 1]]), np.max([self.points[:, 1]])
        _, ax = plt.subplots(figsize=(12,10))
        super()._visualize(x_min, x_max, y_min, y_max)
        return ax