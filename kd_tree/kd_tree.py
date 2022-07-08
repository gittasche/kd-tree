import numpy as np
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
    def __init__(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        super().__init__(np.copy(points))
    
    def add_point(self, point):
        """
        Add a new point to the tree.

        Parameters
        ----------
        point : array-like of shape (D,)
            New point
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
        super()._add_point(point)
    
    def query_knn(self, point, k, sort=False, metric=2):
        """
        k nearest neighbours query.

        Parameters
        ----------
        point : array-like of shape (D,)
            Get neighbours of this point
        k : int
            Number of neighbours
        sort : bool (default: False)
            Sort neighbours by distance to point
        metric : {1, 2, numpy.inf}
            metric for numpy.linalg.norm function.
            1 : l1 norm, sum(abs(x[i]))
            2 : l2 norm, sqrt(sum(x[i]**2))
            numpy.inf : l-inf norm, max(abs(x[i]))
        """
        if not isinstance(point, np.ndarray):
            point = np.array(point)
            
        if k < 1:
            raise ValueError("k must be greater than 0.")
        
        results = []
        super()._query_knn(point, k, metric, results)
        if sort:
            results = sorted(results)[::-1]
        return np.array([result[2] for result in results])
    
    def visualize(self, points):
        """
        Visualize 2d-tree.

        Parameters:
        points : array-like of shape (N, D)
            Points in the tree
        """
        if super().get_dim != 2:
            raise ValueError("Must be 2 dimensions to visualize.")
        
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        x_min, x_max = np.min([points[:, 0]]), np.max([points[:, 0]])
        y_min, y_max = np.min([points[:, 1]]), np.max([points[:, 1]])
        fig, ax = plt.subplots(figsize=(12,10))
        super()._visualize(x_min, x_max, y_min, y_max)