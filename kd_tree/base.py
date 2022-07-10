import numpy as np
import matplotlib.pyplot as plt
import heapq
import itertools

class KDTreeBase:
    """
    kd-tree implementation class.

    Attributes
    ----------
    dim : ndarray of shape(D,)
        dimension of space
    division_axis : int
        splitting hyperplane orthoghonal to division axis
    data : ndarray of shape(D,)
        point in current leaf
    left, right : KDTreeBase object
        references to child leafs
        left and right doesn't make any sense, just analogy with binary tree
    """
    def __init__(self, points=None, division_axis=0):
        self.dim = points.shape[-1]
        self.division_axis = division_axis
        
        if points.shape[0] > 1:
            idx_sort = np.argsort(points[:, division_axis])
            points = points[idx_sort]

            # equivalent to `points.shape[0] // 2`
            middle = points.shape[0] >> 1
            division_axis = (division_axis + 1) % self.dim

            self.data = points[middle]
            self.left = KDTreeBase(points[:middle], division_axis)
            self.right = KDTreeBase(points[middle + 1:], division_axis)
        elif points.shape[0] == 1:
            self.data = points[0]
            self.left = None
            self.right = None
        else:
            self.data = None
    
    def _add_point(self, point):
        """
        Add a new point to the tree.

        Parameters
        ----------
        point : array-like of shape (D,)
            New point
        """
        if self.data is not None:
            division_axis = (self.division_axis + 1) % self.dim
            dx = self.data[self.division_axis] - point[self.division_axis]
            if dx >= 0 and self.left is None:
                self.left = KDTreeBase(np.atleast_2d(point), division_axis)
            elif dx >=0:
                self.left._add_point(point)
            if dx < 0 and self.right is None:
                self.right = KDTreeBase(np.atleast_2d(point), division_axis)
            elif dx < 0:
                self.right._add_point(point)

    def _query_knn(self, point, k, metric, heap, counter=itertools.count()):
        """
        k nearest neighbor query

        Parameters
        ----------
        point : ndarray of shape(D,)
            get neighbors of this point
        k : int
            number of neighbors
        metric : {1, 2, inf}
            metric of distance for numpy.linalg.norm function.
        heap : list
            heap of neighbors candidates,
            initial value suppose to be []
        counter : itertools.count object
            counter to avoid comparison error in heappushpop()
        """
        if self.data is not None:
            dist = np.linalg.norm(self.data - point, ord=metric)
            dx = self.data[self.division_axis] - point[self.division_axis]
            # less distance must have higher priority,
            # so `-dist` in `item` instead of `dist`
            item = (-dist, next(counter), self.data)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, item)

            if dx < 0:
                if self.right is not None:
                    self.right._query_knn(point, k, metric, heap, counter)
            else:
                if self.left is not None:
                    self.left._query_knn(point, k, metric, heap, counter)
            if np.abs(dx) < -heap[0][0]:
                if dx >= 0:
                    if self.right is not None:
                        self.right._query_knn(point, k, metric, heap, counter)
                else:
                    if self.left is not None:
                        self.left._query_knn(point, k, metric, heap, counter)
    
    def _visualize(self, x_min, x_max, y_min, y_max):
        """
        Visualize 2d-tree.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : ndarray's of shape(D,)
            coordinates of vertices of plot rectangular area
        """
        if self.data is not None:
            plt.scatter(self.data[0], self.data[1], s=40, c='IndianRed', alpha=0.5)
            if self.division_axis == 0:
                plt.vlines(x=self.data[0], ymin=y_min, ymax=y_max)
                if self.left is not None:
                    self.left._visualize(x_min, self.data[0], y_min, y_max)
                if self.right is not None:
                    self.right._visualize(self.data[0], x_max, y_min, y_max)
            else:
                plt.hlines(y=self.data[1], xmin=x_min, xmax=x_max)
                if self.left is not None:
                    self.left._visualize(x_min, x_max, y_min, self.data[1])
                if self.right is not None:
                    self.right._visualize(x_min, x_max, self.data[1], y_max)

    @property
    def get_dim(self):
        return self.dim