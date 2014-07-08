#!/usr/bin/python
# -*- coding: utf-8 -*-

import hcluster
from matplotlib import pyplot as plt

class Clusterization:

    def set_points (self, points):
        """
        :param points:
            List of points to claster.
        :return:
            Nothing
        """

        self.points = points
        self.dimensions = len(points[0])



    def draw_serial (self):
        """
        Implements Serial Clusterisation Method.

        :return:
            Nothing
        """

        x = hcluster.pdist(self.points)
        y = hcluster.linkage(x)
        hcluster.dendrogram(y)
        plt.show()


    def king (self):
        """
        Implements King Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


    def k_middle (slef):
        """
        Implements K-Middle Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


    def trout (self):
        """
        Implements "Throut" Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


    def crab (self):
        """
        Implements "Crab" Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


# Call if this is main module
# (not included)
if __name__ == '__main__':
    pts = [[41.5, 26.5], [42.3, 24.5], [42.0, 24.5], \
           [38.2, 25.5], [39.3, 26.0], [41.5, 29.5], \
           [45.5, 30.0], [39.5, 30.5], [42.0, 30.5], \
           [49.8, 30.5], [39.2, 31.0], [41.9, 27.0], \
           [45.6, 27.5], [38.1, 30.5], [44.2, 30.5]]

    cl = Clusterization()
    cl.set_points(pts)
    cl.draw_serial()