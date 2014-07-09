#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import math

class Clusterization:
    """
    Clusterization class provides clusterization procedures for
    2D or 3D points array.
    """

    def __init__(self, points = None, labels = None, xml_enable = False, xml_file_name = None):
        """

        :param points:
            List of points to claster.
        :param labels:
            Labels of points. You can use it for drawing.
        :param xml_enable:
            Enable or disable xml output.
        :param xml_file_name:
            Xml output file name.
        :return:
            Nothing
        """

        self.points = points
        self.dimension = len(points[0])
        self.labels = labels
        self.xml_enable = xml_enable
        self.xml_file_name = xml_file_name
        self.dist_matrix = None
        self.count_dist_matrix()


    def set_points(self, points, labels):
        """

        :param labels:
            Labels of points. You can use it for drawing.
        :param points:
            List of points to claster.
        :return:
            Nothing
        """

        self.points = points
        self.dimension = len(points[0])
        self.labels = labels
        self.count_dist_matrix()


    def enable_xml_output(self, enable = False, file_name = None):
        """
        Enable or disable outputing calculations results into .xml table.

        :param enable:
            True if enable, else False.
        :param file_name:
            File name to write.
        :return:
            Nothing.
        """

        self.xml_enable = enable
        self.xml_file_name = file_name


    def count_dist_matrix(self):
        """
        Count and return distance matrix.

        :param points:
            List of points.
        :return:
            Distance matrix.
        """
        p_len = len(self.points)
        self.dist_matrix = [[0.0 for i in range(p_len)] for j in range(p_len)]
        for i in range(p_len):
            for j in range(p_len):
                cur_dist = 0.0
                for k in range(self.dimension):
                    cur_dist += math.sqrt((self.points[i][k] - self.points[j][k]) ** 2)
                self.dist_matrix[i][j] = cur_dist



    def king(self):
        """
        Implements King Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """



        pass


    def k_middle(slef):
        """
        Implements K-Middle Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


    def trout(self):
        """
        Implements "Throut" Clusterisation Method.

        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        pass


    def crab(self):
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
    pts = [[41.5, 26.5], [42.3, 24.5], [42.0, 24.5],
           [38.2, 25.5], [39.3, 26.0], [41.5, 29.5],
           [45.5, 30.0], [39.5, 30.5], [42.0, 30.5],
           [49.8, 30.5], [39.2, 31.0], [41.9, 27.0],
           [45.6, 27.5], [38.1, 30.5], [44.2, 30.5]]

    cl = Clusterization(pts)
    for row in cl.dist_matrix:
        print row
