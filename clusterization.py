#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import math
import xlwt

class Clusterization:
    """
    Clusterization class provides clusterization procedures for
    2D or 3D points array.
    """

    def __init__(self, points = None, labels = None,
                 xml_enable = False, xml_file_name = None):
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
        self.dimension = None
        if points:
            self.dimension = len(points[0])
        self.labels = labels
        self.xml_enable = xml_enable
        self.xml_file_name = xml_file_name
        self.dist_matrix = None
        self.count_dist_matrix()
        self.clusters = None
        self.clustered_labels = None


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

        :return:
            Distance matrix.
        """

        p_len = len(self.points)
        self.dist_matrix = [[0.0 for i in range(p_len)] for j in range(p_len)]
        for i in range(p_len):
            for j in range(p_len):
                cur_dist = 0.0
                for k in range(self.dimension):
                    cur_dist += (self.points[i][k] - self.points[j][k]) ** 2
                self.dist_matrix[i][j] = cur_dist

        return self.dist_matrix


    def draw(self, clusters = None, clustered_lables = None):
        """
        Draws clusters with clustered_labels. If clusters clusters
        or clustered_labels is None, draws self.clusters with
        self.clustered_labels. In this case you can do it, if you
        call clasterization method before.

        :param clusters:
            List of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        :param clustered_lables:
            List of lists with the same dimension as clusters, but
            the elements are strings, not points.
        :return:
            Nothing
        """

        # Give to clusters pretty view.


    def king(self, limit):
        """
        Implements King Clusterisation Method.

        :param limit:
            Maximal middle distance between point and cluster or
            maximal distance between clusters.
        :return:
            List of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        p_len = len(self.points)
        non_clustered_pts_i =[i for i in range(p_len)]
        clusters_i = []

        # Preparations for xml writing
        work_book = None
        work_sheet = None
        style_table = None
        style_text = None
        last_empty_row = 0
        xls_labels = None
        if self.xml_enable:
            work_book = xlwt.Workbook()
            work_sheet = work_book.add_sheet('Results')
            style_table = xlwt.XFStyle()
            style_table.borders.bottom = 1
            style_table.borders.top = 1
            style_table.borders.left = 1
            style_table.borders.right = 1
            style_text = xlwt.XFStyle()
            if self.labels:
                xls_labels = self.labels
            else:
                xls_labels = [str(i) for i in range(p_len)]

        # Write points into xml.
        if self.xml_enable:
            for i in range(1, self.dimension + 1):
                work_sheet.write(last_empty_row + i, 0, u'x{}'.format(i), style_table)
            for i in range(p_len):
                work_sheet.write(last_empty_row, i + 1, xls_labels[i], style_table)
            last_empty_row += 1
            for i in range(p_len):
                for j in range(self.dimension):
                    work_sheet.write(last_empty_row + j, i + 1, self.points[i][j], style_table)
            last_empty_row += self.dimension + 1

        # Write the distance matrix into xml.
        if self.xml_enable:
            work_sheet.write(last_empty_row , 0, u'Матрица расстояний:', style_text)
            last_empty_row += 1
            for i in range(p_len):
                work_sheet.write(last_empty_row, i + 1, xls_labels[i], style_table)
            last_empty_row += 1
            for j in range(p_len):
                work_sheet.write(last_empty_row + j, 0, xls_labels[j], style_table)
            for i in range(p_len):
                for j in range(p_len):
                    work_sheet.write(last_empty_row + i, j + 1, self.dist_matrix[i][j], style_table)
            last_empty_row += p_len + 1

        while non_clustered_pts_i:

            if len(non_clustered_pts_i) == 1:
                i = non_clustered_pts_i[0]
                clusters_i.append([])
                cur_cluster = len(clusters_i) - 1
                clusters_i[cur_cluster].append(i)
                non_clustered_pts_i.remove(i)

                # Tell it..
                if self.xml_enable:
                    work_sheet.write(last_empty_row, 0, u'Осталась одна точка. \
                     Составим из нее кластер.')
                break

            # Find nearest points.
            min_i = None
            min_dist = self.dist_matrix[0][1]
            for i in range(p_len - 1):
                for j in range(i + 1, p_len):
                    if min_dist > self.dist_matrix[i][j]:
                        min_dist = self.dist_matrix[i][j]
                        min_i = i

            # Put one of the nearest points into the cluster.
            i = non_clustered_pts_i[min_i]
            clusters_i.append([])
            cur_cluster = len(clusters_i) - 1
            clusters_i[cur_cluster].append(i)
            non_clustered_pts_i.remove(i)

            non_clustered_pts_i_copy = non_clustered_pts_i[:]

            for j in non_clustered_pts_i:
                middle_dist = 0
                for point_i in clusters_i[cur_cluster]:
                    middle_dist += self.dist_matrix[point_i][j]
                middle_dist /= len(clusters_i[cur_cluster])

                # To include or not to inclule: that is the question.
                if middle_dist > limit:
                    continue
                else:
                    clusters_i[cur_cluster].append(j)
                    non_clustered_pts_i_copy.remove(j)

            non_clustered_pts_i = non_clustered_pts_i_copy

        # Transform indexes into points.
        clusters = [[0 for i in range(len(clusters_i[j]))] for j in range(len(clusters_i))]
        for i in range(len(clusters_i)):
            for j in range(len(clusters_i[i])):
                clusters[i][j] = self.points[clusters_i[i][j]][:]

        # Transform indexes into labels.
        clustered_labels = None
        if self.labels:
            clustered_labels = [[0 for i in range(len(clusters_i[j]))] for j in range(len(clusters_i))]
            for i in range(len(clusters_i)):
                for j in range(len(clusters_i[i])):
                    clustered_labels[i][j] = self.labels[clusters_i[i][j]][:]

        self.clusters = clusters
        self.clustered_labels = clustered_labels

        if self.xml_enable:
            work_book.save(self.xml_file_name)

        return clusters, clustered_labels


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

    # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #           '11', '12', '13', '14', '15']

    cl = Clusterization(pts)
    cl.enable_xml_output(True, "output.xls")
    cluster, clustered_labels = cl.king(24.0)



    for row in cluster:
        print row
    if clustered_labels:
        for row in clustered_labels:
            print row
