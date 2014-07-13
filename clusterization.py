#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

        self.set_points(points, labels)
        self.enable_xml_output(xml_enable, xml_file_name)
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
        self.dimension = None
        self.dimension = len(points[0])
        self.labels = labels
        if not labels:
            self.labels = ['{}'.format(i) for i in range(len(points))]
        self.__count_dist_matrix()


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


    def __count_dist_matrix(self):
        """
        Count distances matrix.

        :return:
            Nothing.
        """

        p_len = len(self.points)
        self.dist_matrix = [[0.0 for i in range(p_len)] for j in range(p_len)]
        for i in range(p_len):
            for j in range(p_len):
                cur_dist = 0.0
                for k in range(self.dimension):
                    cur_dist += (self.points[i][k] - self.points[j][k]) ** 2
                self.dist_matrix[i][j] = cur_dist


    def __write_points_into_xml_sheet (self, work_sheet, last_empty_row):
        """
        :param work_sheet:
            Worksheet instance from xlwt library.
        :param last_empty_row:
            Row on which points will be begun to write.
        :return:
            New empty row.
        """

        p_len = len(self.points)
        new_last_empty_row = last_empty_row
        style_table = xlwt.XFStyle()
        style_table.borders.bottom = 1
        style_table.borders.top = 1
        style_table.borders.left = 1
        style_table.borders.right = 1

        for i in range(1, self.dimension + 1):
            work_sheet.write(new_last_empty_row + i, 0, u'x{}'.format(i), style_table)
        for i in range(p_len):
            work_sheet.write(new_last_empty_row, i + 1, self.labels[i], style_table)
        new_last_empty_row += 1
        for i in range(p_len):
            for j in range(self.dimension):
                work_sheet.write(new_last_empty_row + j, i + 1, self.points[i][j], style_table)
        new_last_empty_row += self.dimension + 1

        return new_last_empty_row


    def __write_dist_table_into_xml_sheet (self, work_sheet, last_empty_row):
        """
        :param work_sheet:
            Worksheet instance from xlwt library.
        :param last_empty_row:
            Row on which points will be begun to write.
        :return:
            New empty row.
        """

        p_len = len(self.points)
        new_last_empty_row = last_empty_row
        style_table = xlwt.XFStyle()
        style_table.borders.bottom = 1
        style_table.borders.top = 1
        style_table.borders.left = 1
        style_table.borders.right = 1

        work_sheet.write(new_last_empty_row , 0, u'Матрица расстояний:')
        new_last_empty_row += 1
        for i in range(p_len):
            work_sheet.write(new_last_empty_row, i + 1, self.labels[i], style_table)
        new_last_empty_row += 1
        for j in range(p_len):
            work_sheet.write(new_last_empty_row + j, 0, self.labels[j], style_table)
        for i in range(p_len):
            for j in range(p_len):
                work_sheet.write(new_last_empty_row + i, j + 1, self.dist_matrix[i][j], style_table)
        new_last_empty_row += p_len + 1

        return new_last_empty_row


    def __pretty_view (self):
        for cluster in self.clusters:
            cluster.sort()


    def draw(self):
        """
        Draws clusters with clustered_labels.

        :return:
            Nothing
        """

        # TODO: pretty view with crab

        # 2D Drawing
        if self.dimension == 2:
            min_x = min([self.points[i][0] for i in range(len(self.points))])
            max_x = max([self.points[i][0] for i in range(len(self.points))])
            min_y = min([self.points[i][1] for i in range(len(self.points))])
            max_y = max([self.points[i][1] for i in range(len(self.points))])
            wh = 0  # width and simul. heigth
            if (max_x - min_x) > (max_y - min_y):
                wh = max_x - min_x
            else: wh = max_y - min_y
            plt.xlim(min_x - wh/12.0, min_x + wh + wh/12.0)
            plt.ylim(min_y - wh/12.0, min_y + wh + wh/12.0)

            for cluster, label in zip(self.clusters, self.clustered_labels):
                x = [cluster[i][0] for i in range(len(cluster))]
                y = [cluster[i][1] for i in range(len(cluster))]
                l = [label[i] for i in range(len(label))]
                plt.plot(x, y, 'ro-')

                for X, Y, L in zip(x, y, l):
                    plt.annotate(
                      L, xy = (X, Y), xytext = (-10, 10),
                      textcoords = 'offset points', ha = 'center', va = 'bottom',
                      bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        # 3D Drawing
        elif self.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            n = 100
            for cluster, label, color in zip(self.clusters, self.clustered_labels, ['r', 'g', 'b']):
                x = [cluster[i][0] for i in range(len(cluster))]
                y = [cluster[i][1] for i in range(len(cluster))]
                z = [cluster[i][2] for i in range(len(cluster))]
                ax.scatter(x, y, z, c = color)
                # todo: line connections, labels

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()


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
        last_empty_row = 0
        if self.xml_enable:
            work_book = xlwt.Workbook()
            work_sheet = work_book.add_sheet('Results')
            style_table = xlwt.XFStyle()
            style_table.borders.bottom = 1
            style_table.borders.top = 1
            style_table.borders.left = 1
            style_table.borders.right = 1

        # Write limit into xml.
        if self.xml_enable:
            work_sheet.write(last_empty_row, 0, u'Порог равен {}'.format(limit))
            last_empty_row += 2

        # Write points into xml.
        if self.xml_enable:
            last_empty_row = self.__write_points_into_xml_sheet(work_sheet, last_empty_row)

        # Write the distance matrix into xml.
        if self.xml_enable:
            last_empty_row = self.__write_dist_table_into_xml_sheet(work_sheet, last_empty_row)

        # While non clustered points indexes list is
        # not empty, we are clustering em.
        while non_clustered_pts_i:

            if len(non_clustered_pts_i) == 1:
                i = non_clustered_pts_i[0]
                clusters_i.append([])
                cur_cluster = len(clusters_i) - 1
                clusters_i[cur_cluster].append(i)
                non_clustered_pts_i.remove(i)

                # Tell it..
                if self.xml_enable:
                    work_sheet.write(last_empty_row, 0, u'Точка №{} осталась одна.'\
                     u'Кластер №{} будет состоять лишь из этой точки.'.format(i, cur_cluster))
                    last_empty_row += 1
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

            if self.xml_enable:
                last_empty_row += 1
                work_sheet.write(last_empty_row, 0, u'Создадим кластер №{}. ' \
                 u'Добавим туда точку №{}'.format(cur_cluster, i))
                last_empty_row += 1

            non_clustered_pts_i_copy = non_clustered_pts_i[:]

            for j in non_clustered_pts_i:
                computations_str = ''
                middle_dist = 0
                for point_i in clusters_i[cur_cluster]:
                    computations_str += ' + {}'.format(self.dist_matrix[point_i][j])
                    middle_dist += self.dist_matrix[point_i][j]
                middle_dist /= len(clusters_i[cur_cluster])
                computations_str += ') / {} = {}'.format(len(clusters_i[cur_cluster]), middle_dist)
                computations_str = '(' + computations_str[3:]


                if self.xml_enable:
                    work_sheet.write(last_empty_row, 0, u'Рассмотрим точку №{}. ' \
                      u'Среднее расстояние до точек кластера №{} равно {}'
                      .format(j, cur_cluster, computations_str))
                    last_empty_row += 1

                # To include or not to inclule: that is the question.
                if middle_dist > limit:

                    if self.xml_enable:
                        work_sheet.write(last_empty_row, 0, u'Следовательно, не будем ' \
                         u'добавлять данную точку в этот кластер.')
                        last_empty_row += 1

                    continue

                else:
                    clusters_i[cur_cluster].append(j)
                    non_clustered_pts_i_copy.remove(j)

                    if self.xml_enable:
                        work_sheet.write(last_empty_row, 0, u'Следовательно, добавим '\
                         u'данную точку в этот кластер.')
                        last_empty_row += 1

            non_clustered_pts_i = non_clustered_pts_i_copy

            if self.xml_enable:
                last_empty_row += 1
                work_sheet.write(last_empty_row, 0, u'Состав кластера №{}: {}'
                 .format(cur_cluster, clusters_i[cur_cluster]))
                last_empty_row += 2

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

        self.__pretty_view()

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
        # todo: CRABBBB!!

        p_len = len(self.pts)




# Call if this is main module
# (not included)
if __name__ == '__main__':
    pts = [[41.5, 26.5], [42.3, 24.5], [42.0, 24.5],
           [38.2, 25.5], [39.3, 26.0], [41.5, 29.5],
           [45.5, 30.0], [39.5, 30.5], [42.0, 30.5],
           [49.8, 30.5], [39.2, 31.0], [41.9, 27.0],
           [45.6, 27.5], [38.1, 30.5], [44.2, 30.5]]

    pts3d = [[41.5, 26.5, 1.1], [42.3, 24.5, 1.1], [42.0, 24.5, 1.1],
           [38.2, 25.5, 1.1], [39.3, 26.0, 1.1], [41.5, 29.5, 1.1],
           [45.5, 30.0, 1.1], [39.5, 30.5, 1.1], [42.0, 30.5, 1.1],
           [49.8, 30.5, 1.1], [39.2, 31.0, 1.1], [41.9, 27.0, 1.1],
           [45.6, 27.5, 1.1], [38.1, 30.5, 1.1], [44.2, 30.5, 1.1]]

    # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #           '11', '12', '13', '14', '15']

    cl = Clusterization(pts)
    cl.enable_xml_output(True, "output.xls")
    cluster, clustered_labels = cl.king(24.0)
    cl.draw()



    # for row in cluster:
    #     print row
    # if clustered_labels:
    #     for row in clustered_labels:
    #         print row
