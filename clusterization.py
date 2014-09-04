#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import math
import xlwt


class Clusterization:
    """
    Clusterization class provides clusterization procedures for
    2D or 3D points array.
    """

    def __init__(self, points = None, labels = None,
                 xls_enable = False, xls_file_name = None):
        """

        :param points:
            List of points to claster.
        :param labels:
            Labels of points. You can use it for drawing.
        :param xls_enable:
            Enable or disable xls output.
        :param xls_file_name:
            xls output file name.
        :return:
            Nothing
        """

        if points:
            self.set_points(points, labels)
        if xls_enable:
            self.xls_enable_output(xls_enable, xls_file_name)
        else:
            self.xls_disable_output()
        self.clusters = None


    def set_points(self, points, labels=None):
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


    def xls_enable_output(self, file_name):
        """
        Enable outputing calculations results into .xls table.

        :param file_name:
            File name to write.
        """

        self.xls_enable = True
        self.xls_file_name = file_name


    def xls_disable_output(self):
        """
        Disable outputing calculations results into .xls table.
        """

        self.xls_enable = False
        self.xls_file_name = None


    def count_dist_matrix(self, points):
        """
        Count and return distance matrix.

        :param points:
            List of points.
        :return:
            Distance matrix.
        """

        p_len = len(points)
        dimension = len(points[0])
        dist_matrix = [[0.0 for i in range(p_len)] for j in range(p_len)]
        for i in range(p_len):
            for j in range(p_len):
                cur_dist = 0.0
                for k in range(dimension):
                    cur_dist += (points[i][k] - points[j][k]) ** 2
                dist_matrix[i][j] = cur_dist

        return dist_matrix


    def __count_dist_matrix(self):
        """
        Count distances matrix.

        :return:
            Nothing.
        """

        self.dist_matrix = self.count_dist_matrix(self.points)


    def __write_points_into_xls_sheet (self, work_sheet, last_empty_row):
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

        work_sheet.write(new_last_empty_row, 0, u'Точки: ')
        new_last_empty_row += 1

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


    def __write_dist_table_into_xls_sheet (self, work_sheet, last_empty_row):
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


    def __draw_crabbed_cluster(self, cluster, ax=None):
        """
        Draw cluster using "Crab" algorithm.

        :param ax:
        :param cluster:
        :return:
        """

        dist_matrix = self.count_dist_matrix(cluster)
        p_len = len(cluster)
        draw_pts_i = []
        not_draw_pts_i = [i for i in range(p_len)]

        tmp = not_draw_pts_i[0]
        not_draw_pts_i.remove(tmp)
        draw_pts_i.append(tmp)
        if self.dimension == 2:
            plt.plot([cluster[tmp][0]], [cluster[tmp][1]], 'ro-')
        elif self.dimension == 3:
            ax.scatter([cluster[tmp][0]], [cluster[tmp][1]], [cluster[tmp][2]], 'ro-')

        while not_draw_pts_i:

            # Find point with minimum distance to drew points.
            min_i = draw_pts_i[0]
            min_j = not_draw_pts_i[0]
            min_distance = dist_matrix[min_i][min_j]
            for i in draw_pts_i:
                for j in not_draw_pts_i:
                    if dist_matrix[i][j] < min_distance:
                        min_distance = dist_matrix[i][j]
                        min_i = i
                        min_j = j

            # Draw points with min distance.
            x = [cluster[i][0] for i in (min_i, min_j)]
            y = [cluster[i][1] for i in (min_i, min_j)]
            if self.dimension == 2:
                plt.plot(x, y, 'ro-')
            elif self.dimension == 3:
                z = [cluster[i][2] for i in (min_i, min_j)]
                ax.plot_wireframe(x, y, z)
                ax.scatter(x, y, z)

            draw_pts_i.append(min_j)
            not_draw_pts_i.remove(min_j)


    def draw(self):
        """
        Draws clusters.

        :return:
            Nothing
        """

        # 2D Drawing
        if self.dimension == 2:
            # Make good view of aixes
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

            # Draw clusters
            for cluster in self.clusters:
                self.__draw_crabbed_cluster(cluster)

            # Draw labels
            x = [pt[0] for pt in self.points]
            y = [pt[1] for pt in self.points]
            l = [l for l in self.labels]
            for X, Y, L in zip(x, y, l):
                plt.annotate(
                  L, xy = (X, Y), xytext = (0, 10),
                  textcoords = 'offset points', ha = 'center', va = 'bottom',
                  bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        # 3D Drawing
        elif self.dimension == 3:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for cluster in self.clusters:
                self.__draw_crabbed_cluster(cluster, ax)

            # Draw labels.
            labels = []
            for point, l in zip(self.points, self.labels):

                x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
                labels.append(plt.annotate(
                    l,
                    xy = (x2, y2), xytext = (-10, 10),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5)))

            # Recount labels position.
            def update_position(e):
                for point, label in zip(self.points, labels):
                    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
                    label.xy = x2, y2
                    label.update_positions(fig.canvas.renderer)
                    fig.canvas.draw()
            fig.canvas.mpl_connect('button_release_event', update_position)

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

        # Preparations for xls writing
        work_book = None
        work_sheet = None
        style_table = None
        last_empty_row = 0
        if self.xls_enable:
            work_book = xlwt.Workbook()
            work_sheet = work_book.add_sheet('Results')
            style_table = xlwt.XFStyle()
            style_table.borders.bottom = 1
            style_table.borders.top = 1
            style_table.borders.left = 1
            style_table.borders.right = 1

        # Write limit into xls.
        if self.xls_enable:
            work_sheet.write(last_empty_row, 0, u'Порог равен {}'.format(limit))
            last_empty_row += 2

        # Write points into xls.
        if self.xls_enable:
            last_empty_row = self.__write_points_into_xls_sheet(work_sheet, last_empty_row)

        # Write the distance matrix into xls.
        if self.xls_enable:
            last_empty_row = self.__write_dist_table_into_xls_sheet(work_sheet, last_empty_row)

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
                if self.xls_enable:
                    work_sheet.write(last_empty_row, 0, u'Точка №{} осталась одна.'\
                     u'Кластер №{} будет состоять лишь из этой точки.'.format(i, cur_cluster))
                    last_empty_row += 1
                break

            # Find nearest points.
            min_i = non_clustered_pts_i[0]
            min_j = non_clustered_pts_i[1]
            min_dist = self.dist_matrix[min_i][min_j]
            for i in non_clustered_pts_i:
                for j in non_clustered_pts_i:
                    if i != j:
                        if min_dist > self.dist_matrix[i][j]:
                            min_dist = self.dist_matrix[i][j]
                            min_i = i
                            min_j = j

            # Put one of the nearest points into the cluster.
            #min_i = non_clustered_pts_i[min_i]
            #min_j = non_clustered_pts_i[min_j]
            clusters_i.append([])
            cur_cluster = len(clusters_i) - 1
            clusters_i[cur_cluster].append(min_i)
            non_clustered_pts_i.remove(min_i)

            if self.xls_enable:
                work_sheet.write(last_empty_row + 1, 0, u'Создадим кластер №{}'.format(cur_cluster))
                work_sheet.write(last_empty_row + 3, 0, u'Две ближайшие точки: {} и {} ({})'.format(min_i, min_j, min_dist))
                work_sheet.write(last_empty_row + 4, 0, u'Добавим точку {} в кластер.'.format(min_i))
                work_sheet.write(last_empty_row + 5, 0, u'Точка', style_table)
                work_sheet.write_merge(last_empty_row + 5, last_empty_row+ 5, 1, 6, u'Среднее расстояние до кластера',style_table)
                last_empty_row += 6


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

                if self.xls_enable:
                    work_sheet.write(last_empty_row, 0, u'{}'.format(j), style_table)
                    work_sheet.write_merge(last_empty_row, last_empty_row, 1, 6, u'{}'.format(computations_str),style_table)
                    last_empty_row += 1

                # To include or not to inclule: that is the question.
                if middle_dist > limit:
                    continue

                else:
                    clusters_i[cur_cluster].append(j)
                    non_clustered_pts_i_copy.remove(j)

            non_clustered_pts_i = non_clustered_pts_i_copy

            if self.xls_enable:
                last_empty_row += 1
                work_sheet.write(last_empty_row, 0, u'Состав кластера №{}: {}'
                 .format(cur_cluster, clusters_i[cur_cluster]))
                last_empty_row += 2

        # Transform indexes into points.
        clusters = [[0 for i in range(len(clusters_i[j]))] for j in range(len(clusters_i))]
        for i in range(len(clusters_i)):
            for j in range(len(clusters_i[i])):
                clusters[i][j] = self.points[clusters_i[i][j]][:]

        self.clusters = clusters

        if self.xls_enable:
            work_book.save(self.xls_file_name)

        return clusters


    def k_middle(self, ptsi):
        """
        Implements K-Middle Clusterisation Method.

        :param ptsi:
            Indexes of points which will be the init. centres.
        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        cn = len(ptsi)  # Centres number.
        centres = [self.points[i][:] for i in ptsi]
        prev_clusters = [[0] for i in ptsi]
        clusters = [[] for i in ptsi]

        # Preparations for xls writing
        wb = None
        ws = None
        s_table = None
        row = 0
        step = 1
        if self.xls_enable:
            wb = xlwt.Workbook()
            ws = wb.add_sheet('Results')
            s_table = xlwt.XFStyle()
            s_table.borders.bottom = 1
            s_table.borders.top = 1
            s_table.borders.left = 1
            s_table.borders.right = 1

        # Write initial centres into xls.
        if self.xls_enable:
            ws.write(row, 0, u'Начальные центры кластеров:')
            row += 1
            for j in range(self.dimension):
                ws.write(row + j + 1, 0, u'x{}'.format(j + 1), s_table)
            for i in range(cn):
                ws.write(row, i + 1, u'{}'.format(i), s_table)
                for j in range(self.dimension):
                    ws.write(row + j + 1, i + 1, u'{}'.format(self.points[ptsi[i]][j]), s_table)
            row += (self.dimension + 2)

        # Write points into xls.
        if self.xls_enable:
            row = self.__write_points_into_xls_sheet(ws, row)

        # Write the distance matrix into xls.
        if self.xls_enable:
            row = self.__write_dist_table_into_xls_sheet(ws, row)

        equal = False
        while not equal:

            prev_clusters = clusters
            clusters = [[] for i in ptsi]

            if self.xls_enable:
                row += 1
                ws.write(row, 0, u'Итерация №{}'.format(step))
                step += 1
                row += 2

           # Write of distaces between points and clusters centres.
            if self.xls_enable:
                ws.write(row, 0, u'Матрица расстояний между точками и центрами кластеров:')
                row += 1
                for i in range(len(self.points)):
                    ws.write(row, i + 1, u'{}'.format(i), s_table)
                for j in range(cn):
                    ws.write(row + j + 1, 0, u'{}'.format(j), s_table)

            # Distribute points on clusters.
            for i in range(len(self.points)):

                # Initial minimum distance.
                minj = 0
                min_dist = 0.0
                for k in range(self.dimension):
                    min_dist += (centres[minj][k] - self.points[i][k]) ** 2
                min_dist = math.sqrt(min_dist)

                if self.xls_enable:
                    ws.write(row + 1, i + 1, u'{}'.format(min_dist), s_table)

                # Find min distance cluster.
                for j in range(1, cn):
                    tmp_dist = 0.0
                    for k in range(self.dimension):
                        tmp_dist += (centres[j][k] - self.points[i][k]) ** 2
                    tmp_dist = math.sqrt(tmp_dist)
                    if tmp_dist < min_dist:
                        min_dist = tmp_dist
                        minj = j

                    if self.xls_enable:
                        ws.write(row + j + 1, i + 1, u'{}'.format(tmp_dist), s_table)

                # Append point to min distance cluster.
                clusters[minj].append(i)

            # Write clusters view.
            if self.xls_enable:
                row += 5
                ws.write(row, 0, u'Таким образом, получились следующие кластеры:')
                row += 1
                for j in range(cn):
                    ws.write(row + j, 0, u'{}'.format(j))
                for cluster in clusters:
                    ws.write(row, 1, u'{}'.format(cluster))
                    row += 1

            # Recalc the centres.
            for i in range(cn):
                for k in range(self.dimension):
                    centres[i][k] = 0.0
                    for j in range(len(clusters[i])):
                        centres[i][k] += self.points[clusters[i][j]][k]
                    centres[i][k] /= len(clusters[i])

            # Write recalculated centres into xls.
            if self.xls_enable:
                row += 1
                ws.write(row, 0, u'Новые центры кластеров: ')
                row += 1
                for j in range(self.dimension):
                    ws.write(row + j + 1, 0, u'x{}'.format(j + 1), s_table)
                for i in range(cn):
                    ws.write(row, i + 1, u'{}'.format(i), s_table)
                    for j in range(self.dimension):
                        ws.write(row + j + 1, i + 1, u'{}'.format(centres[i][j]), s_table)
                row += (self.dimension + 2)

            # Compare prev and current clusters.
            for i in range(cn):
                clusters[i].sort()
                prev_clusters[i].sort()
            clusters.sort()
            prev_clusters.sort()
            equal = True
            for pcl, cl in zip(prev_clusters, clusters):
                if pcl != cl:
                    equal = False
                    break

            # Write comparation result.
            if self.xls_enable:
                if not equal:
                    ws.write(row, 0, u'Предыдущий кластер не равен текущему, продолжаем вычисления')
                else:
                    ws.write(row, 0, u'Предыдущий кластер равен текущему, вычисления закончены')
                row += 2

        # Transform indexes into points.
        clusters_i = clusters
        clusters = [[0 for i in range(len(clusters_i[j]))] for j in range(len(clusters_i))]
        for i in range(len(clusters_i)):
            for j in range(len(clusters_i[i])):
                clusters[i][j] = self.points[clusters_i[i][j]][:]

        # Save xls.
        if self.xls_enable:
            wb.save(self.xls_file_name)

        self.clusters = clusters
        return clusters


    def trout(self, radius):
        """
        Implements "Throut" Clusterisation Method.

        :param radius:
            Radius of circle.
        :return:
            list of clusters. Cluster is the list of points.
            Point is the list of nubmers ([1.3, 3.5] or [1, 2, 3]).
        """

        plen = len(self.points)
        cptsi = []  # Clustered points indexes.
        ncptsi = [i for i in range(plen)]
        clusters = []

        # Preparations for xls writing
        wb = None
        ws = None
        s_table = None
        row = 0
        steps = [0, 1]
        if self.xls_enable:
            wb = xlwt.Workbook()
            ws = wb.add_sheet('Results')
            s_table = xlwt.XFStyle()
            s_table.borders.bottom = 1
            s_table.borders.top = 1
            s_table.borders.left = 1
            s_table.borders.right = 1

        # Write initial centres into xls.
        if self.xls_enable:
            ws.write(row, 0, u'Радиус равен {}'.format(radius))
            row += 2

        # Write points into xls.
        if self.xls_enable:
            row = self.__write_points_into_xls_sheet(ws, row)

        # Write the distance matrix into xls.
        if self.xls_enable:
            row = self.__write_dist_table_into_xls_sheet(ws, row)

        while ncptsi:
            cluster = []
            centre = self.points[ncptsi[0]][:]
            prev_centre = self.points[ncptsi[0]][:]
            prev_centre[0] += 1     # To be different to centre.

            if self.xls_enable:
                row += 1
                ws.write(row, 0, u'Образуем кластер №{}'.format(steps[0]))
                steps[0] += 1
                steps[1] = 1
                row += 1
                ws.write(row, 0, u'Поместим центр сферы в точку {}'.format(ncptsi[0]))
                ws.write(row + 1, 0, u'Центр сферы: {}'.format(self.points[ncptsi[0]]))
                row += 3


            while centre != prev_centre:
                prev_centre = centre[:]
                cluster = []

                # Begining of table of distances between points and sphere centre.
                if self.xls_enable:
                    ws.write(row, 0, u'Итерация №{}'.format(steps[1]))
                    steps[1] += 1
                    ws.write(row + 1, 0, u'Расстояния точек до центра сферы:')
                    for i in range(plen):
                        ws.write(row + 2, i, u'{}'.format(i), s_table)
                    row += 3

                for i in ncptsi:

                    # Find distance between centre and points.
                    distance = 0.0
                    for j in range(self.dimension):
                        distance += (centre[j] - self.points[i][j]) ** 2
                    distance = math.sqrt(distance)

                    # Ending of table of distances between points and sphere centre.
                    if self.xls_enable:
                        ws.write(row, i, u'{}'.format(distance), s_table)

                    # Make decision to include into the cluster.
                    if distance < radius:
                        cluster.append(i)

                # Write cluster into xls.
                if self.xls_enable:
                    row += 1
                    ws.write(row, 0, u'В итоге получился кластер: {}'.format(cluster))
                    row += 1

                # Recalc the centre.
                for j in range(self.dimension):
                    centre[j] = 0
                    for point in cluster:
                        centre[j] += self.points[point][j]
                    centre[j] /= len(cluster)

                # Write new sphere centre into xls.
                if self.xls_enable:
                    ws.write(row, 0, u'Новый центр сферы: {}'.format(centre))
                    row += 1

                cluster.sort()

                if self.xls_enable:
                    if centre != prev_centre:
                        ws.write(row, 0, u'Новый центр сферы не равен предыдущему, продолжим вычисления')
                    else:
                        ws.write(row, 0, u'Новый центр сферы равен предыдущему, кластер сформирован')
                    row += 2

            clusters.append(cluster)
            for point in cluster:
                ncptsi.remove(point)

        # Write conclusion.
        if self.xls_enable:
            row += 2
            ws.write(row, 0, u'В итоге получились кластеры:')
            row += 1
            for i in range(len(clusters)):
                ws.write(row, 0, u'{}'.format(clusters[i]))
                row += 1

        # Transform indexes into points.
        clusters_i = clusters
        clusters = [[0 for i in range(len(clusters_i[j]))] for j in range(len(clusters_i))]
        for i in range(len(clusters_i)):
            for j in range(len(clusters_i[i])):
                clusters[i][j] = self.points[clusters_i[i][j]][:]

        # Save xls.
        if self.xls_enable:
            wb.save(self.xls_file_name)

        self.clusters = clusters[:]
        return clusters


    def crab(self, nclusters):
        """
        Implements "Crab" Clusterisation Method.

        :param nclusters:
            Number of clusters to make.
        :return:
            Edges.
        """

        plen = len(self.points)
        # Not clustered points indexes
        ncptsi = [i for i in range(plen)]
        cptsi = []
        edges = []

        # Preparations for xls writing
        wb = None
        ws = None
        s_table = None
        row = 0
        step = 1
        if self.xls_enable:
            wb = xlwt.Workbook()
            ws = wb.add_sheet('Results')
            s_table = xlwt.XFStyle()
            s_table.borders.bottom = 1
            s_table.borders.top = 1
            s_table.borders.left = 1
            s_table.borders.right = 1

        # Write initial centres into xls.
        if self.xls_enable:
            ws.write(row, 0, u'Заданное число кластеров: {}'.format(nclusters))
            row += 2

        # Write points into xls.
        if self.xls_enable:
            row = self.__write_points_into_xls_sheet(ws, row)

        # Write the distance matrix into xls.
        if self.xls_enable:
            row = self.__write_dist_table_into_xls_sheet(ws, row)

        # Find min distance.
        imin, jmin = 0, 1
        distmin = self.dist_matrix[imin][jmin]
        for i in range(plen - 1):
            for j in range(i + 1, plen):
                if self.dist_matrix[i][j] < distmin:
                    distmin = self.dist_matrix[i][j]
                    imin, jmin = i, j

        # Add edge between points with min distance.
        edges.append([imin, jmin, distmin])
        cptsi.append(imin)
        cptsi.append(jmin)
        ncptsi.remove(imin)
        ncptsi.remove(jmin)

        # Write min distance into xls.
        if self.xls_enable:
            ws.write(row, 0, u'Две самые ближайшие точки: {} и {} ({})'.format(imin, jmin, distmin))
            row += 2
            ws.write(row, 0, u'Найдем из оставшихся точек ближайшие к уже рассмотренным.')
            row += 1
            ws.write(row, 0, u'Рассм.', s_table)
            ws.write(row, 1, u'Ост.', s_table)
            ws.write(row, 2, u'Расст.', s_table)
            row += 1

        while ncptsi:
            # Find min distance between added and not
            # added points.
            imin = cptsi[0]
            jmin = ncptsi[0]
            distmin = self.dist_matrix[imin][jmin]
            for i in cptsi:
                for j in ncptsi:
                    if self.dist_matrix[i][j] < distmin:
                        distmin = self.dist_matrix[i][j]
                        imin = i
                        jmin = j

            # Write min distance into xls.
            if self.xls_enable:
                ws.write(row, 0, u'{}'.format(imin), s_table)
                ws.write(row, 1, u'{}'.format(jmin), s_table)
                ws.write(row, 2, u'{}'.format(distmin), s_table)
                row += 1

            # Add edge between points with min distance.
            edges.append([imin, jmin, distmin])
            cptsi.append(jmin)
            ncptsi.remove(jmin)

        # Remove edges with maximum distance.
        to_remove = None
        edges.sort(key = lambda i: i[2])
        if nclusters <= 1:
            to_remove = []
        else:
            to_remove = edges[-(nclusters - 1):]
            edges = edges[:-(nclusters - 1)]

        # Write deleted edges into xls.
        if self.xls_enable:
            row += 1
            ws.write(row, 0, u'Удалим самые длинные ребра между вершинами в количестве {} шт.'.format(nclusters - 1))
            row += 1
            ws.write(row, 0, u'Верш.', s_table)
            ws.write(row, 1, u'Верш.', s_table)
            ws.write(row, 2, u'Расст.', s_table)
            row += 1
            for edge in to_remove:
                ws.write(row, 0, u'{}'.format(edge[0]), s_table)
                ws.write(row, 1, u'{}'.format(edge[1]), s_table)
                ws.write(row, 2, u'{}'.format(edge[2]), s_table)
                row += 1

        # Save xls.
        if self.xls_enable:
            wb.save(self.xls_file_name)

        return edges


    def draw_edges(self, edges):
        """

        :param edges:
            Edges to draw.
        :return:
            Nothing.
        """
        # 2D Drawing
        if self.dimension == 2:
            min_x = min([self.points[i][0] for i in range(len(self.points))])
            max_x = max([self.points[i][0] for i in range(len(self.points))])
            min_y = min([self.points[i][1] for i in range(len(self.points))])
            max_y = max([self.points[i][1] for i in range(len(self.points))])
            wh = 0  # width and simul. heigth
            if (max_x - min_x) > (max_y - min_y):
                wh = max_x - min_x
            else:
                wh = max_y - min_y
            plt.xlim(min_x - wh/12.0, min_x + wh + wh/12.0)
            plt.ylim(min_y - wh/12.0, min_y + wh + wh/12.0)

            plen = len(self.points)

            # Draw points.
            x = [self.points[i][0] for i in range(plen)]
            y = [self.points[i][1] for i in range(plen)]
            plt.plot(x, y, 'ro')

            # Draw labels.
            for X, Y, L in zip(x, y, self.labels):
                plt.annotate(
                  L, xy = (X, Y), xytext = (-10, 10),
                  textcoords = 'offset points', ha = 'center', va = 'bottom',
                  bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5))

            # Draw edges.
            for edge in edges:
                x = [self.points[edge[i]][0] for i in [0, 1]]
                y = [self.points[edge[i]][1] for i in [0, 1]]
                plt.plot(x, y, 'ro-')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

        # 3D Drawing
        elif self.dimension == 3:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            labels = []
            x2, y2 = 0, 0

            # Draw points.
            for point, l in zip(self.points, self.labels):
                ax.scatter([point[0]], [point[1]], [point[2]])

                # Draw labels.
                x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
                labels.append(plt.annotate(
                    l,
                    xy = (x2, y2), xytext = (-10, 10),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5)))

            # Recount labels position.
            def update_position(e):
                for point, label in zip(self.points, labels):
                    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
                    label.xy = x2, y2
                    label.update_positions(fig.canvas.renderer)
                    fig.canvas.draw()
            fig.canvas.mpl_connect('button_release_event', update_position)

            # Draw edges.
            for edge in edges:
                x = [self.points[edge[i]][0] for i in (0, 1)]
                y = [self.points[edge[i]][1] for i in (0, 1)]
                z = [self.points[edge[i]][2] for i in (0, 1)]
                ax.plot_wireframe(x, y, z)
                ax.scatter(x, y, z)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()


# Call if this is main module
# (not included)
if __name__ == '__main__':
    pts = [[41.5, 26.5], [42.3, 24.5], [42.0, 24.5],
           [38.2, 25.5], [39.3, 26.0], [41.5, 29.5],
           [45.5, 30.0], [39.5, 30.5], [42.0, 30.5],
           [49.8, 30.5], [39.2, 31.0], [41.9, 27.0],
           [45.6, 27.5], [38.1, 30.5], [44.2, 30.5]]

    pts3d = [[41.5, 26.5, 1.1], [42.3, 24.5, 1.1], [42.0, 24.5, 4.1],
             [38.2, 25.5, 2.1], [39.3, 26.0, 2.1], [41.5, 29.5, 3.1],
             [45.5, 30.0, 3.1], [39.5, 30.5, 3.1], [42.0, 30.5, 2.1],
             [49.8, 30.5, 4.1], [39.2, 31.0, 2.1], [41.9, 27.0, 1.1],
             [45.6, 27.5, 2.1], [38.1, 30.5, 1.1], [44.2, 30.5, 1.1]]

    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              '11', '12', '13', '14', '15']

    cl = Clusterization(pts3d)
    cl.xls_enable_output("output.xls")

    # edges = cl.crab(1)
    # cl.draw_edges(edges)

    cl.k_middle([1,5,7])
    cl.draw()