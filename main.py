#!/usr/bin/python
# -*- coding: utf-8 -*-

# Importing libraries:
# Standart;
import sys

# Not standart;
from PyQt4 import QtCore, QtGui
import cPickle as pickle

# My.
import form
import clusterization


class ClusterizationGUI (QtGui.QWidget, form.Ui_Form):
    """
    Implements GUI for clusterization library.
    """

    def __init__ (self):
        """
        Constructor.
        """

        # Call constructors of superclasses
        QtGui.QWidget.__init__(self)
        form.Ui_Form.setupUi(self, self)

        # Clusteristation object.
        self.cl = clusterization.Clusterization()

        # QT connections.
        self.add_push_button.clicked.connect(self.add_row)
        self.del_push_button.clicked.connect(self.del_row)
        self.save_push_button.clicked.connect(self.save_points)
        self.load_push_button.clicked.connect(self.load_points)
        self.d2_radio_button.clicked.connect(self.set_2d)
        self.d3_radio_button.clicked.connect(self.set_3d)
        self.xls_check_box.clicked.connect(self.set_xls_output)
        self.point_xls_tool_button.clicked.connect(self.point_xls)
        self.xls_name_line_edit.textChanged.connect(self.change_xls_fname)
        self.methods_combo_box.currentIndexChanged.connect(self.enable_methods_stuff)
        self.count_push_button.clicked.connect(self.count)


    def add_row(self):
        """
        Adds new row into points table widget at the end of table.
        """

        row = self.points_table_widget.rowCount()
        self.points_table_widget.insertRow(row)

        # Recount numeration.
        labels = QtCore.QStringList()
        for i in range(row + 1):
            labels << QtCore.QString("%1").arg(i)
        self.points_table_widget.setVerticalHeaderLabels(labels)


    def del_row(self):
        """
        Deletes pointed row.
        """

        row_i = self.points_table_widget.currentRow()
        self.points_table_widget.removeRow(row_i)

        # Recount numeration.
        rowsn = self.points_table_widget.rowCount()
        labels = QtCore.QStringList()
        for i in range(rowsn + 1):
            labels << QtCore.QString("%1").arg(i)
        self.points_table_widget.setVerticalHeaderLabels(labels)


    def save_points(self):
        """
        Saves the points from table into file.
        """

        points = self.__read_points()
        if not points:
            return

        # Pick the points into file.
        fName = QtGui.QFileDialog.getSaveFileName(self, 'Save')
        if not fName:
            return
        fName = fName.toUtf8().data()
        f = open(fName, 'wb')
        pickle.dump(points, f, 2)
        f.close()


    def load_points(self):
        """
        Load points from file into the table.
        """

        # Unpick points from the file.
        f_name = QtGui.QFileDialog.getOpenFileName(self, 'Open')
        if not f_name:
            return
        f_name = f_name.toUtf8().data()
        f = open(f_name, 'rb')
        points = pickle.load(f)
        f.close()

        # Detect and set the dimension.
        dimension = len(points[0])
        self.points_table_widget.setColumnCount(dimension)
        if dimension == 2:
            self.d2_radio_button.setChecked(True)
        elif dimension == 3:
            self.d3_radio_button.setChecked(True)

        # Put points intp the table.
        pts_n = len(points)
        self.points_table_widget.setRowCount(pts_n)
        for i in range(pts_n):
            for j in range(dimension):
                item = QtGui.QTableWidgetItem(QtCore.QString.number(points[i][j]))
                self.points_table_widget.setItem(i, j, item)


    def set_2d(self):
        """
        Set dimension to 2d.
        """

        self.points_table_widget.setColumnCount(2)


    def set_3d(self):
        """
        Set dimension to 3d.
        """

        self.points_table_widget.setColumnCount(3)
        item = QtGui.QTableWidgetItem(u'z')
        self.points_table_widget.setHorizontalHeaderItem(2, item)


    def set_xls_output(self):
        """
        Sets or unsets *.xls output.
        """

        if self.xls_check_box.isChecked():

            # Activate widgets to point the output file name.
            self.xls_name_line_edit.setEnabled(True)
            self.point_xls_tool_button.setEnabled(True)

            self.cl.xls_enable_output(self.xls_name_line_edit.text().toUtf8().data())

        else:

            # Deactivate widgets to point the output file name.
            self.xls_name_line_edit.setEnabled(False)
            self.point_xls_tool_button.setEnabled(False)

            self.cl.xls_disable_output()


    def point_xls(self):
        """
        Point the *.xls file to output results.
        """

        fname = QtGui.QFileDialog.getSaveFileName(self, 'Save')
        if fname:
            self.xls_name_line_edit.setText(fname)
            fname = fname.toUtf8().data()
            self.cl.xls_enable_output(fname)


    def change_xls_fname(self, fname):
        """
        Changet name of *.xls output file.

        :param fname:
            The new file name.
        """

        self.cl.xls_enable_output(fname.toUtf8().data())


    def enable_methods_stuff(self, m_index):
        """
        Enable widgets depends on method.

        :param m_index:
            Method's index.
        """

        # At first disable all.
        self.border_spin_box.setEnabled(False)
        self.radius_spin_box.setEnabled(False)
        self.clusters_number_spin_box.setEnabled(False)
        self.centres_line_edit.setEnabled(False)

        if m_index == 0:    # King
            self.border_spin_box.setEnabled(True)

        elif m_index == 1:  # K-middle
            self.centres_line_edit.setEnabled(True)

        elif m_index == 2:  # Trout
            self.radius_spin_box.setEnabled(True)

        elif m_index == 3:  # Crab
            self.clusters_number_spin_box.setEnabled(True)

        elif m_index == 4:  # Serial
            pass

    def __read_points(self):
        """
        Return points from table.
        """

        pts_n = self.points_table_widget.rowCount()
        dimension = self.points_table_widget.columnCount()
        points = [[0.0 for j in range(dimension)] for i in range(pts_n)]

        # Reading points into the list.
        for i in range(pts_n):
            for j in range(dimension):
                item = self.points_table_widget.item(i,j)
                if not item :
                    QtGui.QMessageBox.about(self, u"Ошибка!", u"Заполните все поля или удалите ненужные.")
                    return None
                points[i][j] = item.text().toDouble()[0]

        return points

    def count(self):
        """
        count_push_botton handler.
        """

        # Get points.
        points = self.__read_points()
        if not points:
            return
        self.cl.set_points(points)

        method = self.methods_combo_box.currentIndex()

        # todo :threading

        # King
        if method == 0:

            # Get limit (or border).
            limit = self.border_spin_box.value()
            limit = limit ** 2

            # Do math.
            self.cl.king(limit)
            self.cl.draw()

        # K-middle
        elif method == 1:

            # Get centres.
            try:
                centres = list(self.centres_line_edit.text().toUtf8().data().split(';'))
                for i in range(len(centres)):
                    centres[i] = int(centres[i])
            except:
                QtGui.QMessageBox.about(self, u"Ошибка!", u"Неправильные центры кластеров.")
                return

            # Do math.
            self.cl.k_middle(centres)
            self.cl.draw()

        # Trout
        elif method == 2:

            # Get radius.
            radius = self.radius_spin_box.value()

            # Do math.
            self.cl.trout(radius)
            self.cl.draw()

        # Crab
        elif method == 3:

            # Get number of clusters.
            n = self.clusters_number_spin_box.value()

            # Do math.
            edges = self.cl.crab(n)
            self.cl.draw_edges(edges)

        # Serial
        elif method == 4:
            # todo: serial
            pass

# Call if this is main module
# (not included)
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    w = ClusterizationGUI()
    w.show()
    app.exec_()