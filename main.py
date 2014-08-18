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

        # QT connections
        self.add_push_button.clicked.connect(self.add_row)
        self.del_push_button.clicked.connect(self.del_row)
        self.save_push_button.clicked.connect(self.save_points)
        self.load_push_button.clicked.connect(self.load_points)
        self.d2_radio_button.clicked.connect(self.set_2d)
        self.d3_radio_button.clicked.connect(self.set_3d)


    def add_row(self):
        """
        Adds new row into points table widget at the end of table.
        """

        row = self.points_table_widget.rowCount()
        self.points_table_widget.insertRow(row)


    def del_row(self):
        """
        Deletes pointed row.
        """

        row_i = self.points_table_widget.currentRow()
        self.points_table_widget.removeRow(row_i)


    def save_points(self):
        """
        Saves the points from table into file.
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
                    return
                points[i][j] = item.text().toDouble()[0]

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

# Call if this is main module
# (not included)
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    w = ClusterizationGUI()
    w.show()
    app.exec_()