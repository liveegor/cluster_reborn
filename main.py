#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import Form
from PyQt4 import QtCore, QtGui


class MyForm (QtGui.QWidget, Form.Ui_Form):

    def __init__ (self):
        ''' Default constructor '''
        # Call constructors of superclasses
        QtGui.QWidget.__init__(self)
        Form.Ui_Form.setupUi(self, self)


# Call if this is main module
# (not included)
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    w = MyForm()
    w.show()
    app.exec_()