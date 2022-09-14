# coding: utf-8
import errno
import sys
import os
import numpy

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot # pyqtSlot 프로퍼티를 사용하기 위함
from numpy import var


class Form(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("sitemake.ui", self)

    # Qt Designer에서 버튼의 Clicked와 연결해둔 Slot들
    @pyqtSlot()
    def slot1(self):
        t11 = int(self.ui.lineEdit_11.text())
        t12 = self.ui.lineEdit_12.text()
        t13 = int(self.ui.lineEdit_13.text())
        t21 = int(self.ui.lineEdit_21.text())
        t22 = self.ui.lineEdit_22.text()
        t23 = int(self.ui.lineEdit_23.text())
        t31 = int(self.ui.lineEdit_31.text())
        t32 = self.ui.lineEdit_32.text()
        t33 = int(self.ui.lineEdit_33.text())

        for i in range(t11, t13+1):
            for j in range(t21, t23+1):
                for k in range(t31, t33+1):
                    t55 = str(i) + t12 +' '+ str(j) + t22 +' '+ str(k) + t32
                    self.ui.listWidget.addItem(t55)

    # 저장 처리 로직
    @pyqtSlot()
    def slot2(self):
        if self.ui.listWidget.count() < 1:
            return
        s1 = os.getcwd()
        print(s1)
        try:
            if not (os.path.isdir("test")):
                os.makedirs(os.path.join("test"))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

        os.chdir("test")
        for index in range(self.ui.listWidget.count()):
            f = open(self.ui.listWidget.item(index).text(), 'w')
            f.close()
            print(self.ui.listWidget.item(index).text())
        os.chdir("..\\")
        print(os.getcwd())

    @pyqtSlot()
    def slot3(self):
        sys.exit(app.exec())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    w.show()
    sys.exit(app.exec())