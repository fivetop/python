import os

mp4Dir = "e:/home/pi/Videos/mp4"
#if not os.path.isdir(mp4Dir):
#	os.mkdir(mp4Dir)
#os.makedirs('a/b/c', exist_ok=True)

import sys
from PyQt5.QtWidgets import QApplication, QWidget

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()


if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = MyApp()
   sys.exit(app.exec_())