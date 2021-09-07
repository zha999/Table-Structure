from PyQt5 import QtCore, QtGui, QtWidgets,QtWebEngineWidgets
import os
import sys


class Ui_Dialog(object):
    def setupUi(self, Dialog, path):
        Dialog.setObjectName("Dialog")
        Dialog.resize(600, 500)

        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.centralwidget = QtWidgets.QWidget(Dialog)
        self.centralwidget.setObjectName("centralwidget")

        self.webEngineView = QtWebEngineWidgets.QWebEngineView(self.centralwidget)

        html = QtCore.QUrl().fromLocalFile(path)
        self.webEngineView.load(html)
        self.verticalLayout.addWidget(self.webEngineView)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("result", "result"))

if __name__ == "__main__":
    struct_res_file = r"E:\PyWindow\page_1_tbl_0.png.html"

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog, struct_res_file)
    Dialog.show()
    sys.exit(app.exec_())