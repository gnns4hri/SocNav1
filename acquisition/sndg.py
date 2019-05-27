import sys
import math
import random

try:
    from torch.utils.data import DataLoader
    import dgl
    import torch
    import socnav
except:
    pass

import numpy as np
from WorldGenerator import WorldGenerator

from PySide2 import QtCore, QtGui, QtWidgets
from ui_sndg import Ui_MainWindow

def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

class SNDG_APP(QtWidgets.QMainWindow):
    def __init__(self, args, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.world = None
        self.model = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.graphicsView.show()
        self.ui.graphicsView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.graphicsView.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.labels = [ self.ui.label0, self.ui.label1, self.ui.label2, self.ui.label3, self.ui.label4, self.ui.label5 ]

        self.dataset = None
        if len(args) == 1:
            if args[0].endswith('.json'):
                self.dataset = open(args[0], 'r').readlines() # Get all the dataset including duplicates
                random.shuffle(self.dataset)
            else:
                print('The dataset file name should be a json file (check extension).')
                sys.exit(-1)
        elif len(args)>1:
            print('The **optional** dataset file name "dataset.json" should be the only parameter.')
            sys.exit(-1)

    @QtCore.Slot(int)
    def on_slider_valueChanged(self, value):
        value = float(len(self.labels)-1) * float(value)/float(self.ui.slider.maximum()-self.ui.slider.minimum())
        for i, label in zip(range(len(self.labels)), self.labels):
            v = 1.-math.fabs(i-value)
            if v < 0: v = 0
            label.setStyleSheet('color: rgba(0, 0, 0, {});'.format(v))
        self.ui.sendButton.setEnabled(True)


    @QtCore.Slot()
    def on_sendButton_clicked(self):
        textVal = str(self.ui.slider.value()).zfill(3)
        self.ui.statusbar.showMessage("send" + textVal)
        self.ui.sendButton.setEnabled(False)
        self.world.serialize(self.ui.slider.value())
        self.on_getButton_clicked()


    @QtCore.Slot()
    def on_getButton_clicked(self):
        self.populateWorld()
        if self.ui.estimateBox.isChecked():
            self.on_estimateButton_clicked()


    def generateDataset(self, number):
        for i in range(number):
            self.on_sendButton_clicked()


    @QtCore.Slot()
    def on_estimateButton_clicked(self):
        import torch
        import pickle
        import gat
        # import gcn

        self.ui.statusbar.showMessage("estimate")
        if self.model is None:
            params = pickle.load(open('model.prms', 'rb'))
            # self.model = gcn.GCN(*params)
            self.model = gat.GAT(*params)
            self.model.load_state_dict(torch.load('model.tch'))
            self.model.eval()


        device = torch.device("cpu")
        structure = self.world.serialize()
        train_dataset = socnav.SocNavDataset(structure, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(device)
            self.model.g = subgraph
            # for layer in self.model.layers:
            for layer in self.model.gat_layers:
                layer.g = subgraph
            logits = self.model(feats.float())[0].detach().numpy()[0]
            translate = 1. / (1. + math.exp(-logits))*100
            if translate < 0: translate = 0
            if translate > 100: translate = 100
            self.ui.slider.setValue(int(translate))

    def closeEvent(self, event):
        if self.dataset:
            fd = open('saved.json', 'w')
            fd.write(self.current_line)
            for x in self.dataset:
                fd.write(x)
            fd.close()
        event.accept()

    def eventFilter(self, receiver, event):
        if event.type() is not QtCore.QEvent.Type.KeyRelease:
            return super(SNDG_APP, self).eventFilter(receiver, event)
        elif event.key() == 16777235:
            self.ui.slider.setValue(self.ui.slider.value()+1)
        elif event.key() == 16777238:
            self.ui.slider.setValue(self.ui.slider.value()+7)
        elif event.key() == 16777239:
            self.ui.slider.setValue(self.ui.slider.value()-7)
        elif event.key() == 16777237:
            self.ui.slider.setValue(self.ui.slider.value()-1)
        elif event.key() == 16777220:
            if self.ui.sendButton.isEnabled():
                self.on_sendButton_clicked()
        elif event.key() == QtCore.Qt.Key_Home or event.key() == QtCore.Qt.Key_5:
            self.ui.slider.setValue(self.ui.slider.maximum())
        elif event.key() == QtCore.Qt.Key_End or event.key() == QtCore.Qt.Key_1:
            self.ui.slider.setValue(self.ui.slider.minimum())
        elif event.key() == QtCore.Qt.Key_2:
            self.ui.slider.setValue(self.ui.slider.maximum()/4*1)
        elif event.key() == QtCore.Qt.Key_3:
            self.ui.slider.setValue(self.ui.slider.maximum() / 4 * 2)
        elif event.key() == QtCore.Qt.Key_4:
            self.ui.slider.setValue(self.ui.slider.maximum() / 4 * 3)
        event.accept()
        return True

    def populateWorld(self):
        if self.dataset is None:
            self.world = WorldGenerator()
        else:
            try:
                self.current_line = self.dataset.pop(0)
            except IndexError:
                sys.exit(0)
            self.world = WorldGenerator(self.current_line)
        self.ui.graphicsView.setScene(self.world)
        self.ui.sendButton.setEnabled(False)

if __name__ == "__main__":
    import signal
    signal.signal( signal.SIGINT, signal.SIG_DFL )

    app = QtWidgets.QApplication()
    QtCore.qsrand(QtCore.QTime(0, 0, 0).secsTo(QtCore.QTime.currentTime()))
    sndg = SNDG_APP(sys.argv[1:])
    app.installEventFilter(sndg)
    sndg.populateWorld()
    sndg.show()

    # sndg.generateDataset(2500)
    # sys.exit()
    sys.exit(app.exec_())
