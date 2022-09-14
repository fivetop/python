from layers import Input, Fully_Connected, Convolutional_2D

class Model(object):
    def __init__(self):
        self.layers = []

    def set_name(self, name):
        self.name = name

    def add_layer(self, layer):
        self.layers.append(layer)

    def del_layer(self, index):
        del self.layers[index]
