#This file holds all of the layer classes
'''
class Layer(object):
    def __init__(self, name, hm_nodes):
        self.name = name
        self.hm_nodes = hm_nodes
'''


#Should mirror TFLearn layers when done

class Input(object):
    def __init__(self, name, hm_nodes):
        self.hm_nodes = hm_nodes
        self.name = name
        self.type = "input_layer"

class Fully_Connected(object):
    def __init__(self, name, hm_nodes):
        self.hm_nodes = hm_nodes
        self.name = name
        self.type = "fully_connected"

class Convolutional_2D(object):
    def __init__(self, name, hm_nodes):
        self.hm_nodes = hm_nodes
        self.name = name
        self.type = 'convolutional2d'

class Dropout(object):
    def __init__(self, name, rate):
        self.rate = rate
        self.name = name
        self.type = 'dropout'

class Max_Pooling_2D(object):
    def __init__(self, name, hm_nodes):
        self.kernel_size = kernel_size
        self.name = name
        self.type = 'max_pooling_2d'
