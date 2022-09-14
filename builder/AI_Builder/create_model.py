import os

from past.builtins import raw_input
from tqdm import tqdm #show progress bar on for loops

from model_class import Model
from layers import Input, Fully_Connected, Convolutional_2D
import numpy as np
import pickle

os.system('clear')
print("What would you like to do?\n\n1. Create new model \n2. Load existing model\n")
in1 = input('> ')

os.system('clear')
model = Model()
if in1 == 1: #create new model
    print('\nWhat would you like to name your model?\n')
    in2 = raw_input('> ')
    model.name = in2
    print('Model created successfuly!')
elif in1 == 2:
    print('\nWhat is the name of your model?\n')
    in2 = raw_input('> ')
    model = pickle.load(open('{}.p'.format(in2), 'rb'))

editing = True
while(editing):
    os.system('clear')
    print('\n\nWhat would you like to do with your model?\n\n1. Add a layer\n2. Remove a layer\n3. Display the model\n4. Save Model\n5. Quit')
    in3 = input('> ')
    if in3 == 1:
        print('\nWhat kind of layer would you like to add?\n1. Input Layer\n2. Fully Connected Layer\n3. Convolutional Layer\n4. Nevermind')
        in4 = input('> ')
        if in4 == 1:
            name = raw_input('What would you like to name this layer? > ')
            hm_nodes = raw_input('This kind of layer should have the same number of nodes as the data. How many nodes should be in this layer? > ')
            layer = Input(name, hm_nodes)
            model.add_layer(layer)
        if in4 == 2:
            name = raw_input('What would you like to name this layer? > ')
            hm_nodes = raw_input('How many nodes should be in this layer? > ')
            layer = Fully_Connected(name, hm_nodes)
            model.add_layer(layer)
        if in4 == 3:
            name = raw_input('What would you like to name this layer? > ')
            hm_nodes = raw_input('How many nodes should be in this layer? > ')
            layer = Convolutional(name, hm_nodes)
            model.add_layer(layer)
    if in3 == 2:
        print('\nWhat layer index would you like to remove? (Use \'999\' to not remove a layer)')
        in5 = input('> ')
        if in5 != 999:
            in6 = input('Are you sure you want to delete the {} layer named {}? (Y/N)'.format(model.layers[in5].type, model.layers[in5].name))
            if in6.isEqual('Y'):
                model.del_layer(in5)
                print('Deleted Successfuly')
                _ = raw_input('Press enter to continue.')
            else:
                print('Layer not deleted.')
                _ = raw_input('Press enter to continue.')
    if in3 == 3:
        print('\n=================================\n=================================\n')
        print('Model name: {}\n'.format(model.name))
        for i in range(len(model.layers)):
            layer = model.layers[i]
            print('Layer {}: \t Type: {} \tNodes: {}\tName: {}'.format(i, layer.type, layer.hm_nodes, layer.name))
        print('\n=================================\n=================================\n')
        _ = raw_input('Press enter to continue.')
    if in3 == 4:
        pickle.dump(model, open('{}.p'.format(model.name),"wb"))
        print('Saved successfuly!')
    if in3 == 5:
        editing = False
