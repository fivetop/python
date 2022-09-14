#Train existing models on pre-processed data
import os
import tflearn
from tqdm import tqdm
from model_class import Model
from layers import Input, Fully_Connected, Convolutional
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pandas as pd
import pickle
import numpy as np

#
# breast-cancer-wisconsin.csv-processed.csv
#

os.system('clear') #clear the screen

#Load model
model_obj = Model()
print('What is the name of your model?')
model_name = raw_input('> ')
model_obj = pickle.load(open('{}.p'.format(model_name), 'rb'))
print('Model loaded successfuly.')
#_ = raw_input()

###########Format data for network###########
print('What is the name of your preprocessed training data?')
file_name = raw_input('> ')
df = pd.read_csv(file_name)
print('\nFile loaded successfuly.')
print('\nWhat is the label of the column the the network should be predicting?')
target_col = raw_input('> ')

print('\nWhat percentage of the data should be used for accuracy testing? Please enter as a float (ex: 0.2)')
test_perc = raw_input('> ')

print('\nWhat learning rate would you like to use?')
learning_rate = raw_input('> ')

print('\nHow many epochs do you want to train through?')
hm_epochs = raw_input('> ')

print('Processing Training Data')
hm_test_rows = int(len(df.index) * float(test_perc))
hm_train_rows = len(df.index) - hm_test_rows

train = df.head(hm_train_rows) #just rows
test = df.tail(hm_test_rows)

X = np.array(train.drop([target_col],1).astype(float)) #drop target_col
X = np.array(X).reshape(hm_train_rows, 9) #turn into readable shape
y = np.array(train[target_col]) #get only target_col

new_y = []
for i in range(len(y)): #convert class to tensors
    if y[i] == 2:
        new_y.append([1,0]) #benign
    elif y[i] == 4:
        new_y.append([0,1]) #malignant
y = new_y

y = np.array(y).reshape(hm_train_rows, 2) #turn into readable shape
print(y.shape)

test_X = np.array(test.drop([target_col],1).astype(float))
test_X = np.array(test_X).reshape(hm_test_rows, 9)
test_y = np.array(test[target_col])

new_test_y = []
for i in range(len(test_y)): #convert class to tensors
    if test_y[i] == 2:
        new_test_y.append([1,0]) #benign
    elif test_y[i] == 4:
        new_test_y.append([0,1]) #malignant
test_y = new_test_y

test_y = np.array(test_y).reshape(hm_test_rows, 2)

###########Rebuild Model###########
print('Building network')

import tensorflow as tf
tf.reset_default_graph()

#Confirm input layer
input_layer = model_obj.layers[0]
if int(input_layer.hm_nodes) != int(len(df.columns)-1):
    print("Your input layer and training data do not match. The model will crash.\nLayer has {} nodes and the data requires {}".format(input_layer.hm_nodes, (len(df.columns)-1)))
    _ = raw_input()

#net = input_data(shape=[None, input_layer.hm_nodes, input_layer.hm_nodes, 1], name=input_layer.name) #Start net with input layer
net = input_data(shape=[None,input_layer.hm_nodes], name=input_layer.name) #Start net with input layer

for layer in tqdm(model_obj.layers):
    if layer.type == 'input_layer':
        #Do nothing because input was already created
        pass
    elif layer.type == 'fully_connected':
        #Add a fully_connected layer
        net = fully_connected(net, layer.hm_nodes, activation='relu')
    elif layer.type == 'convolutional':
        #not implimented yet
        pass

net = regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

###########Training###########
print('Training Model')

model = tflearn.DNN(net) #, tensorboard_dir='log')
#model.fit({'input': X}, {'targets': y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_y}), show_metric=True, run_id=model_name)

model.fit(X, y, n_epoch=hm_epochs, validation_set=(test_X, test_y), show_metric=True, run_id=model_name)

model.save(model_name)
