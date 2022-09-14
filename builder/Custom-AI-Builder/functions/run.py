def node(weights, previous_layer, bias):
  """Calculates the brightness of a node.
  
  For some node, this function multiplies every node in the previous layer with the coressponding weight and adds all them together.
  After adding the bias, the value is put through the sigmoid and returned.
  """
  e = 2.71828
  x = 0
  for k in range(len(previous_layer)):
    x = x + weights[k] * previous_layer[k]
  return(1 / (1 + e ** (-(x + bias))))


def function(previous_layer):
  """Calls node() for every node and organizes the results.
  
  The information is obtained from text files and provided to Node() and the ouput of Node() is put in a list.

  NN is a list of layers and each layer is a list of the brightness of each node in it.

  The brightness of the nodes in the first layer(the input layer) are not calculated as there is no previous_layer.
  Instead, the input is assigned to the nodes' values.     // Wrong way around??
  """
  NN = []
  with open("data/layers_size.txt") as f: layers_size = eval(f.read())
  with open("data/weights.txt") as f: all_weights = eval(f.read())
  with open("data/biases.txt") as f: all_biases = eval(f.read())
  NN.append(previous_layer)
  for i in range(len(layers_size) - 1):
    current_layer = []
    for j in range(layers_size[i + 1]):
      current_layer.append(node(all_weights[i][j], previous_layer, all_biases[i][j]))
    previous_layer = current_layer
    NN.append(current_layer)

  return(NN)