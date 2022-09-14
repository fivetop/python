auto_msg = """
What would you like to do?

You may:
1: Use the current layers size,
2: Enter your own layers size,
0: Or Cancel this action.
"""


def function():
  """Fills the weights and biases with dummy values.
  
  If you enter your own layers size the layers_size.txt file is updated.
  """
  while True:
    user_req = input(auto_msg)
    if user_req == "1": 
      with open("data/layers_size.txt") as f: layers_size = eval(f.read())
    elif user_req == "2": 
      layers_size = eval(input("Enter the size of the layers as a list: \n"))
      with open("data/layers_size.txt", "w") as f: f.write(str(layers_size))
    elif user_req == "0": return
    else:
      print("Please enter 1, 2, or 0.")
      continue
    break
  
  z = []
  for i in range(1, len(layers_size)):
    y = []
    for a in range(layers_size[i]):
      x = []
      for b in range(layers_size[i-1]):
        x.append(0)
      y.append(x)
    z.append(y)
  
  with open("data/weights.txt", "w") as f: f.write(str(z))

  z = []
  for i in range(1, len(layers_size)):
    y = []
    for a in range(layers_size[i]):
      y.append(0)
    z.append(y)
  
  with open("data/biases.txt", "w") as f: f.write(str(z))