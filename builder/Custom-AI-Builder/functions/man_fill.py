man_msg = """
What would you like to fill?

You may:
1: Fill size of the layers,
2: Fill the weights,
3: Fill the biases,
0: Or Return to the main interface.
"""

def function():
  """Writes your input to the data files.
  
  Your input must be a list of integers from 0 to 1
  """
  while True:
    user_req = input(man_msg)
    if user_req == "1":
      with open("data/layers_size.txt", "w") as f: 
        f.write(input("Enter the size of the layers: \n"))
    elif user_req == "2":
      with open("data/weights.txt", "w") as f: 
        f.write(input("Enter the weights: \n"))
    elif user_req == "3":
      with open("data/biases.txt", "w") as f: 
        f.write(input("Enter the biases: \n"))
    elif user_req == "0": break
    else: print("Please enter 1, 2, 3, or 0.")