import sys
sys.path.append('./functions/')

import SQLite

from functions import auto_fill, man_fill, run, backpropagate, SQLite

SQLite.setup()

msg = """
What would you like to do?

You may:
1: Automatically fill the NN,
2: Manually fill the NN,
3: Run the NN,
4: Backpropagate,
5: Open the training data interface, 
0: Or Close the program.
"""


while True:
  user_req = input(msg)
  if user_req == "1": auto_fill.function()
  elif user_req == "2": man_fill.function()
  elif user_req == "3": print(run.function(eval(input("Enter the is the input(as a list): \n")))[-1])
  elif user_req == "4": backpropagate.function(int(input("Accuracy: ")), int(input("Times: ")))
  elif user_req == "5": SQLite.interface()
  elif user_req == "0": break
  else: print("Please enter a, m, r, or c.")