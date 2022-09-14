from functions import run, SQLite


def cost(a, w, b, y):
  """Determines the optimal weights, biases, and activations for each node.

  The name has nothing to do with what it does
  """
  w2 = [[] for i in range(len(a[-1]))]
  b2 = []
  y2 = [[] for i in range(len(a[-2]))]
  c = [[], []]
  for i in range(len(a[-1])):
    b2.append(b[-1][i] + (a[-1][i] - y[i]))
    for j in range(len(a[-2])):
      w2[i].append(w[-1][i][j] + ((a[-1][i] - y[i]) * abs(a[-2][j])))
      y2[j].append(a[-2][j] + ((a[-1][i] - y[i]) * abs(w[-1][i][j])))
  y2 = list(map(lambda x: sum(x)/len(x), y2))
  if len(a) > 2: c = cost(a[:-1], w[:-1], b[:-1], y2)
  return [c[0] + [w2], c[1] + [b2]]


def function(accuracy, times):
  """Calls cost() for some training data and averages the results
  """
  with open("data/weights.txt") as f: w = eval(f.read())
  with open("data/biases.txt") as f: b = eval(f.read())
  for g in range(times):
    NN_cost = [[], []]
    for h in range(accuracy):
      d = SQLite.get_row(True)
      if not d: break
      c = cost(run.function(eval(d[0])), w, b, eval(d[1]))
      NN_cost[0].append(c[0])
      NN_cost[1].append(c[1])
    if not d: break
    with open("data/weights.txt", "w") as f: f.write(multi_list_averager(NN_cost[0]))
    with open("data/biases.txt", "w") as f: f.write(multi_list_averager(NN_cost[1]))
  if not d: print("SQLite Get Row Error:  No Rows In Table \nPlease insert training data and try again.")


def multi_list_averager(multi_list):
  """Gets the column average for a 2d list of lists.
  """
  avg = []
  for i in range(len(multi_list[0])):
    avg.append([])
    for j in range(len(multi_list[0][i])):
      if type(multi_list[0][i][j]) == list:
        avg[i].append([])
        for k in range(len(multi_list[0][i][j])):
          x = sum([multi_list[l][i][j][k] for l in range(len(multi_list))]) / len(multi_list)
          avg[i][j].append(x)
      else:
        x = sum([multi_list[l][i][j] for l in range(len(multi_list))]) / len(multi_list)
        avg[i].append(x)
  return str(avg)