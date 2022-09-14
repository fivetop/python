import sqlite3
from data import sql


SQLite_msg = """
What would you like to do?

You may:
1: Add training data,
2: Delete the training data table,
3: Delete just one row of training data,
4: Run the setup sequence(Create table, index),
5: View all the training data,
0: Or Return to the main interface.
"""


def get_conn():
  """Connects to the database and returns the connection.
  """
  try: conn = sqlite3.connect("data/input.db")
  except Exception as e: print("SQLite Connection Error: ", e)
  else: 
    print("Successfuly connected to the database.")
    return conn


def setup():
  """Runs SQL that is necessary to use the training_data.
  """
  conn = get_conn()
  if conn:
    c = conn.cursor()
    try:
      with conn: c.execute(sql.create_table)
    except Exception as e: print("SQLite Create Table Error: ", e)
    finally: conn.close()



def interface():
  """Executes the SQL that the user requests.

  You can insert data, delete the table, delete a row, or run setup()
  """
  conn = get_conn()
  if conn:
    c = conn.cursor()
    try:
      with conn:
        while True: 
          user_req = input(SQLite_msg)
          if user_req == "1": c.execute(sql.insert_row, (input("Input(as a list): "), input("Output(as a list): ")))
          elif user_req == "2": c.execute("DROP TABLE training_data;")
          elif user_req == "3": c.execute(sql.delete_row, (input("Enter the id of the row: "),))
          elif user_req == "4": setup()
          elif user_req == "5": 
            c.execute("SELECT * FROM training_data")
            print(c.fetchall())
          elif user_req == "0": break
          else: print("Please enter 1, 2, 3, 4, or 0.")
    except Exception as e: print("SQLite Interface Error: ", e)
    finally: conn.close()


def get_row(set_used=False):
  """Returns the row with the lowest last_used.

  If set_used, then the last_used of that row will be set to 1 higher than the largest last_used
  """
  conn = get_conn()
  if conn:
    c = conn.cursor()
    try:
      with conn:
        c.execute(sql.get_row)
        row = c.fetchall()
        if not row: return
        if set_used: c.execute(sql.set_used, (row[0][0],))
    except Exception as e: print("SQLite Get Row/Set last_used Error: ", e)
    else: return row[0][1:]
    finally: conn.close()