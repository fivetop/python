create_table = """
CREATE TABLE IF NOT EXISTS training_data (
  id INTEGER PRIMARY KEY,
  input TEXT NOT NULL,
  output TEXT NOT NULL,
  last_used INTEGER DEFAULT 0
);
"""

insert_row = """
INSERT INTO training_data (input, output)
VALUES(?, ?);
"""

get_row = """
SELECT id, input, output 
FROM training_data 
ORDER BY last_used
LIMIT 1;
"""

set_used = """
UPDATE training_data 
SET last_used = (
  SELECT MAX(last_used)
  FROM training_data
) + 1
WHERE id = ?;
"""

delete_row = """
DELETE FROM training_data
WHERE id = ?;
"""