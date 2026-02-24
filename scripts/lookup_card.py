
import sqlite3
import sys

db_path = "data/cards.cdb"
card_id = sys.argv[1]

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name, desc FROM texts WHERE id=?", (card_id,))
row = cursor.fetchone()
if row:
    print(f"Card: {row[0]}")
    print(f"Desc: {row[1]}")
else:
    print("Card not found")
