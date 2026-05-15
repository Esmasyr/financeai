import sqlite3
import hashlib
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "financeai.db")

password = "yenisifre123"

salt = secrets.token_hex(16)
hashed = hashlib.sha256((password + salt).encode()).hexdigest()
final_hash = f"sha256:{salt}:{hashed}"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute(
    "UPDATE users SET password=? WHERE username='manage'",
    (final_hash,)
)

conn.commit()
conn.close()

print("Sifre guncellendi.")
