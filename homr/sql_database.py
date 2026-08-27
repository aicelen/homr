import sqlite3
import os
from pathlib import Path

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
datagen_path = os.path.join(dataset_root, "datagen")
os.makedirs(datagen_path, exist_ok=True)

class DataGen:
    def __init__(self):
        self.conn = sqlite3.connect(os.path.join(datagen_path, "datagen.db"))
        self.cursor = self.conn.cursor()

    def create(self):
        "Creates the database"
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS page(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT
            )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS staff(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            path TEXT,
            tokens TEXT,
            FOREIGN KEY (page_id) REFERENCES page(id)
                ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def add_page(self, path_to_image: str) -> int:
        "adds a page and returns the created id"
        self.cursor.execute("INSERT INTO page (path) VALUES (?)", (path_to_image,))
        return self.cursor.lastrowid


    def add_staff(self, page_id: int, path_to_image: str, tokens: str):
        "adds a staff"
        self.cursor.execute("INSERT INTO staff (page_id, path, tokens) VALUES (?,?,?)", (page_id, path_to_image, str(tokens),))

    def close(self):
        self.conn.close()

datagen_db = DataGen()
datagen_db.create()
