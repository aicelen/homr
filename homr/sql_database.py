import os
import sqlite3
from pathlib import Path
from dataclasses import dataclass

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")
datagen_path = os.path.join(dataset_root, "datagen")
org_images_path = os.path.join(datagen_path, "org_images")
datagen_train_index = os.path.join(datagen_path, "index.txt")
os.makedirs(org_images_path, exist_ok=True)

@dataclass
class Page:
    name: str
    musicxml: str
    staffs: list[str]
    layout: list[int]

class DataGen:
    def __init__(self):
        self.conn = sqlite3.connect(
            os.path.join(datagen_path, "datagen.db"),
            timeout=30
        )

        self.conn.execute(
            "PRAGMA journal_mode=WAL;"
        )  # so we can access it from multiple py instances
        self.conn.execute("PRAGMA busy_timeout=30000;") # timeout
        self.cursor = self.conn.cursor()
        self.create()

    def create(self):
        "Creates the database"
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS page(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            musicxml TEXT,
            name TEXT,
            layout TEXT
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

    def add_page(self, path_to_image: str, musicxml: str, name: str) -> int:
        "adds a page and returns the created id"
        self.cursor.execute(
            "INSERT INTO page (path, musicxml, name) VALUES (?,?,?)",
            (
                path_to_image,
                musicxml,
                name,
            ),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def add_staff(self, page_id: int, path_to_image: str, tokens: str):
        "adds a staff"
        self.cursor.execute(
            "INSERT INTO staff (page_id, path, tokens) VALUES (?,?,?)",
            (
                page_id,
                path_to_image,
                str(tokens),
            ),
        )
        self.conn.commit()

    def add_layout(self, layout: list, page_id: int):
        "adds the layout information of the staff, for example [4,4,4,4] 4 systems with 4 staffs each"
        self.cursor.execute(
            "UPDATE page SET layout = ? WHERE id = ?",
            (
                str(layout).strip("[]"),
                page_id,
            ),
        )
        self.conn.commit()


    def get_data_samples(self, id: int) -> Page:
        """
        Returns all staffs from one given page id
        """
        # Get staffs. Order by id so that we get the same voice-major order
        # in which the staffs were inserted (see parse_staffs).
        self.cursor.execute(
            "SELECT path FROM staff WHERE page_id = ? ORDER BY id",
            (id,),
        )
        staffs = [row[0] for row in self.cursor.fetchall()]

        self.cursor.execute(
            """
            SELECT name, musicxml, layout
            FROM page
            WHERE id == ?
            """,
            (id,),
        )
        data_from_page = self.cursor.fetchone()
        layout = [int(x) for x in data_from_page[2].strip("[]").split(",")]
        return Page(data_from_page[0], data_from_page[1], staffs, layout)

    def get_page_count(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM page")
        return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()


datagen_db = DataGen()
