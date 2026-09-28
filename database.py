import json
import sqlite3
from pathlib import Path

from config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT,
    title TEXT NOT NULL,
    artist TEXT,
    path TEXT NOT NULL UNIQUE,
    duration REAL,
    source_url TEXT,
    thumbnail TEXT,
    chroma_path TEXT,
    global_feature_json TEXT
);

CREATE TABLE IF NOT EXISTS fingerprints (
    hash TEXT NOT NULL,
    song_id INTEGER NOT NULL,
    offset INTEGER NOT NULL,
    FOREIGN KEY(song_id) REFERENCES songs(song_id)
);

CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON fingerprints(hash);
CREATE INDEX IF NOT EXISTS idx_fingerprint_song ON fingerprints(song_id);
"""

def connect(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def initialize(db_path=DB_PATH):
    con = connect(db_path)
    con.executescript(SCHEMA)
    con.commit()
    return con

def rebuild(db_path=DB_PATH):
    path = Path(db_path)
    if path.exists():
        path.unlink()
    return initialize(db_path)

def insert_song(
    con,
    *,
    video_id,
    title,
    artist,
    path,
    duration,
    source_url=None,
    thumbnail=None,
    chroma_path=None,
    global_feature=None
):
    sql = (
        "INSERT INTO songs("
        "video_id,title,artist,path,duration,source_url,thumbnail,chroma_path,global_feature_json"
        ") VALUES(?,?,?,?,?,?,?,?,?)"
    )
    cur = con.execute(
        sql,
        (
            video_id,
            title,
            artist,
            str(path),
            float(duration),
            source_url,
            thumbnail,
            str(chroma_path) if chroma_path else None,
            json.dumps(global_feature.tolist()) if global_feature is not None else None,
        ),
    )
    return int(cur.lastrowid)

def insert_fingerprints(con, song_id, hashes):
    con.executemany(
        "INSERT INTO fingerprints(hash,song_id,offset) VALUES(?,?,?)",
        [(h, int(song_id), int(offset)) for h, offset in hashes],
    )

def fetch_matching_hash_rows(con, query_hashes, batch_size=800):
    unique_hashes = list(dict.fromkeys(h for h, _ in query_hashes))
    rows = []

    for i in range(0, len(unique_hashes), batch_size):
        batch = unique_hashes[i:i + batch_size]
        if not batch:
            continue
        placeholders = ",".join("?" for _ in batch)
        sql = f"SELECT hash,song_id,offset FROM fingerprints WHERE hash IN ({placeholders})"
        rows.extend(con.execute(sql, batch).fetchall())

    return [(r["hash"], r["song_id"], r["offset"]) for r in rows]

def get_song(con, song_id):
    return con.execute(
        "SELECT * FROM songs WHERE song_id=?",
        (int(song_id),)
    ).fetchone()

def list_songs(con):
    return con.execute(
        "SELECT * FROM songs ORDER BY song_id"
    ).fetchall()

def count_songs(con):
    return int(con.execute("SELECT COUNT(*) FROM songs").fetchone()[0])
