import sqlite3
from pathlib import Path

db_path = Path("data") / "music.db"

con = sqlite3.connect(db_path)

rows = con.execute(
    "SELECT song_id, chroma_path FROM songs"
).fetchall()

for song_id, old_path in rows:

    if old_path:
        filename = Path(old_path).name
    else:
        filename = f"song_{song_id:05d}.npz"

    new_path = f"data/features/{filename}"

    con.execute(
        "UPDATE songs SET chroma_path=? WHERE song_id=?",
        (new_path, song_id)
    )

con.commit()
con.close()

print("Database converted to portable feature paths.")