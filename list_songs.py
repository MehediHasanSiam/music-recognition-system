from database import connect, list_songs

def main():
    rows = list_songs(connect())

    if not rows:
        print("Database is empty. Run build_database.py first.")
        return

    print("song_id | title | artist")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['song_id']:>7} | "
            f"{row['title']} | "
            f"{row['artist']}"
        )

if __name__ == "__main__":
    main()
