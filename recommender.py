import json

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from database import connect, list_songs

def recommend_similar(song_id, top_k=5):
    rows = list_songs(connect())

    ids = []
    features = []
    metadata = []

    for row in rows:
        raw = row["global_feature_json"]

        if not raw:
            continue

        ids.append(int(row["song_id"]))
        features.append(
            np.asarray(json.loads(raw), dtype=np.float32)
        )
        metadata.append(row)

    song_id = int(song_id)

    if song_id not in ids or len(ids) < 2:
        return []

    X = np.vstack(features)
    X = StandardScaler().fit_transform(X)
    similarities = cosine_similarity(X)

    i = ids.index(song_id)
    order = np.argsort(-similarities[i])

    result = []

    for j in order:
        if ids[j] == song_id:
            continue

        row = metadata[j]

        result.append({
            "song_id": ids[j],
            "title": row["title"],
            "artist": row["artist"],
            "similarity": float(similarities[i, j]),
            "thumbnail": row["thumbnail"],
            "source_url": row["source_url"],
        })

        if len(result) >= top_k:
            break

    return result
