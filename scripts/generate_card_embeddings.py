"""
Generate pretrained card embeddings from card text using sentence-transformers.

Reads card names and descriptions from cards.cdb, encodes them with
all-MiniLM-L6-v2 (384-dim), then reduces to embed_dim via PCA.

Output: data/card_embeddings.npy with shape (vocab_size, embed_dim)
where index 0 is a zero vector (padding).

Usage:
    uv run --group embeddings python scripts/generate_card_embeddings.py
"""

import sqlite3
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cards.cdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "card_embeddings.npy"
EMBED_DIM = 32


def main():
    # Load card codes in the same order as CardIndex
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id FROM datas ORDER BY id")
    card_codes = [row[0] for row in cur.fetchall()]
    print(f"Found {len(card_codes)} cards in datas table")

    # Get text for each card
    texts = []
    for code in card_codes:
        cur.execute("SELECT name, desc FROM texts WHERE id=?", (code,))
        row = cur.fetchone()
        if row:
            name, desc = row
            texts.append(f"{name}: {desc}" if desc else name)
        else:
            texts.append("")
    conn.close()

    non_empty = sum(1 for t in texts if t)
    print(f"Card texts: {non_empty} non-empty, {len(texts) - non_empty} empty")

    # Encode with sentence-transformers
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Encoding card texts...")
    raw_embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    print(f"Raw embeddings shape: {raw_embeddings.shape}")

    # PCA to target dimension
    print(f"Reducing to {EMBED_DIM} dimensions via PCA...")
    pca = PCA(n_components=EMBED_DIM)
    reduced = pca.fit_transform(raw_embeddings)
    variance = sum(pca.explained_variance_ratio_)
    print(f"PCA explained variance: {variance:.1%}")

    # Prepend zero row for padding index 0
    padding = np.zeros((1, EMBED_DIM), dtype=np.float32)
    embeddings = np.vstack([padding, reduced.astype(np.float32)])
    print(f"Final embeddings shape: {embeddings.shape}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), embeddings)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
