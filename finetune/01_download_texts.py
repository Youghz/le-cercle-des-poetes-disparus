"""
Étape 1 : Télécharger les textes libres de droit depuis Project Gutenberg.

Tous ces auteurs sont morts avant 1900 → domaine public mondial.
"""

import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_texts"

# Project Gutenberg — textes en français
# Format : https://www.gutenberg.org/cache/epub/{ID}/pg{ID}.txt
SOURCES = {
    "hugo": {
        "Les Misérables - Tome 1 Fantine": 17489,
        "Les Misérables - Tome 2 Cosette": 17490,
        "Les Misérables - Tome 3 Marius": 17491,
        "Notre-Dame de Paris": 19657,
        "Les Contemplations": 54615,
    },
    "voltaire": {
        "Candide": 4650,
        "Lettres philosophiques": 28858,
        "Dictionnaire philosophique": 18569,
        "Zadig": 4647,
    },
    "rousseau": {
        "Du contrat social": 23684,
        "Les Confessions": 3913,
        "Émile ou De l'éducation": 5427,
    },
}


def download_text(ebook_id: int, dest: Path) -> bool:
    url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
    if dest.exists():
        print(f"  Déjà téléchargé : {dest.name}")
        return True
    try:
        print(f"  Téléchargement : {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
        print(f"  OK : {dest.name} ({dest.stat().st_size // 1024} Ko)")
        return True
    except Exception as e:
        print(f"  ERREUR : {e}")
        return False


def main():
    print("=" * 50)
    print("  Téléchargement des textes — Project Gutenberg")
    print("=" * 50)

    for author, works in SOURCES.items():
        author_dir = RAW_DIR / author
        author_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {author.upper()} ---")
        for title, ebook_id in works.items():
            safe_name = title.replace(" ", "_").replace("'", "").replace("é", "e")
            dest = author_dir / f"{safe_name}.txt"
            download_text(ebook_id, dest)

    print("\n\nTéléchargement terminé.")
    print(f"Fichiers dans : {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
