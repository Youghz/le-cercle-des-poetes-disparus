"""
Étape 2 : Préparer les données de fine-tuning.

Transforme les textes bruts en un dataset JSONL au format chat :
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Stratégie : découper les textes en passages et créer des paires
(question sur un thème → réponse dans le style de l'auteur).
"""

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_texts"
OUTPUT_DIR = Path(__file__).parent / "datasets"

SYSTEM_PROMPTS = {
    "hugo": (
        "Tu es Victor Hugo, écrivain romantique français. "
        "Tu t'exprimes avec lyrisme, passion et grandeur."
    ),
    "voltaire": (
        "Tu es Voltaire, philosophe des Lumières. "
        "Tu t'exprimes avec ironie, raison et mordant."
    ),
    "rousseau": (
        "Tu es Jean-Jacques Rousseau, philosophe genevois. "
        "Tu t'exprimes avec sincérité, sensibilité et profondeur."
    ),
}

# Prompts variés pour le fine-tuning
USER_PROMPTS = [
    "Donne ton opinion sur ce passage.",
    "Continue cette réflexion dans ton style.",
    "Que penses-tu de cette idée ?",
    "Développe cette pensée.",
    "Réagis à ce texte.",
]


def clean_gutenberg_text(text: str) -> str:
    """Retire les en-têtes/pieds de page Gutenberg."""
    # Chercher le début du texte réel
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.index("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def split_into_passages(text: str, min_len: int = 200, max_len: int = 800) -> list[str]:
    """Découpe le texte en passages de taille raisonnable."""
    # Découper par paragraphes (double saut de ligne)
    paragraphs = re.split(r"\n\s*\n", text)

    passages = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 50:
            continue

        if len(current) + len(para) < max_len:
            current += "\n\n" + para if current else para
        else:
            if len(current) >= min_len:
                passages.append(current.strip())
            current = para

    if len(current) >= min_len:
        passages.append(current.strip())

    return passages


def create_training_examples(author: str, passages: list[str]) -> list[dict]:
    """Crée des exemples d'entraînement au format chat."""
    system_prompt = SYSTEM_PROMPTS[author]
    examples = []

    for i, passage in enumerate(passages):
        user_prompt = USER_PROMPTS[i % len(USER_PROMPTS)]

        # Si le passage est long, on coupe en contexte + réponse
        if len(passage) > 400:
            mid = len(passage) // 3
            # Trouver la fin de phrase la plus proche
            cut = passage.find(".", mid)
            if cut == -1 or cut > mid + 100:
                cut = mid
            context = passage[:cut + 1].strip()
            response = passage[cut + 1:].strip()
        else:
            context = ""
            response = passage

        if not response or len(response) < 100:
            continue

        user_content = f"{user_prompt}\n\n{context}" if context else user_prompt

        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
        })

    return examples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Préparation des datasets de fine-tuning")
    print("=" * 50)

    for author in SYSTEM_PROMPTS:
        author_dir = RAW_DIR / author
        if not author_dir.exists():
            print(f"\n[SKIP] Pas de textes pour {author}. Lancez d'abord 01_download_texts.py")
            continue

        print(f"\n--- {author.upper()} ---")
        all_examples = []

        for txt_file in sorted(author_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8", errors="replace")
            cleaned = clean_gutenberg_text(raw)
            passages = split_into_passages(cleaned)
            examples = create_training_examples(author, passages)
            all_examples.extend(examples)
            print(f"  {txt_file.name}: {len(passages)} passages → {len(examples)} exemples")

        # Sauvegarder en JSONL
        output_file = OUTPUT_DIR / f"{author}_train.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  Total : {len(all_examples)} exemples → {output_file.name}")

    print(f"\nDatasets dans : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
