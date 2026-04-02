"""
Étape 4 : Importer les modèles fine-tunés dans Ollama.

Deux options :
  A) Modelfile avec SYSTEM prompt enrichi (sans fine-tuning, rapide)
  B) Modelfile pointant vers le GGUF fine-tuné (après étapes 1-3)

Ce script crée les deux options.
"""

import subprocess
from pathlib import Path

FINETUNE_DIR = Path(__file__).parent
FUSED_DIR = FINETUNE_DIR / "fused_models"
MODELFILES_DIR = FINETUNE_DIR / "modelfiles"

WRITERS = {
    "hugo": {
        "name": "Victor Hugo",
        "system": (
            "Tu es Victor Hugo, écrivain romantique français du XIXe siècle. "
            "Tu t'exprimes avec lyrisme, passion et grandeur. Tu défends les opprimés, "
            "tu crois au progrès de l'humanité et à la force de l'amour et de la justice. "
            "Tu utilises des métaphores puissantes et un style emphatique. "
            "Tu cites parfois tes propres oeuvres : Les Misérables, Notre-Dame de Paris, "
            "Les Contemplations. Tu parles toujours en français."
        ),
    },
    "voltaire": {
        "name": "Voltaire",
        "system": (
            "Tu es Voltaire, philosophe et écrivain français des Lumières. "
            "Tu es sarcastique, rationnel et mordant. Tu combats les superstitions, "
            "l'intolérance et l'injustice par l'ironie et la raison. "
            "Tu as un esprit vif et provocateur. Tu cites parfois Candide, "
            "le Dictionnaire philosophique, les Lettres philosophiques. "
            "Tu parles toujours en français."
        ),
    },
    "rousseau": {
        "name": "Jean-Jacques Rousseau",
        "system": (
            "Tu es Jean-Jacques Rousseau, philosophe genevois du XVIIIe siècle. "
            "Tu es introspectif, sensible et défenseur de la nature. "
            "Tu prônes le retour à l'authenticité et l'éducation naturelle. "
            "Tu es parfois mélancolique mais toujours sincère. Tu cites parfois "
            "Du contrat social, Les Confessions, Émile. "
            "Tu parles toujours en français."
        ),
    },
}


def create_modelfile_prompt_only(author: str) -> Path:
    """Option A : Modelfile avec SYSTEM prompt seul (pas besoin de fine-tuning)."""
    writer = WRITERS[author]
    content = f"""FROM llama3.1

SYSTEM \"\"\"{writer['system']}\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""
    path = MODELFILES_DIR / f"Modelfile.{author}"
    path.write_text(content)
    return path


def create_modelfile_finetuned(author: str) -> Path | None:
    """Option B : Modelfile pointant vers le modèle fine-tuné (GGUF nécessaire)."""
    # Vérifier si un modèle fusionné existe
    fused_dir = FUSED_DIR / author
    if not fused_dir.exists():
        return None

    writer = WRITERS[author]
    content = f"""FROM {fused_dir.resolve()}

SYSTEM \"\"\"{writer['system']}\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""
    path = MODELFILES_DIR / f"Modelfile.{author}.finetuned"
    path.write_text(content)
    return path


def import_to_ollama(model_name: str, modelfile_path: Path):
    """Crée le modèle dans Ollama."""
    print(f"  ollama create {model_name} -f {modelfile_path}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  OK : {model_name} créé dans Ollama")
    else:
        print(f"  ERREUR : {result.stderr.strip()}")


def main():
    MODELFILES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Import des modèles dans Ollama")
    print("=" * 60)

    # Option A : Modèles avec SYSTEM prompt (toujours disponible)
    print("\n--- Option A : Modèles avec prompt enrichi ---")
    for author in WRITERS:
        modelfile = create_modelfile_prompt_only(author)
        model_name = f"{author}-writer"
        import_to_ollama(model_name, modelfile)

    # Option B : Modèles fine-tunés (si disponibles)
    print("\n--- Option B : Modèles fine-tunés ---")
    for author in WRITERS:
        modelfile = create_modelfile_finetuned(author)
        if modelfile:
            model_name = f"{author}-finetuned"
            import_to_ollama(model_name, modelfile)
        else:
            print(f"  [SKIP] Pas de modèle fine-tuné pour {author}")
            print(f"         Lancez d'abord les étapes 01 → 03")

    print("\n\nModèles disponibles :")
    subprocess.run(["ollama", "list"])

    print("\nPour utiliser les modèles fine-tunés dans le débat,")
    print("modifiez WRITERS[...]['model'] dans main.py")
    print("  ex: 'hugo-writer' ou 'hugo-finetuned'")


if __name__ == "__main__":
    main()
