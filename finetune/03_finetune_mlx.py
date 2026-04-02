"""
Étape 3 : Fine-tuning avec MLX-LM (QLoRA) sur Apple Silicon.

Prérequis :
  pip install mlx-lm

Ce script lance le fine-tuning QLoRA pour chaque auteur,
puis fusionne les adapters avec le modèle de base.
"""

import subprocess
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
ADAPTERS_DIR = Path(__file__).parent / "adapters"
FUSED_DIR = Path(__file__).parent / "fused_models"

# Modèle de base compatible MLX — Mistral 7B Instruct
BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

AUTHORS = ["hugo", "voltaire", "rousseau"]

# Hyperparamètres QLoRA
LORA_CONFIG = {
    "iters": 200,          # nombre d'itérations (augmenter pour meilleur résultat)
    "batch_size": 2,        # adapté pour 24 GB RAM
    "learning_rate": 1e-5,
    "lora_layers": 8,       # nombre de couches LoRA
}


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*50}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERREUR: {desc} a échoué (code {result.returncode})")
        sys.exit(1)


def finetune_author(author: str):
    """Lance le fine-tuning QLoRA pour un auteur."""
    dataset_file = DATASETS_DIR / f"{author}_train.jsonl"
    if not dataset_file.exists():
        print(f"[SKIP] Dataset manquant : {dataset_file}")
        return

    adapter_dir = ADAPTERS_DIR / author
    adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--data", str(dataset_file),
        "--adapter-path", str(adapter_dir),
        "--iters", str(LORA_CONFIG["iters"]),
        "--batch-size", str(LORA_CONFIG["batch_size"]),
        "--learning-rate", str(LORA_CONFIG["learning_rate"]),
        "--num-layers", str(LORA_CONFIG["lora_layers"]),
    ]

    run_cmd(cmd, f"Fine-tuning QLoRA — {author}")


def fuse_author(author: str):
    """Fusionne l'adapter LoRA avec le modèle de base."""
    adapter_dir = ADAPTERS_DIR / author
    if not (adapter_dir / "adapters.safetensors").exists():
        print(f"[SKIP] Pas d'adapter pour {author}")
        return

    fused_dir = FUSED_DIR / author
    fused_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", BASE_MODEL,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ]

    run_cmd(cmd, f"Fusion adapter — {author}")


def main():
    print("=" * 60)
    print("  Fine-tuning QLoRA avec MLX-LM")
    print(f"  Modèle de base : {BASE_MODEL}")
    print(f"  Config : {LORA_CONFIG}")
    print("=" * 60)

    # Vérifier que mlx_lm est installé
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("\nmlx-lm n'est pas installé. Installation...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-lm"], check=True)

    for author in AUTHORS:
        finetune_author(author)

    print("\n" + "=" * 60)
    print("  Fusion des adapters")
    print("=" * 60)

    for author in AUTHORS:
        fuse_author(author)

    print("\n\nFine-tuning terminé !")
    print(f"Modèles fusionnés dans : {FUSED_DIR.resolve()}")
    print("\nProchaine étape : lancer 04_import_ollama.py pour importer dans Ollama")


if __name__ == "__main__":
    main()
