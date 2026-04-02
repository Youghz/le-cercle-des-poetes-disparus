# Le Cercle des Poètes Disparus

Un programme qui fait débattre trois grands écrivains français — **Victor Hugo**, **Voltaire** et **Jean-Jacques Rousseau** — sur le sujet de votre choix.

Chaque écrivain a sa propre personnalité et son propre style d'argumentation, fidèle à ses oeuvres réelles. Un modérateur anime les échanges, résume les arguments et relance le débat à chaque tour.

Tout tourne **en local** sur votre machine, sans envoyer de données sur internet.

## Comment ça marche, simplement

Imaginez une table ronde télévisée, mais avec des écrivains du passé :

1. Vous proposez un sujet (par exemple : *"Le progrès technique rend-il l'homme plus heureux ?"*)
2. **Victor Hugo** prend la parole en premier — passionné, lyrique, défenseur des opprimés
3. **Voltaire** répond — sarcastique, rationnel, mordant
4. **Jean-Jacques Rousseau** conclut le tour — introspectif, sensible, proche de la nature
5. Le **modérateur** résume et relance un nouveau tour
6. Et ainsi de suite, pendant le nombre de tours que vous choisissez

Les réponses de chaque écrivain sont générées par une intelligence artificielle (le modèle Mistral) qui tourne directement sur votre ordinateur.

## La stack technique

| Technologie | Rôle |
|---|---|
| **[LangGraph](https://github.com/langchain-ai/langgraph)** | Orchestre le débat : définit l'ordre de passage des écrivains, les tours, et la logique de fin. C'est le "chef d'orchestre" du programme. |
| **[LangChain](https://github.com/langchain-ai/langchain)** | Fournit l'interface pour communiquer avec le modèle d'IA : gère les messages, les rôles (système, utilisateur, assistant) et la connexion à Ollama. |
| **[Ollama](https://ollama.com/)** | Fait tourner le modèle d'IA en local sur votre machine. Aucune donnée n'est envoyée à un serveur externe. |
| **[Mistral](https://mistral.ai/)** | Le modèle d'IA utilisé. Mistral est un modèle open-source, performant et léger, qui fonctionne bien en local. |
| **[uv](https://github.com/astral-sh/uv)** | Gestionnaire de dépendances Python, rapide et moderne. Gère l'environnement virtuel du projet. |

## Structure du projet

```
le-cercle-des-poetes-disparus/
├── main.py                          # Le programme principal (le débat)
├── finetune/                        # Pipeline optionnel de fine-tuning
│   ├── 01_download_texts.py         # Télécharge les oeuvres libres de droit
│   ├── 02_prepare_dataset.py        # Prépare les données d'entraînement
│   ├── 03_finetune_mlx.py           # Entraîne le modèle sur les textes
│   └── 04_import_ollama.py          # Importe le modèle entraîné dans Ollama
└── pyproject.toml                   # Dépendances du projet
```

## Lancer le débat

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé
- [uv](https://github.com/astral-sh/uv) installé

### Installation

```bash
git clone https://github.com/Youghz/le-cercle-des-poetes-disparus.git
cd le-cercle-des-poetes-disparus
uv venv && source .venv/bin/activate
uv sync
ollama pull mistral
```

### Lancement

```bash
python main.py
```

Le programme vous demande un sujet et un nombre de tours, puis le débat commence.

## Aller plus loin : le fine-tuning

Par défaut, les écrivains sont simulés grâce à des descriptions de personnalité (system prompts). Pour des réponses encore plus fidèles à leur style, il est possible d'entraîner le modèle sur leurs oeuvres réelles.

Les textes utilisés proviennent de [Project Gutenberg](https://www.gutenberg.org/) et sont **libres de droit** (les trois auteurs sont morts avant 1900).

```bash
source .venv/bin/activate
uv add mlx-lm
python finetune/01_download_texts.py   # Télécharge les oeuvres
python finetune/02_prepare_dataset.py  # Prépare les données
python finetune/03_finetune_mlx.py     # Entraîne (nécessite un Mac Apple Silicon)
python finetune/04_import_ollama.py    # Importe dans Ollama
```

## Licence

Les textes littéraires utilisés sont dans le domaine public. Le code du projet est libre d'utilisation.
