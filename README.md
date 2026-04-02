# Le Cercle des Poetes Disparus

Un programme qui fait debattre trois grands ecrivains francais -- **Victor Hugo**, **Voltaire** et **Jean-Jacques Rousseau** -- sur le sujet de votre choix.

Chaque ecrivain a sa propre personnalite et son propre style d'argumentation, fidele a ses oeuvres reelles. Un moderateur anime les echanges, resume les arguments et relance le debat a chaque tour.

Tout tourne **en local** sur votre machine, sans envoyer de donnees sur internet.

## Comment ca marche, simplement

Imaginez une table ronde televisee, mais avec des ecrivains du passe :

1. Vous proposez un sujet (par exemple : *"L'intelligence artificielle va-t-elle remplacer les artistes ?"*)
2. **Victor Hugo** prend la parole en premier -- passionne, lyrique, defenseur des opprimes
3. **Voltaire** repond -- sarcastique, rationnel, mordant
4. **Jean-Jacques Rousseau** conclut le tour -- introspectif, sensible, proche de la nature
5. Le **moderateur** resume et relance un nouveau tour
6. Et ainsi de suite, pendant le nombre de tours que vous choisissez

Les reponses de chaque ecrivain sont generees par une intelligence artificielle (le modele Mistral 7B) qui a ete **fine-tunee sur les oeuvres reelles** de chaque auteur, et qui tourne directement sur votre ordinateur.

## La stack technique

| Technologie | Role |
|---|---|
| **[LangGraph](https://github.com/langchain-ai/langgraph)** | Orchestre le debat : definit l'ordre de passage des ecrivains, les tours, et la logique de fin. C'est le "chef d'orchestre" du programme. |
| **[LangChain](https://github.com/langchain-ai/langchain)** | Fournit l'interface pour communiquer avec le modele d'IA : gere les messages, les roles (systeme, utilisateur, assistant) et la connexion a Ollama. |
| **[Ollama](https://ollama.com/)** | Fait tourner le modele d'IA en local sur votre machine. Aucune donnee n'est envoyee a un serveur externe. |
| **[Mistral 7B](https://mistral.ai/)** | Le modele d'IA de base, open-source et performant. Fine-tune individuellement pour chaque ecrivain. |
| **[MLX](https://github.com/ml-explore/mlx)** | Framework Apple pour le fine-tuning QLoRA sur Apple Silicon. Utilise la memoire unifiee des puces M. |
| **[llama.cpp](https://github.com/ggml-org/llama.cpp)** | Conversion des modeles fine-tunes au format GGUF, le format natif d'Ollama. |
| **[uv](https://github.com/astral-sh/uv)** | Gestionnaire de dependances Python, rapide et moderne. |

## La chaine de traitement complete

Voici tout le parcours des donnees, du texte brut jusqu'au debat :

```
Oeuvres libres de droit (Project Gutenberg)
    |
    |  01_download_texts.py
    |  Telecharge ~7 250 pages de textes en francais
    v
Textes bruts (.txt)
    Hugo: Les Miserables (3 tomes), Notre-Dame de Paris, Les Contemplations
    Voltaire: Candide, Dictionnaire philosophique, Lettres philosophiques, Zadig
    Rousseau: Du contrat social, Les Confessions, Emile
    |
    |  02_prepare_dataset.py
    |  Nettoyage Gutenberg, decoupage en passages, format chat JSONL
    v
Datasets d'entrainement (train.jsonl par auteur)
    Hugo: 3 539 exemples
    Voltaire: 1 140 exemples
    Rousseau: 3 251 exemples
    |
    |  03_finetune_mlx.py
    |  QLoRA 4-bit via MLX-LM, 200 iterations, ~8 min/auteur sur M4 Pro
    v
Adapters LoRA (safetensors MLX)
    Hugo: loss 2.806 -> 1.980
    Voltaire: loss initiale -> 2.021
    Rousseau: loss initiale -> 2.107
    |
    |  mlx_lm fuse --dequantize
    |  Fusion adapter + modele de base, dequantization float16
    v
Modeles HuggingFace complets (safetensors float16, ~14 GB chacun)
    |
    |  llama.cpp convert_hf_to_gguf.py
    |  Conversion au format GGUF (format natif d'Ollama)
    v
Fichiers GGUF (1 fichier par auteur, ~14 GB)
    |
    |  ollama create
    |  Import dans Ollama avec Modelfile
    v
Modeles disponibles dans Ollama
    hugo-finetuned, voltaire-finetuned, rousseau-finetuned
    |
    |  main.py (LangGraph + LangChain + ChatOllama)
    v
Debat entre ecrivains
```

### Pourquoi autant d'etapes ?

Trois formats de modeles coexistent, chacun optimise pour un usage different :

| Format | Produit par | Utilise par | Pourquoi |
|---|---|---|---|
| **MLX safetensors** | Fine-tuning MLX | Apple Silicon uniquement | Optimise pour la memoire unifiee des puces M |
| **HuggingFace safetensors** | Fusion + dequantization | Ecosysteme ML standard | Format pivot universel, noms de tenseurs standards |
| **GGUF** | llama.cpp convert | Ollama, llama.cpp | Format binaire unique, supporte la quantization, performant pour l'inference |

La conversion `MLX -> HuggingFace -> GGUF` est necessaire parce que :
- MLX utilise ses propres noms de tenseurs + des tenseurs de quantization (`biases`, `scales`) que GGUF ne comprend pas
- Il faut d'abord dequantizer (4-bit -> float16) pour obtenir des poids standards
- Puis convertir les noms de tenseurs HuggingFace vers le schema GGUF

## Structure du projet

```
le-cercle-des-poetes-disparus/
├── main.py                          # Le programme principal (le debat)
├── finetune/
│   ├── 01_download_texts.py         # Telecharge les oeuvres depuis Gutenberg
│   ├── 02_prepare_dataset.py        # Nettoie et prepare les JSONL
│   ├── 03_finetune_mlx.py           # Fine-tuning QLoRA avec MLX-LM
│   ├── 04_import_ollama.py          # Cree les Modelfiles et importe
│   ├── raw_texts/                   # Textes telecharges (genere)
│   ├── datasets/                    # JSONL par auteur (genere)
│   ├── adapters/                    # Adapters LoRA (genere)
│   ├── hf_models/                   # Modeles HF dequantizes (genere)
│   └── gguf/                        # Fichiers GGUF finaux (genere)
└── pyproject.toml
```

## Lancer le debat

### Prerequis

- Python 3.10+
- [Ollama](https://ollama.com/) installe
- [uv](https://github.com/astral-sh/uv) installe
- Mac Apple Silicon (pour le fine-tuning uniquement)

### Installation rapide (sans fine-tuning)

```bash
git clone https://github.com/Youghz/le-cercle-des-poetes-disparus.git
cd le-cercle-des-poetes-disparus
uv venv && source .venv/bin/activate
uv sync
ollama pull mistral
python main.py
```

### Installation complete (avec fine-tuning)

```bash
git clone https://github.com/Youghz/le-cercle-des-poetes-disparus.git
cd le-cercle-des-poetes-disparus
uv venv && source .venv/bin/activate
uv sync
uv add mlx-lm torch gguf

# 1. Telecharger les oeuvres (~30 sec)
python finetune/01_download_texts.py

# 2. Preparer les datasets (~5 sec)
python finetune/02_prepare_dataset.py

# 3. Fine-tuning QLoRA (~25 min pour les 3 auteurs sur M4 Pro)
python finetune/03_finetune_mlx.py

# 4. Conversion MLX -> HuggingFace (dequantize)
for author in hugo voltaire rousseau; do
  python -m mlx_lm fuse \
    --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
    --adapter-path finetune/adapters/$author \
    --save-path finetune/hf_models/$author \
    --dequantize
done

# 5. Conversion HuggingFace -> GGUF
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
for author in hugo voltaire rousseau; do
  python /tmp/llama.cpp/convert_hf_to_gguf.py \
    finetune/hf_models/$author \
    --outfile finetune/gguf/${author}-f16.gguf \
    --outtype f16
done

# 6. Import dans Ollama
for author in hugo voltaire rousseau; do
  echo "FROM $(pwd)/finetune/gguf/${author}-f16.gguf" > /tmp/Modelfile.$author
  ollama create ${author}-finetuned -f /tmp/Modelfile.$author
done

# 7. Lancer le debat
python main.py
```

## Exemple de conversation

> Sujet : **L'intelligence artificielle va-t-elle remplacer les artistes ?** (2 tours)

---

**[Victor Hugo]**

Il est vrai que l'intelligence artificielle s'avere plus puissante qu'on ne le pensait un temps. Nous sommes temoins de son developpement rapide et de sa capacite a realiser des choses qui n'auraient peut-etre pas ete imaginables dix ans auparavant. Cependant, il nous faudrait un peu d'impatience pour penser que cette technologie puisse remplacer l'artiste humain, authentique et unique. Il faut comprendre qu'il s'agit de deux choses radicalement differentes. L'humain cree, il a des sentiments, des idees originales, des emotions qui se refletent dans son oeuvre. Cela ne peut etre reproduit ou simule par un outil. Alors que l'intelligence artificielle est une machine, elle n'a pas d'emotion ni de sentiments. Enfin, il nous faut garder a l'esprit que l'art n'a pas seulement une valeur materielle. Il a aussi un bien plus grand role dans notre societe. Il sert a exprimer les sentiments, les idees et la vision du monde de l'humain. Cela se perd dans la mecanique d'une intelligence artificielle.

---

**[Voltaire]**

Bien sur, je suis content de vous repondre. Bien que les machines soient devenues plus compliquees et les algorithmes plus sophistiques, il est impitoyable d'imaginer qu'ils puissent jamais remplir le role des artistes. En tant que philosophe, j'ai toujours pense que l'art n'etait pas uniquement une chose qui s'exprimait par les yeux et les oreilles; il etait aussi un moyen de communiquer notre vie interieure, notre passions, nos emotions. Il y a des choses que les machines ne peuvent jamais comprendre, et qui constituent la base de l'art. L'humain est unique par son ame, sa pensee, ses idees et ses sentiments. Il n'est pas possible que cette singularite soit reproduite par un algorithme programme dans une machine.

---

**[Jean-Jacques Rousseau]**

L'art de l'etre humain, qui a ete cree pour tous ses sens, n'est pas fait pour l'imitation, mais pour l'expression. La machine manque d'un sens -- le toucher -- ce qui signifie qu'on ne peut pas dire qu'elle possede une sensibilite humaine complete; de plus, sa memoire est limitee et finie, contrairement a la memoire de l'homme qui a ete dote du pouvoir de se souvenir des choses passees aussi bien que presentes.

---

**[Moderateur]**

Les trois philosophes ont souligne l'importance de la singularite humaine dans l'art, arguant que les machines ne peuvent jamais egaler la creativite et l'emotion d'un etre humain. Le debat se poursuit sur la facon dont l'intelligence artificielle peut influencer l'art humain.

---

**[Victor Hugo]** *(tour 2)*

Il n'en faut pas plus, en effet. Il est evident que les machines ne peuvent jamais remplacer les createurs humains. Ils manquent d'emotion et de passion. La machine n'a qu'un langage logique et rationnel qui n'est pas suffisant pour exprimer les plus grands sentiments de l'homme.

---

**[Jean-Jacques Rousseau]** *(tour 2)*

Cet art est cree a l'image de son createur. On ne peut pas dire que cette image puisse etre exprimee sans le toucher de l'artiste. Quand une machine fait de la musique, il lui manque l'emotion qui doit preceder la melodie; quand elle ecrit une histoire, il lui manque le gout, le jugement et le style de l'etre humain qui peut juger comment utiliser au mieux son art pour le rendre plus agreable.

---

**[Moderateur]**

Les philosophes ont souligne l'importance de la singularite humaine dans l'art, arguant que les machines ne peuvent jamais egaler la creativite et l'emotion d'un etre humain.

## Sources des textes

Tous les textes utilises pour le fine-tuning sont dans le **domaine public** (auteurs morts avant 1900) et proviennent de [Project Gutenberg](https://www.gutenberg.org/) :

| Auteur | Oeuvres | Source |
|---|---|---|
| Victor Hugo | Les Miserables (3 tomes), Notre-Dame de Paris, Les Contemplations | Gutenberg #17489-17491, #19657, #54615 |
| Voltaire | Candide, Dictionnaire philosophique, Lettres philosophiques, Zadig | Gutenberg #4650, #18569, #28858, #4647 |
| J.-J. Rousseau | Du contrat social, Les Confessions, Emile | Gutenberg #23684, #3913, #5427 |

## Licence

Les textes litteraires utilises sont dans le domaine public. Le code du projet est libre d'utilisation.
