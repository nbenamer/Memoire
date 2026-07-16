from pathlib import Path

FILES = {
    "requirements.txt": """requests
python-dotenv
transformers
torch
sentence-transformers
faiss-cpu
scikit-learn
rouge-score
pandas
numpy
""",
    ".gitignore": """venv/
.venv/
__pycache__/
*.pyc
data/
.env
.DS_Store
.ipynb_checkpoints/
""",
    ".cursorrules": """Projet : generateur de fiches d'arret a partir de decisions de la Cour de cassation (Judilibre).
Langue : code et commentaires en francais.
Stack : Python 3.11, HuggingFace (CamemBERT/JuriBERT, BARThez), FAISS, sentence-transformers.
Architecture : pipeline modulaire (acquisition -> segmentation -> resume -> retrieval -> evaluation).
Contraintes :
- Ne jamais logguer ni commiter de donnees de decisions (RGPD, pseudonymisation).
- Privilegier des fonctions courtes, testees, documentees.
- Toujours expliquer les choix techniques en une phrase.
""",
    "README.md": """# Memoire - Generateur de fiches d'arret

Prototype de generation automatique de fiches d'arret a partir des decisions
de la Cour de cassation (open data Judilibre).

## Pipeline
1. Acquisition (API Judilibre)
2. Segmentation (regles + modele)
3. Resume (extractif + abstractif)
4. Recherche semantique (embeddings + FAISS)
5. Evaluation (ROUGE + grille juridique)

## Installation
    python -m venv venv
    # Windows : venv\\Scripts\\activate
    # macOS/Linux : source venv/bin/activate
    pip install -r requirements.txt
""",
    "src/pipeline.py": '''"""Orchestration de la chaine de traitement d'un arret vers une fiche."""


def generer_fiche(arret_id: str) -> dict:
    """D'un identifiant d'arret vers une fiche structuree."""
    # 1. acquisition
    # 2. segmentation
    # 3. resume
    # 4. retrieval / ancrage
    # 5. evaluation
    raise NotImplementedError


if __name__ == "__main__":
    print("Pipeline a implementer.")
''',
}

PACKAGES = ["src", "src/acquisition", "src/segmentation",
            "src/summarization", "src/retrieval", "src/evaluation"]
EMPTY_DIRS = ["data", "notebooks", "tests"]

for pkg in PACKAGES:
    Path(pkg).mkdir(parents=True, exist_ok=True)
    (Path(pkg) / "__init__.py").touch()

for d in EMPTY_DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / ".gitkeep").touch()

for path, content in FILES.items():
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("Structure du projet creee.")