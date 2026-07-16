# Memoire - Generateur de fiches d'arret

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
    # Windows : venv\Scripts\activate
    # macOS/Linux : source venv/bin/activate
    pip install -r requirements.txt
