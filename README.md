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

Ajouter un fichier `.env` a la racine (jamais commite) contenant :

    JUDILIBRE_CLIENT_ID=...
    JUDILIBRE_CLIENT_SECRET=...
    JUDILIBRE_BASE_URL=https://api.piste.gouv.fr/cassation/judilibre/v1.0
    JUDILIBRE_OAUTH_TOKEN_URL=https://oauth.piste.gouv.fr/api/oauth/token

## Construction du corpus

Le maillon 3 (resume) a besoin de deux corpus complementaires stockes tous les
deux dans `data/corpus/` (git-ignored). Chaque arret est un fichier
`<id>.json`, plus un index `_index.csv` qui trace l'origine
(passe, annee, publication, presence du sommaire) sans jamais persister le
texte de la decision.

### Passe A : "supervisee" (paires arret -> sommaire officiel)

But : constituer un corpus d'apprentissage pour l'abstractif.
Cible : au moins 2000 arrets avec sommaire non vide.

    venv/bin/python -m src.acquisition.build_corpus \
        --passe a \
        --publication b r \
        --exiger-sommaire \
        -n 2000

`publication in {b, r}` = bulletin ou rapport annuel : cette selection
concentre les arrets pour lesquels la Cour publie un sommaire officiel.

### Passe B : "representative" (evaluation en usage reel)

But : mesurer le comportement du pipeline sur un tirage tout-venant
(arrets avec ou sans sommaire).

    venv/bin/python -m src.acquisition.build_corpus \
        --passe b \
        -n 500

Les deux passes ecrivent dans le meme dossier `data/corpus/` ; l'index
`data/corpus/_index.csv` conserve la trace de la passe qui a decouvert
chaque id (aucune ligne en double).

### Reprise

La reprise est intrinseque : les fichiers deja sur disque ne sont pas
reecrits. On peut donc relancer la meme commande apres une interruption,
ou monter progressivement `-n` sur plusieurs sessions.

## Analyse du corpus

    venv/bin/python -m src.segmentation.rapport

Produit un CSV horodate dans `reports/` (une ligne par arret, uniquement
des metadonnees publiques et des compteurs -- jamais de texte) et un
resume console incluant :

* la repartition par origine de segmentation (`api` / `regles` / `indetermine`),
* la presence de chaque zone de fiche et sa longueur moyenne,
* le nombre de paires `(arret, sommaire)` exploitables et leur repartition
  par annee et par chambre,
* la longueur moyenne du texte et du sommaire, et le taux de compression
  moyen (indispensable pour calibrer les hyperparametres du maillon 3),
* le taux de sommaire officiel par valeur de `publication`.

## Split gele pour l'entrainement (maillon 3)

Le split est reconstruit sur les paires exploitables selon la strategie
d'entree du modele. Il est fige dans `data/splits/` (ids uniquement,
`seed=42`, stratification par `chambre x decennie` regroupee).

Note importante sur les strategies :

* les mesures de l'etape A ont fait ressortir 1999 paires exploitables
  pour la strategie `motivations+expose` et 1963 pour la strategie
  `motivations` (36 arrets ont un expose non vide sans motivations
  reconnues) ;
* l'entrainement (etape C) utilise `motivations` : le split est donc
  regele sur ce sous-ensemble a 1963 paires (train=1570 / val=196 /
  test=197). Aucun modele n'a ete entraine sur le split precedent :
  ce regelage ne fait pas fuiter d'information.

Pour regenerer les splits (memes seed et stratification) :

    venv/bin/python -m src.summarization.dataset --strategie-split motivations

## Investigation d'un arret

    venv/bin/python scripts/inspecter_arret.py <id_arret>
    venv/bin/python scripts/inspecter_arret.py --indetermine

Affiche a l'ecran (sans persister ni logger) les metadonnees publiques
et un apercu de 200 caracteres, pour identifier manuellement les arrets
qui sortent du perimetre (avis, ordonnances, etc.) et decider s'ils
doivent etre ecartes du corpus d'apprentissage.
