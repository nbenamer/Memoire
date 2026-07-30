"""Sous-decoupage du bloc `moyens` en (enonce du moyen, reponse de la Cour).

Judilibre delimite le bloc `moyens` mais ne separe PAS a l'interieur les
enonces (arguments du demandeur au pourvoi) des reponses (motivation
propre de la Cour). Or c'est cette distinction qui interesse la fiche
d'arret. Ce module fait cette separation par regles.

Principe :
  * on cherche toutes les positions des intitules "Enonce du moyen" et
    "Reponse de la Cour" (post-2019, intitules normalises) ;
  * on decoupe le texte a chaque marker : chaque segment porte le nom
    du marker qui le precede ;
  * on signale explicitement les ambiguites (markers consecutifs de meme
    type, parite cassee entre enonces et reponses) plutot que de deviner
    (regle .cursorrules : abstention plutot qu'invention).

Contrat : quand `avertissements` est non vide, l'appelant doit conserver
le bloc `moyens` d'origine et ignorer les sous-segments.
"""

from __future__ import annotations

import re

from .models import ENONCE_MOYEN, REPONSE_COUR, Segment

# `re.IGNORECASE` couvre "Enonce/Enoncé/ENONCE/etc.". Le `\S*` permet
# d'accepter un suffixe eventuel (numero ou complement) sur la meme ligne
# sans le rattacher au corps du segment (le marker sert de frontiere).
_RE_ENONCE = re.compile(
    r"^\s*[EÉée]nonc[eé]\s+du\s+moyen\b.*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_REPONSE = re.compile(
    r"^\s*R[eé]ponse\s+de\s+la\s+Cour\b.*$",
    re.MULTILINE | re.IGNORECASE,
)


def decouper_moyens(
    texte_moyens: str,
    debut_absolu: int = 0,
) -> tuple[list[Segment], list[str]]:
    """Decoupe le bloc `moyens` en sous-segments enonce / reponse.

    Retourne `(segments, avertissements)`. `segments` est vide (et
    `avertissements` non vide) quand le decoupage est jugee non fiable.
    Les offsets des `Segment` retournes sont exprimes dans le referentiel
    d'origine (`arret.texte`), via `debut_absolu`.
    """
    if not texte_moyens:
        return [], ["bloc moyens vide"]

    marqueurs: list[tuple[int, str]] = []
    for m in _RE_ENONCE.finditer(texte_moyens):
        marqueurs.append((m.start(), ENONCE_MOYEN))
    for m in _RE_REPONSE.finditer(texte_moyens):
        marqueurs.append((m.start(), REPONSE_COUR))

    # Silencieux quand rien n'est trouve : c'est le cas nominal API
    # (les zones sont deja separees en amont, il n'y a pas de decoupage
    # supplementaire a faire). Les vrais avertissements ne sortent que
    # pour des cas semantiquement problematiques (parite cassee, etc.).
    if not marqueurs:
        return [], []

    marqueurs.sort(key=lambda x: x[0])
    n_enonce = sum(1 for _, t in marqueurs if t == ENONCE_MOYEN)
    n_reponse = sum(1 for _, t in marqueurs if t == REPONSE_COUR)

    # Cas nominal API post-2019 : un seul type de marker est present dans
    # le bloc (l'API a deja separe enonces dans `moyens` et reponses dans
    # `motivations`). Rien a faire ici, silence => l'appelant conserve le
    # bloc parent tel quel sans avertissement.
    if n_enonce == 0 or n_reponse == 0:
        return [], []

    avertissements: list[str] = []
    for i in range(1, len(marqueurs)):
        if marqueurs[i][1] == marqueurs[i - 1][1]:
            avertissements.append(
                f"deux marqueurs '{marqueurs[i][1]}' consecutifs "
                f"aux offsets {marqueurs[i-1][0]} et {marqueurs[i][0]}"
            )
    if n_enonce != n_reponse:
        avertissements.append(
            f"parite cassee : {n_enonce} enonce(s) vs {n_reponse} reponse(s)"
        )
    if avertissements:
        return [], avertissements

    segments: list[Segment] = []
    for i, (pos, nom) in enumerate(marqueurs):
        fin = marqueurs[i + 1][0] if i + 1 < len(marqueurs) else len(texte_moyens)
        segments.append(
            Segment(
                nom_zone=nom,
                texte=texte_moyens[pos:fin],
                debut=debut_absolu + pos,
                fin=debut_absolu + fin,
                source="regles",
            )
        )
    return segments, []
