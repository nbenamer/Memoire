"""Maillon 3 : summarization (extractive + abstractive).

Modules :
  * `dataset`    : construit les paires supervisees et les splits geles.
  * `extractif`  : baseline extractive (aucun entrainement).
"""

from .dataset import (
    STRATEGIES,
    Exemple,
    construire_paires,
    construire_splits,
    mesurer_longueurs,
    sauvegarder_splits,
)
from .extractif import (
    FicheExtractive,
    evaluer_extractif,
    extraire_fiche,
)

__all__ = [
    "STRATEGIES",
    "Exemple",
    "construire_paires",
    "construire_splits",
    "mesurer_longueurs",
    "sauvegarder_splits",
    "FicheExtractive",
    "extraire_fiche",
    "evaluer_extractif",
]
