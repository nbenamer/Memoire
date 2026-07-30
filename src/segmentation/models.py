"""Modele de donnees du maillon `segmentation`.

Deux dataclasses seulement :
  * `Segment`         : une portion de texte tracable (nom canonique, offsets
                        absolus dans `arret.texte`, source de production).
  * `ArretSegmente`   : agregation ordonnee de `Segment`, avec accesseurs
                        pratiques pour les composantes de la fiche d'arret
                        (`.faits`, `.moyens`, `.motivations`, `.dispositif`,
                        `.annexes`, `.enonces_moyens`, `.reponses_cour`).

Noms de zones canoniques (utilises par les maillons aval) :
  * "faits_procedure"  : introduction + expose (zones API fusionnees)
  * "moyens"           : bloc moyens (haut niveau), garde tel quel si le
                         sous-decoupage echoue
  * "enonce_moyen"     : sous-segment issu du decoupage regex de `moyens`
  * "reponse_cour"     : sous-segment issu du decoupage regex de `moyens`
  * "motivations"      : motivations de la Cour
  * "dispositif"       : dispositif final
  * "annexes"          : moyens annexes
  * "indetermine"      : fallback total (aucun marqueur reconnu)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

FAITS_PROCEDURE = "faits_procedure"
MOYENS = "moyens"
ENONCE_MOYEN = "enonce_moyen"
REPONSE_COUR = "reponse_cour"
MOTIVATIONS = "motivations"
DISPOSITIF = "dispositif"
ANNEXES = "annexes"
INDETERMINE = "indetermine"


@dataclass(frozen=True)
class Segment:
    """Portion de texte extraite avec ses offsets absolus dans `arret.texte`."""

    nom_zone: str
    texte: str
    debut: int
    fin: int
    source: str  # "api" | "regles" | "fallback"

    def __post_init__(self) -> None:
        if self.fin < self.debut:
            raise ValueError(
                f"Segment invalide : fin={self.fin} < debut={self.debut}"
            )


@dataclass
class ArretSegmente:
    """Segmentation d'un `Arret` en composantes de la fiche.

    `origine` decrit d'ou viennent les segments haut niveau :
      * "api"          : zones fournies par Judilibre (chemin nominal)
      * "regles"       : detection par intitules regex (arrets anciens)
      * "indetermine"  : aucun marqueur reconnu, texte livre en bloc
    Les `avertissements` sont non fatals mais doivent remonter cote fiche
    finale (RGPD/fiabilite : signaler l'incertitude plutot que d'inventer).
    """

    arret_id: str
    segments: list[Segment]
    origine: str
    avertissements: list[str] = field(default_factory=list)

    def par_zone(self, nom_zone: str) -> list[Segment]:
        """Retourne les segments d'une zone donnee, tries par offset."""
        return sorted(
            (s for s in self.segments if s.nom_zone == nom_zone),
            key=lambda s: s.debut,
        )

    def _concat(self, nom_zone: str) -> str:
        return "\n".join(s.texte for s in self.par_zone(nom_zone))

    @property
    def faits(self) -> str:
        """Faits + procedure (introduction + expose fusionnes)."""
        return self._concat(FAITS_PROCEDURE)

    @property
    def moyens(self) -> str:
        """Bloc moyens : renvoie le sous-decoupage si disponible, sinon le bloc brut."""
        sous = [
            s
            for s in self.segments
            if s.nom_zone in (ENONCE_MOYEN, REPONSE_COUR)
        ]
        if sous:
            return "\n".join(s.texte for s in sorted(sous, key=lambda s: s.debut))
        return self._concat(MOYENS)

    @property
    def motivations(self) -> str:
        return self._concat(MOTIVATIONS)

    @property
    def dispositif(self) -> str:
        return self._concat(DISPOSITIF)

    @property
    def annexes(self) -> str:
        return self._concat(ANNEXES)

    @property
    def enonces_moyens(self) -> list[str]:
        """Liste des textes de chaque enonce de moyen (peut etre vide)."""
        return [s.texte for s in self.par_zone(ENONCE_MOYEN)]

    @property
    def reponses_cour(self) -> list[str]:
        """Liste des textes de chaque reponse de la Cour (peut etre vide)."""
        return [s.texte for s in self.par_zone(REPONSE_COUR)]

    def a_du_sous_decoupage_moyens(self) -> bool:
        """Vrai si `decouper_moyens` a produit des sous-segments exploitables."""
        return any(
            s.nom_zone in (ENONCE_MOYEN, REPONSE_COUR) for s in self.segments
        )
