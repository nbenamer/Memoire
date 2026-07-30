"""Modele de donnees interne pour une decision judiciaire.

La dataclass `Arret` est la traduction francaise du schema `decisionFull`
de l'API JUDILIBRE (Swagger 2.0). Elle isole le reste du pipeline des
noms de champs et des conventions cote API : le mapping se fait en un
seul endroit (voir `judilibre._mapper_arret`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Arret:
    """Decision de justice normalisee (unite de travail du pipeline).

    Les champs marques *Optional* peuvent etre absents de la reponse API
    (arrets anciens, decisions sans sommaire, absence de zonage, etc.).
    """

    id: str
    juridiction: str
    chambre: str
    formation: str | None
    numero_pourvoi: str
    numeros: list[str] = field(default_factory=list)
    ecli: str | None = None
    date_decision: str | None = None                     # ISO-8601 court
    type_decision: str | None = None                     # arret, qpc, ...
    sens_solution: str | None = None                     # clef normalisee
    solution_libelle: str | None = None                  # `solution_alt`
    texte: str | None = None                             # texte pseudonymise
    sommaire: str | None = None                          # sommaire officiel
    themes: list[str] = field(default_factory=list)
    publication: list[str] = field(default_factory=list)
    visa: list[dict[str, Any]] = field(default_factory=list)
    zones: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    partielle: bool | None = None
    interet_particulier: bool | None = None
    decision_attaquee: dict[str, Any] | None = None      # `contested`

    def to_dict(self) -> dict[str, Any]:
        """Serialisation dict prete pour `json.dumps` (tuples -> listes)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, donnees: dict[str, Any]) -> "Arret":
        """Reconstruit un `Arret` depuis un dict (JSON deserialise).

        Symetrique de `to_dict` : les zones sont serialisees en listes en
        JSON, on les reconvertit en tuples pour rester coherent avec le
        mapping produit par le client.
        """
        zones_brut = donnees.get("zones") or {}
        zones = {
            nom: [(int(a), int(b)) for a, b in (segs or [])]
            for nom, segs in zones_brut.items()
        }
        return cls(
            id=donnees["id"],
            juridiction=donnees.get("juridiction") or "",
            chambre=donnees.get("chambre") or "",
            formation=donnees.get("formation"),
            numero_pourvoi=donnees.get("numero_pourvoi") or "",
            numeros=list(donnees.get("numeros") or []),
            ecli=donnees.get("ecli"),
            date_decision=donnees.get("date_decision"),
            type_decision=donnees.get("type_decision"),
            sens_solution=donnees.get("sens_solution"),
            solution_libelle=donnees.get("solution_libelle"),
            texte=donnees.get("texte"),
            sommaire=donnees.get("sommaire"),
            themes=list(donnees.get("themes") or []),
            publication=list(donnees.get("publication") or []),
            visa=list(donnees.get("visa") or []),
            zones=zones,
            partielle=donnees.get("partielle"),
            interet_particulier=donnees.get("interet_particulier"),
            decision_attaquee=donnees.get("decision_attaquee"),
        )
