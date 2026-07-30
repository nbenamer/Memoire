"""Tests unitaires de `build_corpus.collecter_corpus` (aucun reseau).

Le `JudilibreClient` est remplace par un `_ClientFactice` qui `yield`
une liste d'`Arret` deja construite : tout le flux (sauvegarde, index,
filtres) est testable hors ligne.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterator

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret, JudilibreClient  # noqa: E402
from src.acquisition.build_corpus import (  # noqa: E402
    COLONNES_INDEX,
    NOM_INDEX,
    collecter_corpus,
)


def _arret(
    id_: str,
    *,
    sommaire: str | None = None,
    annee: str = "2024",
    publication: list[str] | None = None,
) -> Arret:
    return Arret(
        id=id_,
        juridiction="cc",
        chambre="civ1",
        formation=None,
        numero_pourvoi="00-00.000",
        texte="TXT",
        sommaire=sommaire,
        date_decision=f"{annee}-01-15",
        publication=publication or [],
    )


class _ClientFactice:
    """Simulacre minimal : reproduit uniquement `iter_export`."""

    def __init__(self, arrets: list[Arret]) -> None:
        self._arrets = list(arrets)

    def iter_export(self, **_kwargs) -> Iterator[Arret]:   # signature laxe
        for arret in self._arrets:
            yield arret


def _lire_index(chemin: Path) -> list[dict[str, str]]:
    with chemin.open(encoding="utf-8", newline="") as fichier:
        return list(csv.DictReader(fichier))


def test_collecter_corpus_ecrit_fichiers_et_index(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    arrets = [
        _arret("a1", sommaire="som", publication=["b"], annee="2024"),
        _arret("a2", sommaire=None,  publication=["c"], annee="2020"),
    ]
    stats = collecter_corpus(
        _ClientFactice(arrets),   # type: ignore[arg-type]
        corpus,
        n_max=10,
        passe="a",
    )

    assert stats["sauves"] == 2
    assert stats["skippes"] == 0
    assert stats["filtres_sans_sommaire"] == 0
    assert (corpus / "a1.json").exists()
    assert (corpus / "a2.json").exists()

    index = _lire_index(corpus / NOM_INDEX)
    assert [ligne["id"] for ligne in index] == ["a1", "a2"]
    assert all(ligne["passe"] == "a" for ligne in index)
    assert index[0]["sommaire_present"] == "1"
    assert index[1]["sommaire_present"] == "0"
    assert index[0]["publication"] == "b"
    assert index[0]["annee"] == "2024"
    assert set(index[0].keys()) == set(COLONNES_INDEX)


def test_exiger_sommaire_filtre_les_arrets_sans_sommaire(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    arrets = [
        _arret("a1", sommaire=None),
        _arret("a2", sommaire="som"),
        _arret("a3", sommaire=None),
        _arret("a4", sommaire="som"),
    ]
    stats = collecter_corpus(
        _ClientFactice(arrets),   # type: ignore[arg-type]
        corpus,
        n_max=10,
        passe="a",
        exiger_sommaire=True,
    )

    assert stats["sauves"] == 2
    assert stats["filtres_sans_sommaire"] == 2
    # Le CSV d'index ne mentionne pas les arrets filtres en amont
    assert [l["id"] for l in _lire_index(corpus / NOM_INDEX)] == ["a2", "a4"]


def test_reprise_ne_reecrit_pas_et_ne_duplique_pas_index(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Simule un fichier deja present ET une entree deja dans l'index.
    JudilibreClient.sauvegarde_json(_arret("a1", sommaire="som"), corpus / "a1.json")
    (corpus / NOM_INDEX).write_text(
        "id,passe,annee,publication,sommaire_present\na1,a,2024,,1\n",
        encoding="utf-8",
    )

    stats = collecter_corpus(
        _ClientFactice([_arret("a1", sommaire="som"), _arret("a2", sommaire="som")]),   # type: ignore[arg-type]
        corpus,
        n_max=10,
        passe="b",   # nouvelle passe : ne doit PAS reindexer a1
    )

    assert stats["sauves"] == 1     # seul a2 est nouveau
    assert stats["skippes"] == 1
    ids = [l["id"] for l in _lire_index(corpus / NOM_INDEX)]
    assert ids == ["a1", "a2"]      # pas de doublon


def test_n_max_borne_le_nombre_de_sauvegardes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    arrets = [_arret(f"a{i}", sommaire="som") for i in range(20)]
    stats = collecter_corpus(
        _ClientFactice(arrets),   # type: ignore[arg-type]
        corpus,
        n_max=5,
        passe="a",
    )
    assert stats["sauves"] == 5
    fichiers_json = sorted(p.name for p in corpus.glob("a*.json"))
    assert fichiers_json == ["a0.json", "a1.json", "a2.json", "a3.json", "a4.json"]
