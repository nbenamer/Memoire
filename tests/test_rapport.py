"""Test du rapport de segmentation sur arrets factices (0 I/O reseau)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation.rapport import (  # noqa: E402
    COLONNES_CSV,
    analyser_arret,
    generer_rapport,
)


def _arret(
    id_: str,
    texte: str,
    zones: dict[str, list[tuple[int, int]]] | None = None,
    sommaire: str | None = None,
    date_decision: str | None = "2024-01-15",
    chambre: str = "civ1",
    publication: list[str] | None = None,
) -> Arret:
    """Fabrique un `Arret` minimal (jamais de vrai texte de decision)."""
    zones_completes = {
        "introduction": [],
        "expose": [],
        "moyens": [],
        "motivations": [],
        "dispositif": [],
        "annexes": [],
    }
    if zones:
        zones_completes.update(zones)
    return Arret(
        id=id_,
        juridiction="cc",
        chambre=chambre,
        formation=None,
        numero_pourvoi="00-00.000",
        texte=texte,
        zones=zones_completes,
        sommaire=sommaire,
        date_decision=date_decision,
        publication=publication or [],
    )


def _persister(arret: Arret, dossier: Path) -> Path:
    chemin = dossier / f"{arret.id}.json"
    chemin.write_text(
        json.dumps(arret.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return chemin


def test_analyser_arret_calcule_les_flags() -> None:
    """Sur un arret avec zones factices, tous les flags doivent etre coherents."""
    texte = "AAAA_expose_BBBB_moyens_CCCC_dispositif_EEEE"
    zones = {
        "expose":     [(texte.index("expose"),     texte.index("expose") + 6)],
        "moyens":     [(texte.index("moyens"),     texte.index("moyens") + 6)],
        "dispositif": [(texte.index("dispositif"), texte.index("dispositif") + 10)],
    }
    stats = analyser_arret(_arret("a1", texte, zones, sommaire="ok", publication=["b"]))

    assert stats["arret_id"] == "a1"
    assert stats["origine"] == "api"
    assert stats["has_faits_procedure"] == 1
    assert stats["has_moyens"] == 1
    assert stats["has_motivations"] == 0
    assert stats["has_dispositif"] == 1
    assert stats["has_sommaire"] == 1
    assert stats["invariant_offsets_ok"] == 1
    assert stats["annee"] == 2024
    assert stats["publication"] == "b"
    assert stats["chambre"] == "civ1"


def test_generer_rapport_agrege_correctement(tmp_path: Path) -> None:
    """Corpus varie : verifie l'agregat produit et le CSV ecrit."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rapport_dir = tmp_path / "reports"

    # a1 : API + zones + sommaire, pub=b, 2024
    t1 = "AAAA_expose_BBBB_moyens_CCCC_dispositif_EEEE"
    z1 = {
        "expose":     [(t1.index("expose"),     t1.index("expose") + 6)],
        "moyens":     [(t1.index("moyens"),     t1.index("moyens") + 6)],
        "dispositif": [(t1.index("dispositif"), t1.index("dispositif") + 10)],
    }
    _persister(
        _arret("a1", t1, z1, sommaire="sommaire officiel court", publication=["b"]),
        corpus,
    )

    # a2 : fallback regles, pub=r, 2020, avec sommaire (long texte)
    t2 = (
        "En-tete\n"
        "1. Faits et procédure\n"
        + "faits" * 20 + "\n"
        "2. Examen des moyens\n"
        + "moyens" * 30 + "\n"
        "PAR CES MOTIFS\n"
        "REJETTE."
    )
    _persister(
        _arret(
            "a2", t2, zones=None,
            sommaire="som" * 10,
            date_decision="2020-06-01",
            chambre="soc",
            publication=["r"],
        ),
        corpus,
    )

    # a3 : indetermine, pas de sommaire, 1998, pub vide
    _persister(
        _arret(
            "a3", "texte quelconque sans marqueur", zones=None,
            sommaire=None, date_decision="1998-03-20", chambre="civ2",
            publication=None,
        ),
        corpus,
    )

    # a4 : API, sans sommaire (pas une paire exploitable), pub=b
    _persister(
        _arret(
            "a4", t1, z1, sommaire=None, date_decision="2024-05-01",
            chambre="civ1", publication=["b"],
        ),
        corpus,
    )

    stats = generer_rapport(corpus, rapport_dir, ecrire_csv=True)

    assert stats["n"] == 4
    assert stats["origines"] == {"api": 2, "regles": 1, "indetermine": 1}

    # 2 arrets avec sommaire -> 2 paires exploitables (a1, a2)
    assert stats["paires_exploitables"] == 2
    assert stats["paires_par_annee"] == {2024: 1, 2020: 1}
    assert stats["paires_par_chambre"] == {"civ1": 1, "soc": 1}

    # Taux compression : moyennes non nulles, valeurs raisonnables (0 < taux < 1).
    assert 0.0 < stats["moy_taux_compression"] < 1.0
    assert 0.0 < stats["med_taux_compression"] < 1.0

    # Taux de sommaire par publication : b={1/2}, r={1/1}, ""={0/1}
    par_pub = dict((cle, (avec, total)) for cle, avec, total, _ in stats["taux_sommaire_par_publication"])
    assert par_pub["b"] == (1, 2)
    assert par_pub["r"] == (1, 1)
    assert par_pub["(vide)"] == (0, 1)

    # CSV : bonnes colonnes, aucune fuite de texte
    chemin_csv = Path(stats["chemin_csv"])
    assert chemin_csv.exists()
    with chemin_csv.open(encoding="utf-8", newline="") as f:
        lecteur = csv.DictReader(f)
        assert lecteur.fieldnames == COLONNES_CSV
        lignes = list(lecteur)
    assert {ligne["arret_id"] for ligne in lignes} == {"a1", "a2", "a3", "a4"}
    # Tous les champs numeriques doivent etre des entiers, sauf annee qui
    # peut etre vide. Les champs textuels autorises sont arret_id, chambre,
    # publication, origine.
    champs_texte_ok = {"arret_id", "chambre", "publication", "origine"}
    for ligne in lignes:
        for colonne, valeur in ligne.items():
            if colonne in champs_texte_ok or valeur == "":
                continue
            assert valeur.isdigit(), (
                f"colonne {colonne} = {valeur!r} n'est pas un entier"
            )


def test_generer_rapport_corpus_vide(tmp_path: Path) -> None:
    corpus_vide = tmp_path / "vide"
    corpus_vide.mkdir()
    stats = generer_rapport(corpus_vide, tmp_path / "reports", ecrire_csv=True)
    assert stats["n"] == 0
    assert stats["paires_exploitables"] == 0
    assert "chemin_csv" not in stats


def test_generer_rapport_ignore_json_illisible_et_index(tmp_path: Path) -> None:
    """Le rapport doit ignorer `_index.csv` et les fichiers JSON corrompus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corrompu.json").write_text("{not: json", encoding="utf-8")
    (corpus / "_index.csv").write_text("id,passe\n", encoding="utf-8")
    _persister(_arret("a1", "texte quelconque"), corpus)

    stats = generer_rapport(corpus, tmp_path / "reports", ecrire_csv=False)
    assert stats["n"] == 1
    assert stats["fichiers_illisibles"] == ["corrompu.json"]
