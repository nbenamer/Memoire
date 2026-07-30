"""Tests offline de la baseline extractive."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import segmenter  # noqa: E402
from src.summarization.extractif import (  # noqa: E402
    MENTION_PROBLEME_DE_DROIT,
    evaluer_extractif,
    extraire_fiche,
    _phrase_principe,
    _phrases,
    _selectionner_top_phrases,
)


def _arret(
    id_: str,
    texte: str,
    zones: dict[str, list[tuple[int, int]]] | None = None,
    sommaire: str | None = None,
    sens_solution: str | None = None,
) -> Arret:
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
        chambre="civ1",
        formation=None,
        numero_pourvoi="00-00.000",
        texte=texte,
        zones=zones_completes,
        sommaire=sommaire,
        sens_solution=sens_solution,
        date_decision="2024-05-01",
    )


def test_phrases_filtre_les_fragments_courts() -> None:
    texte = "Une phrase suffisamment longue. Ok. Une seconde phrase reelle."
    p = _phrases(texte)
    assert len(p) == 2   # "Ok." trop court
    assert "Ok" not in p[0] and "Ok" not in p[1]


def test_selectionner_top_phrases_ordre_original() -> None:
    texte = (
        "Le contrat est signe entre les parties. "
        "La clause de non-concurrence est ambigue. "
        "Un simple detail. "
        "Le juge apprecie souverainement la duree."
    )
    top = _selectionner_top_phrases(texte, n=2)
    assert len(top) == 2
    # Verifie l'ordre d'origine (les phrases apparaissent dans l'ordre du texte)
    positions = [texte.index(p) for p in top]
    assert positions == sorted(positions)


def test_phrase_principe_prefere_les_amorces() -> None:
    motivations = (
        "Le pourvoi conteste la decision. "
        "Vu l'article 1240 du code civil, aux termes duquel tout fait de "
        "l'homme oblige a reparation. "
        "Le moyen manque en fait."
    )
    p = _phrase_principe(motivations)
    assert p.startswith("Vu l'article")


def test_extraire_fiche_faits_solution_probleme_de_droit() -> None:
    texte = (
        "Introduction generale. "
        "Le contrat de vente est conclu le 12 mars 2019 entre les parties. "
        "Un litige nait sur la garantie des vices caches. "
        "La cour d'appel a rejete la demande du vendeur. "
        "Enonce du moyen "
        "Le pourvoi fait grief a l'arret. "
        "Reponse de la Cour "
        "Vu l'article 1641 du code civil ; "
        "il resulte de ce texte que la garantie s'applique. "
        "PAR CES MOTIFS, "
        "CASSE ET ANNULE l'arret rendu."
    )
    # zones API : on renseigne "expose" (faits) et "motivations"
    debut_expose = texte.index("Le contrat")
    fin_expose = texte.index("Enonce du moyen") - 1
    debut_mot = texte.index("Vu l'article")
    fin_mot = texte.index("PAR CES MOTIFS")
    debut_disp = texte.index("PAR CES MOTIFS")
    fin_disp = len(texte)
    zones = {
        "expose":      [(debut_expose, fin_expose)],
        "motivations": [(debut_mot, fin_mot)],
        "dispositif":  [(debut_disp, fin_disp)],
    }
    arret = _arret("A1", texte, zones, sommaire="som", sens_solution="Cassation")
    fiche = extraire_fiche(segmenter(arret), arret, n_phrases_faits=2)

    assert fiche.id == "A1"
    # Les faits doivent contenir au moins une phrase reelle (aucune invention).
    assert "contrat" in fiche.faits_procedure.lower() \
        or "vices caches" in fiche.faits_procedure.lower()
    # La solution doit contenir le sens et un morceau de la phrase de principe.
    assert "Cassation" in fiche.solution
    assert "1641" in fiche.solution or "garantie" in fiche.solution.lower()
    # Le probleme de droit reste explicitement non produit.
    assert fiche.probleme_de_droit == MENTION_PROBLEME_DE_DROIT
    # La partie juridique concatene faits + solution sans invention.
    assert MENTION_PROBLEME_DE_DROIT not in fiche.partie_juridique()


def test_evaluer_extractif_ignore_ids_manquants_et_sans_sommaire(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    texte = (
        "Introduction. "
        "Le litige oppose une societe a son ancien salarie sur un solde. "
        "La cour d'appel a condamne l'employeur au paiement. "
        "Vu l'article L. 3242-1 du code du travail ; "
        "il resulte de ce texte que le salaire doit etre paye a echeance. "
        "PAR CES MOTIFS, REJETTE le pourvoi."
    )
    debut_expose = texte.index("Le litige")
    fin_expose = texte.index("Vu l'article") - 1
    debut_mot = texte.index("Vu l'article")
    fin_mot = texte.index("PAR CES MOTIFS")
    zones = {
        "expose":      [(debut_expose, fin_expose)],
        "motivations": [(debut_mot, fin_mot)],
        "dispositif":  [(texte.index("PAR CES MOTIFS"), len(texte))],
    }
    arret_ok = _arret("A", texte, zones,
                      sommaire="Le salaire doit etre paye a echeance.",
                      sens_solution="Rejet")
    (corpus / "A.json").write_text(json.dumps(arret_ok.to_dict()), encoding="utf-8")

    arret_sans_som = _arret("B", texte, zones, sommaire=None)
    (corpus / "B.json").write_text(json.dumps(arret_sans_som.to_dict()), encoding="utf-8")

    scores = evaluer_extractif(corpus, ["A", "B", "MANQUANT"], n_phrases_faits=2)
    assert scores["n_ok"] == 1
    assert scores["n_sans_sommaire"] == 1
    assert scores["n_manquants"] == 1
    # ROUGE renvoie des flottants entre 0 et 1 (F1)
    for metrique in ("rouge1", "rouge2", "rougeL"):
        agg = scores[metrique]
        assert 0.0 <= agg["moyenne"] <= 1.0
