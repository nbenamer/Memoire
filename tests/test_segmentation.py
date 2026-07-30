"""Tests unitaires du maillon `segmentation`.

Aucun appel reseau : les tests factices construisent leurs propres
`Arret` en memoire ; le test integrant les arrets reels utilise les
fichiers deja persistes dans `data/echantillon/` (skip si absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import (  # noqa: E402
    ArretSegmente,
    ENONCE_MOYEN,
    FAITS_PROCEDURE,
    INDETERMINE,
    MOYENS,
    REPONSE_COUR,
    Segment,
    decouper_moyens,
    segmenter,
)


# ---------- Fabriques d'arrets factices ----------


def _arret_avec_zones(texte: str, zones: dict[str, list[tuple[int, int]]]) -> Arret:
    """Fabrique un `Arret` minimal avec un texte et des zones donnees."""
    zones_completes = {
        "introduction": [],
        "expose": [],
        "moyens": [],
        "motivations": [],
        "dispositif": [],
        "annexes": [],
    }
    zones_completes.update(zones)
    return Arret(
        id="test",
        juridiction="cc",
        chambre="civ1",
        formation=None,
        numero_pourvoi="00-00.000",
        texte=texte,
        zones=zones_completes,
    )


def _arret_sans_zones(texte: str) -> Arret:
    """Fabrique un `Arret` sans zones (chemin fallback)."""
    return _arret_avec_zones(texte, {})


# ---------- Chemin nominal : zones API disponibles ----------


def test_segmentation_api_extrait_les_bons_sous_textes() -> None:
    """Chaque zone API doit produire un segment avec le bon slicing."""
    texte = "AAAA_expose_BBBB_moyens_CCCC_motivations_DDDD_dispositif_EEEE"
    #        0         1         2         3         4         5
    #        0123456789012345678901234567890123456789012345678901234567890
    zones = {
        "expose":      (lambda t=texte: [(t.index("expose"), t.index("expose") + len("expose"))])(),
        "moyens":      (lambda t=texte: [(t.index("moyens"), t.index("moyens") + len("moyens"))])(),
        "motivations": (lambda t=texte: [(t.index("motivations"), t.index("motivations") + len("motivations"))])(),
        "dispositif":  (lambda t=texte: [(t.index("dispositif"), t.index("dispositif") + len("dispositif"))])(),
    }
    seg = segmenter(_arret_avec_zones(texte, zones))

    assert seg.origine == "api"
    assert seg.avertissements == []
    assert seg.faits == "expose"
    assert seg.moyens == "moyens"
    assert seg.motivations == "motivations"
    assert seg.dispositif == "dispositif"


def test_segmentation_api_fusionne_introduction_et_expose() -> None:
    """`introduction` et `expose` vont tous les deux dans `.faits`."""
    texte = "INTRO...EXPOSE...MOYENS..."
    zones = {
        "introduction": [(0, 7)],
        "expose":       [(8, 15)],
        "moyens":       [(16, 23)],
    }
    seg = segmenter(_arret_avec_zones(texte, zones))

    faits_segments = seg.par_zone(FAITS_PROCEDURE)
    assert [s.texte for s in faits_segments] == ["INTRO..", "EXPOSE."]
    assert seg.faits == "INTRO..\nEXPOSE."


def test_segmentation_api_gere_plusieurs_segments_par_zone() -> None:
    """Une zone peut contenir plusieurs segments (ordre par offset)."""
    texte = "A_moyen1_B_moyen2_C_dispositif"
    zones = {
        "moyens":     [(11, 17), (2, 8)],   # volontairement desordre
        "dispositif": [(20, 30)],
    }
    seg = segmenter(_arret_avec_zones(texte, zones))
    moyens_segs = seg.par_zone(MOYENS)
    assert [s.texte for s in moyens_segs] == ["moyen1", "moyen2"]
    assert seg.moyens == "moyen1\nmoyen2"


def test_segmentation_api_ignore_offsets_hors_bornes() -> None:
    """Les segments dont les offsets sont hors du texte sont ignores."""
    texte = "court"
    zones = {"moyens": [(0, 5), (10, 20)]}   # 10,20 hors bornes
    seg = segmenter(_arret_avec_zones(texte, zones))
    assert seg.moyens == "court"


# ---------- Sous-decoupage moyens (regex) ----------


TEXTE_MOYENS_MIXTE = (
    "Enonce du moyen\n"
    "corps enonce 1\n"
    "Reponse de la Cour\n"
    "corps reponse 1\n"
    "Enonce du moyen\n"
    "corps enonce 2\n"
    "Reponse de la Cour\n"
    "corps reponse 2\n"
)


def test_decouper_moyens_cas_nominal() -> None:
    """Alternance enonce/reponse => 4 sous-segments dans le bon ordre."""
    segments, avertissements = decouper_moyens(TEXTE_MOYENS_MIXTE)
    assert avertissements == []
    assert [s.nom_zone for s in segments] == [
        ENONCE_MOYEN, REPONSE_COUR, ENONCE_MOYEN, REPONSE_COUR,
    ]
    assert segments[0].texte.startswith("Enonce du moyen")
    assert segments[1].texte.startswith("Reponse de la Cour")


def test_decouper_moyens_offsets_absolus() -> None:
    """Les offsets sont exprimes dans le referentiel global fourni."""
    segments, _ = decouper_moyens(TEXTE_MOYENS_MIXTE, debut_absolu=1000)
    assert segments[0].debut == 1000
    assert segments[-1].fin == 1000 + len(TEXTE_MOYENS_MIXTE)


def test_decouper_moyens_un_seul_type_silencieux() -> None:
    """Bloc avec seulement des `Enonce du moyen` => pas decoupable, silencieux.

    C'est le cas nominal API post-2019 : les reponses sont dans une autre
    zone (`motivations`). On ne genere pas d'avertissement.
    """
    texte = "Enonce du moyen\ncorps enonce\n"
    segments, avertissements = decouper_moyens(texte)
    assert segments == []
    assert avertissements == []


def test_decouper_moyens_parite_cassee_signalee() -> None:
    """2 enonces + 1 reponse => bloc conserve + avertissement."""
    texte = (
        "Enonce du moyen\ncorps 1\n"
        "Reponse de la Cour\ncorps r1\n"
        "Enonce du moyen\ncorps 2\n"
    )
    segments, avertissements = decouper_moyens(texte)
    assert segments == []
    assert any("parite cassee" in a for a in avertissements)


def test_decouper_moyens_aucun_marqueur_silencieux() -> None:
    """Aucun marqueur => silencieux, pas d'avertissement bruit.

    Le sous-decoupage n'est qu'un raffinement : quand il ne peut rien
    faire, l'appelant conserve simplement le bloc `moyens` parent, sans
    generer d'alerte inutile (le cas API nominal produirait sinon un
    faux positif sur chaque arret).
    """
    texte = "moyen sans marqueur explicite"
    segments, avertissements = decouper_moyens(texte)
    assert segments == []
    assert avertissements == []


def test_decouper_moyens_bloc_vide() -> None:
    segments, avertissements = decouper_moyens("")
    assert segments == []
    assert avertissements == ["bloc moyens vide"]


# ---------- Fallback : arrets sans zones API ----------


def test_fallback_intitules_post_2019() -> None:
    """Intitules 'Faits et procedure' + 'Examen des moyens' + PAR CES MOTIFS."""
    texte = (
        "En-tete administratif\n"
        "1. Faits et procédure\n"
        "les faits\n"
        "2. Examen des moyens\n"
        "les moyens\n"
        "PAR CES MOTIFS\n"
        "REJETTE."
    )
    seg = segmenter(_arret_sans_zones(texte))
    assert seg.origine == "regles"
    assert "1. Faits et procédure" in seg.faits
    assert "2. Examen des moyens" in seg.moyens
    assert "PAR CES MOTIFS" in seg.dispositif


def test_fallback_intitules_sans_numerotation() -> None:
    """Version non numerotee des intitules doit aussi etre reconnue."""
    texte = (
        "En-tete\n"
        "Faits et procédure\n"
        "les faits\n"
        "Examen des moyens\n"
        "corps des moyens\n"
        "PAR CES MOTIFS\n"
        "CASSE."
    )
    seg = segmenter(_arret_sans_zones(texte))
    assert seg.origine == "regles"
    assert "les faits" in seg.faits
    assert "corps des moyens" in seg.moyens
    assert "CASSE." in seg.dispositif


def test_fallback_arret_ancien_bloc_unique() -> None:
    """Sans intitules mais avec PAR CES MOTIFS : bloc unique 'moyens' + avert."""
    texte = (
        "Attendu que la cour d'appel a retenu que...\n"
        "considerant que...\n"
        "PAR CES MOTIFS\n"
        "REJETTE le pourvoi."
    )
    seg = segmenter(_arret_sans_zones(texte))
    assert seg.origine == "regles"
    assert seg.faits == ""
    assert seg.moyens.startswith("Attendu que")
    assert seg.dispositif.startswith("PAR CES MOTIFS")
    assert any("non separables" in a for a in seg.avertissements)


def test_fallback_aucun_marqueur_indetermine() -> None:
    """Texte sans aucun intitule => segment unique 'indetermine'."""
    texte = "un texte quelconque sans aucun marqueur canonique"
    seg = segmenter(_arret_sans_zones(texte))
    assert seg.origine == INDETERMINE
    assert len(seg.segments) == 1
    assert seg.segments[0].nom_zone == INDETERMINE
    assert any("aucun marqueur" in a for a in seg.avertissements)


def test_texte_vide_donne_indetermine() -> None:
    seg = segmenter(_arret_sans_zones(""))
    assert seg.origine == INDETERMINE
    assert seg.segments == []
    assert seg.avertissements == ["texte de la decision vide"]


# ---------- Segment : invariants ----------


def test_segment_rejette_fin_avant_debut() -> None:
    with pytest.raises(ValueError):
        Segment(nom_zone="moyens", texte="", debut=10, fin=5, source="api")


# ---------- Tests sur les arrets reels persistes (skip si absents) ----------


@pytest.mark.parametrize(
    "chemin",
    sorted((RACINE / "data" / "echantillon").glob("*.json")),
    ids=lambda p: p.stem,
)
def test_segmentation_sur_arret_reel(chemin: Path) -> None:
    """Chaque arret reel doit produire un `ArretSegmente` coherent.

    Contrats verifies (sans supposer un chemin particulier) :
      * pas de crash ;
      * `origine` dans {api, regles, indetermine} ;
      * les offsets de chaque segment retombent bien sur `arret.texte` ;
      * la concatenation des zones ne depasse pas la taille du texte.
    """
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    arret = Arret.from_dict(donnees)
    seg = segmenter(arret)

    assert isinstance(seg, ArretSegmente)
    assert seg.origine in {"api", "regles", INDETERMINE}
    assert seg.arret_id == arret.id
    texte = arret.texte or ""
    for s in seg.segments:
        assert 0 <= s.debut <= s.fin <= len(texte)
        assert texte[s.debut: s.fin] == s.texte
    total = sum(s.fin - s.debut for s in seg.segments)
    assert total <= len(texte)


def test_segmentation_arret_2026_a_zones_api() -> None:
    """L'arret recent doit passer par le chemin nominal `api`."""
    chemin = RACINE / "data" / "echantillon" / "6a4de42193c619cd1f841229.json"
    if not chemin.exists():
        pytest.skip("arret recent non persiste")
    arret = Arret.from_dict(json.loads(chemin.read_text(encoding="utf-8")))
    seg = segmenter(arret)
    assert seg.origine == "api"
    assert seg.faits != ""
    assert seg.moyens != ""
    assert seg.motivations != ""
    assert seg.dispositif != ""
    assert seg.avertissements == []


def test_segmentation_arret_2014_tombe_en_fallback() -> None:
    """L'arret ancien n'a pas de zones API : fallback regles."""
    chemin = RACINE / "data" / "echantillon" / "6079c56a9ba5988459c57490.json"
    if not chemin.exists():
        pytest.skip("arret ancien non persiste")
    arret = Arret.from_dict(json.loads(chemin.read_text(encoding="utf-8")))
    seg = segmenter(arret)
    assert seg.origine == "regles"
    assert seg.dispositif.lstrip().startswith("PAR CES MOTIFS")
    assert len(seg.avertissements) >= 1
