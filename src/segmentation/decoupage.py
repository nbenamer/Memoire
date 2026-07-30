"""Segmentation d'un `Arret` en composantes de la fiche.

Chemin nominal : les zones fournies par l'API Judilibre sont converties
en `Segment` (chacune fusionnee vers un nom canonique de la fiche). Un
sous-decoupage regex tente ensuite de separer les enonces de moyen des
reponses de la Cour a l'interieur du bloc `moyens`.

Fallback (arrets anciens sans zones API) : detection par regles sur les
intitules normalises post-2019 (numerotation "1. Faits et procedure",
"2. Examen des moyens") et sur "PAR CES MOTIFS" comme frontiere du
dispositif. Si aucun marqueur n'est reconnu, on livre le texte en un
seul segment `indetermine` avec avertissement (regle .cursorrules :
abstention plutot qu'invention).
"""

from __future__ import annotations

import logging
import re

from ..acquisition import Arret
from .models import (
    ArretSegmente,
    DISPOSITIF,
    FAITS_PROCEDURE,
    INDETERMINE,
    MOYENS,
    Segment,
)
from .reponse_cour import decouper_moyens

logger = logging.getLogger(__name__)

# Mapping des noms de zones API vers les noms canoniques du pipeline.
# `introduction` et `expose` sont fusionnees dans `faits_procedure`
# (l'en-tete administratif est peu couteux et le maillon summarization
# extractif saura le filtrer si besoin).
NOMS_ZONE_API_VERS_CANONIQUE: dict[str, str] = {
    "introduction": FAITS_PROCEDURE,
    "expose": FAITS_PROCEDURE,
    "moyens": MOYENS,
    "motivations": "motivations",
    "dispositif": DISPOSITIF,
    "annexes": "annexes",
}

# --- Regex de fallback (arrets anciens ou zones API absentes) ---
_RE_PAR_CES_MOTIFS = re.compile(r"^\s*PAR\s+CES\s+MOTIFS\b", re.MULTILINE)
_RE_FAITS_PROCEDURE = re.compile(
    r"^\s*(?:\d+\.\s+)?Faits?\s+et\s+proc[eé]dure\b",
    re.MULTILINE | re.IGNORECASE,
)
_RE_EXAMEN_MOYENS = re.compile(
    r"^\s*(?:\d+\.\s+)?Examen\s+(du|des)\s+moyens?\b",
    re.MULTILINE | re.IGNORECASE,
)


def segmenter(arret: Arret) -> ArretSegmente:
    """Point d'entree du maillon. Aucune I/O reseau."""
    texte = arret.texte or ""
    if not texte:
        return ArretSegmente(
            arret_id=arret.id,
            segments=[],
            origine=INDETERMINE,
            avertissements=["texte de la decision vide"],
        )

    avertissements: list[str] = []
    if _zones_api_exploitables(arret):
        segments = _segmenter_par_api(arret)
        origine = "api"
    else:
        segments, avert_regles = _segmenter_par_regles(texte)
        avertissements.extend(avert_regles)
        if segments:
            origine = "regles"
        else:
            segments = [
                Segment(
                    nom_zone=INDETERMINE,
                    texte=texte,
                    debut=0,
                    fin=len(texte),
                    source="fallback",
                )
            ]
            origine = INDETERMINE
            avertissements.append(
                "aucun marqueur reconnu, texte livre en bloc 'indetermine'"
            )

    segments = _affiner_moyens(segments, avertissements)
    return ArretSegmente(
        arret_id=arret.id,
        segments=segments,
        origine=origine,
        avertissements=avertissements,
    )


def _zones_api_exploitables(arret: Arret) -> bool:
    """Vrai si au moins une zone API contient un segment valide."""
    zones = arret.zones or {}
    return any(zones.get(nom) for nom in NOMS_ZONE_API_VERS_CANONIQUE)


def _segmenter_par_api(arret: Arret) -> list[Segment]:
    """Convertit les offsets API en `Segment` canoniques (tries)."""
    texte = arret.texte or ""
    segments: list[Segment] = []
    for nom_api, sous_segments in (arret.zones or {}).items():
        nom_canonique = NOMS_ZONE_API_VERS_CANONIQUE.get(nom_api)
        if nom_canonique is None or not sous_segments:
            continue
        for debut, fin in sous_segments:
            # L'API renvoie parfois `fin == len(texte) + 1` (off-by-one
            # systematique cote Judilibre) : on clampe silencieusement,
            # `str[a:b]` tolere de toute facon un `b` hors bornes.
            if fin == len(texte) + 1:
                fin = len(texte)
            if debut < 0 or fin > len(texte) or fin < debut:
                logger.warning(
                    "Segment API hors bornes ignore : zone=%s [%d,%d] len=%d",
                    nom_api,
                    debut,
                    fin,
                    len(texte),
                )
                continue
            segments.append(
                Segment(
                    nom_zone=nom_canonique,
                    texte=texte[debut:fin],
                    debut=debut,
                    fin=fin,
                    source="api",
                )
            )
    segments.sort(key=lambda s: s.debut)
    return segments


def _segmenter_par_regles(
    texte: str,
) -> tuple[list[Segment], list[str]]:
    """Fallback pour arrets sans zones API.

    Trois niveaux, du plus au moins fiable :
      1. Intitules post-2019 numerotes + "PAR CES MOTIFS" => decoupage
         faits_procedure / moyens / dispositif.
      2. "PAR CES MOTIFS" seul => bloc unique 'moyens' avant, dispositif
         apres, avertissement sur la non-separation des composantes.
      3. Rien => liste vide (l'appelant produira 'indetermine').
    """
    segments: list[Segment] = []
    avertissements: list[str] = []

    m_par_ces = _RE_PAR_CES_MOTIFS.search(texte)
    m_faits = _RE_FAITS_PROCEDURE.search(texte)
    m_examen = _RE_EXAMEN_MOYENS.search(texte)
    borne_dispositif = m_par_ces.start() if m_par_ces else len(texte)

    if m_faits and m_examen and m_examen.start() > m_faits.start():
        segments.append(
            Segment(
                nom_zone=FAITS_PROCEDURE,
                texte=texte[m_faits.start(): m_examen.start()],
                debut=m_faits.start(),
                fin=m_examen.start(),
                source="regles",
            )
        )
        segments.append(
            Segment(
                nom_zone=MOYENS,
                texte=texte[m_examen.start(): borne_dispositif],
                debut=m_examen.start(),
                fin=borne_dispositif,
                source="regles",
            )
        )
    elif m_par_ces:
        segments.append(
            Segment(
                nom_zone=MOYENS,
                texte=texte[:borne_dispositif],
                debut=0,
                fin=borne_dispositif,
                source="regles",
            )
        )
        avertissements.append(
            "arret ancien : faits/procedure/motivations non separables, "
            "bloc unique 'moyens' avant PAR CES MOTIFS"
        )
    else:
        return [], []

    if m_par_ces:
        segments.append(
            Segment(
                nom_zone=DISPOSITIF,
                texte=texte[borne_dispositif:],
                debut=borne_dispositif,
                fin=len(texte),
                source="regles",
            )
        )
    return segments, avertissements


def _affiner_moyens(
    segments: list[Segment],
    avertissements: list[str],
) -> list[Segment]:
    """Remplace les segments `moyens` par leur sous-decoupage si fiable.

    En cas d'ambiguite, on conserve le bloc `moyens` d'origine et on
    reporte l'avertissement (fiabilite > couverture).
    """
    resultat: list[Segment] = []
    for seg in segments:
        if seg.nom_zone != MOYENS:
            resultat.append(seg)
            continue
        sous_segments, sous_avert = decouper_moyens(seg.texte, debut_absolu=seg.debut)
        if sous_segments and not sous_avert:
            resultat.extend(sous_segments)
        else:
            resultat.append(seg)
            if sous_avert:
                avertissements.extend(sous_avert)
    return resultat
