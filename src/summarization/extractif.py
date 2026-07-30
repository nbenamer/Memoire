"""Baseline extractive (aucun entrainement) pour le maillon 3.

Produit une `FicheExtractive` a partir d'un `ArretSegmente`, sans jamais
inventer de contenu :
  * `faits_procedure` : les N phrases les plus saillantes (score TF-IDF)
                        de la zone faits_procedure, dans l'ordre d'origine.
  * `solution`        : sens du dispositif (arret.sens_solution ou premier
                        mot du dispositif) + phrase de principe la plus
                        saillante des motivations.
  * `probleme_de_droit` : NON produit (mention explicite : le probleme
                        de droit n'est ecrit nulle part, il faut
                        l'abstractif du maillon 3 pour le reformuler).

Cette baseline sert de point de comparaison ROUGE contre l'abstractif :
on veut savoir de combien BARThez ameliore un simple extractif.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import ArretSegmente, segmenter  # noqa: E402

logger = logging.getLogger("summarization.extractif")

MENTION_PROBLEME_DE_DROIT = (
    "[non produit par l'extractif : le probleme de droit doit etre "
    "reformule par l'abstractif du maillon 3]"
)

# Segmentation phrases : regex simple. Suffisant pour la baseline ;
# on ne coupe pas sur les abreviations en '.' isolees car les phrases
# juridiques commencent presque toujours par une majuscule.
_RE_SEP_PHRASES = re.compile(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂÊÎÔÛÇ])")

# Amorces de "phrase de principe" observees en pratique dans les
# motivations de la Cour de cassation.
_MOTS_PRINCIPE = (
    "aux termes",
    "il resulte",
    "il résulte",
    "la cour",
    "vu ",
    "attendu que",
    "selon",
)


def _phrases(texte: str, longueur_min: int = 15) -> list[str]:
    """Decoupe en phrases (filtre les fragments trop courts, non phrase)."""
    if not texte:
        return []
    brutes = _RE_SEP_PHRASES.split(texte.strip())
    phrases = [p.strip() for p in brutes]
    return [p for p in phrases if len(p) >= longueur_min]


def _selectionner_top_phrases(
    texte: str,
    n: int,
) -> list[str]:
    """N phrases les plus saillantes selon TF-IDF, dans l'ordre d'origine.

    Score = norme TF-IDF de la phrase (approximation simple de la
    "densite d'information" par rapport au vocabulaire du document).
    """
    phrases = _phrases(texte)
    if not phrases:
        return []
    if len(phrases) <= n:
        return phrases

    # Import local : sklearn seulement quand appele.
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True, min_df=1)
    matrice = vec.fit_transform(phrases)
    # Score = somme des poids TF-IDF par phrase (equivalent a la norme L1).
    scores = matrice.sum(axis=1)
    scores_liste = [float(scores[i, 0]) for i in range(matrice.shape[0])]
    # Selection des indices des top-N puis tri par position d'origine.
    indices_top = sorted(
        sorted(range(len(phrases)), key=lambda i: -scores_liste[i])[:n]
    )
    return [phrases[i] for i in indices_top]


def _phrase_principe(motivations: str) -> str:
    """Selectionne la phrase de principe des motivations.

    Heuristique :
      1. Chercher la premiere phrase qui commence par un des marqueurs
         `_MOTS_PRINCIPE` -> reflet fidele du raisonnement de la Cour.
      2. Sinon, la phrase la plus saillante (TF-IDF).
    """
    phrases = _phrases(motivations)
    if not phrases:
        return ""
    for phrase in phrases:
        debut = phrase.lstrip().lower()
        if any(debut.startswith(marqueur) for marqueur in _MOTS_PRINCIPE):
            return phrase
    top = _selectionner_top_phrases(motivations, n=1)
    return top[0] if top else phrases[0]


def _sens_dispositif(arret: Arret, segmente: ArretSegmente) -> str:
    """Sens du dispositif : `arret.sens_solution` si disponible, sinon
    premier mot significatif du dispositif (CASSE, REJETTE, ANNULE, ...)."""
    if arret.sens_solution:
        return arret.sens_solution.strip()
    dispositif = segmente.dispositif.strip()
    if not dispositif:
        return ""
    # Le dispositif commence en general par "PAR CES MOTIFS,\nCASSE ..."
    for ligne in dispositif.splitlines():
        ligne = ligne.strip(" ,.;:")
        if not ligne or ligne.upper().startswith("PAR CES MOTIFS"):
            continue
        # Premier mot en majuscules significatif
        mots = ligne.split()
        if mots and mots[0].isupper() and len(mots[0]) >= 4:
            return mots[0]
        return ligne[:80]
    return ""


@dataclass
class FicheExtractive:
    """Fiche produite sans invention, tracable a des phrases sources."""

    id: str
    faits_procedure: str
    solution: str
    probleme_de_droit: str = MENTION_PROBLEME_DE_DROIT
    avertissements: list[str] = field(default_factory=list)

    def partie_juridique(self) -> str:
        """Concatenation faits + solution : la partie comparable au sommaire.

        On exclut la mention `MENTION_PROBLEME_DE_DROIT` du ROUGE pour ne
        pas biaiser la mesure (le baseline s'abstient volontairement).
        """
        morceaux = [m for m in (self.faits_procedure, self.solution) if m]
        return "\n".join(morceaux)


def extraire_fiche(
    arret_segmente: ArretSegmente,
    arret: Arret,
    *,
    n_phrases_faits: int = 3,
) -> FicheExtractive:
    """Genere une `FicheExtractive` a partir d'un arret segmente.

    Le probleme de droit n'est PAS produit : on ne l'invente pas.
    """
    avertissements: list[str] = []

    faits = arret_segmente.faits
    phrases_faits = _selectionner_top_phrases(faits, n=n_phrases_faits)
    faits_extraits = " ".join(phrases_faits)
    if not faits_extraits:
        avertissements.append("faits_procedure indisponible")

    principe = _phrase_principe(arret_segmente.motivations)
    sens = _sens_dispositif(arret, arret_segmente)
    parties_solution = [p for p in (sens, principe) if p]
    solution = " — ".join(parties_solution)
    if not solution:
        avertissements.append("solution indisponible")

    return FicheExtractive(
        id=arret.id,
        faits_procedure=faits_extraits,
        solution=solution,
        avertissements=avertissements,
    )


# ----------------------------------------------------------------------
# Evaluation ROUGE contre le sommaire officiel
# ----------------------------------------------------------------------


def _charger_arret(chemin: Path) -> Arret:
    return Arret.from_dict(json.loads(chemin.read_text(encoding="utf-8")))


def _rouge_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )


def evaluer_extractif(
    dossier_corpus: Path,
    ids: Iterable[str],
    *,
    n_phrases_faits: int = 3,
) -> dict[str, object]:
    """Evalue la baseline sur un sous-ensemble d'ids (typiquement le val).

    Retourne les scores ROUGE-1/2/L (F1 moyenne + mediane) et les
    compteurs d'exemples ignores. Ne persiste ni ne loggue aucun texte.
    """
    dossier_corpus = Path(dossier_corpus)
    scorer = _rouge_scorer()

    scores: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
    n_ok = 0
    n_sans_sommaire = 0
    n_fiche_vide = 0
    n_manquants = 0

    for id_ in ids:
        chemin = dossier_corpus / f"{id_}.json"
        if not chemin.exists():
            n_manquants += 1
            continue
        arret = _charger_arret(chemin)
        cible = (arret.sommaire or "").strip()
        if not cible:
            n_sans_sommaire += 1
            continue
        segmente = segmenter(arret)
        fiche = extraire_fiche(segmente, arret, n_phrases_faits=n_phrases_faits)
        prediction = fiche.partie_juridique()
        if not prediction:
            n_fiche_vide += 1
            continue
        resultats = scorer.score(cible, prediction)
        for metrique, valeur in resultats.items():
            scores[metrique].append(valeur.fmeasure)
        n_ok += 1

    def _agreger(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"moyenne": 0.0, "mediane": 0.0}
        return {
            "moyenne": statistics.mean(vals),
            "mediane": statistics.median(vals),
        }

    return {
        "n_ok": n_ok,
        "n_manquants": n_manquants,
        "n_sans_sommaire": n_sans_sommaire,
        "n_fiche_vide": n_fiche_vide,
        "rouge1": _agreger(scores["rouge1"]),
        "rouge2": _agreger(scores["rouge2"]),
        "rougeL": _agreger(scores["rougeL"]),
    }


def _lire_split(chemin: Path) -> list[str]:
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    return list(donnees.get("ids", []))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluation ROUGE de la baseline extractive sur un split."
    )
    p.add_argument("--dossier-corpus", type=Path, default=RACINE / "data" / "corpus")
    p.add_argument(
        "--split",
        type=Path,
        default=RACINE / "data" / "splits" / "val.json",
        help="Fichier de split (defaut: data/splits/val.json).",
    )
    p.add_argument("--n-phrases-faits", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.split.exists():
        logger.error("Split introuvable : %s", args.split)
        return 2
    ids = _lire_split(args.split)
    logger.info("Evaluation de la baseline extractive sur %d ids (%s)",
                len(ids), args.split.name)
    scores = evaluer_extractif(
        args.dossier_corpus, ids, n_phrases_faits=args.n_phrases_faits,
    )
    print()
    print(f"Baseline extractive - split : {args.split.name}")
    print(f"  n exemples scores      : {scores['n_ok']}")
    print(f"  n sans sommaire        : {scores['n_sans_sommaire']}")
    print(f"  n fiche extraite vide  : {scores['n_fiche_vide']}")
    print(f"  n ids manquants        : {scores['n_manquants']}")
    print()
    for metrique in ("rouge1", "rouge2", "rougeL"):
        agg = scores[metrique]
        print(f"  {metrique:<7} F1  : moy={agg['moyenne']:.4f}   med={agg['mediane']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
