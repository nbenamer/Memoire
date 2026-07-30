"""CLI d'acquisition d'un corpus d'arrets depuis Judilibre.

Persiste un fichier JSON par arret dans `data/corpus/<id>.json` et
tient a jour un index `data/corpus/_index.csv` (metadonnees publiques
uniquement : id, passe, annee, publication, sommaire_present).

Deux passes complementaires sont supportees :

* passe "a" (supervisee) : filtree sur `publication in {b, r}` avec
  `--exiger-sommaire` -> corpus d'apprentissage pour le maillon 3
  (paires arret -> sommaire officiel) ;
* passe "b" (representative) : sans filtre -> corpus d'evaluation
  (comportement en usage reel, incluant les arrets sans sommaire).

La reprise est intrinseque : les fichiers deja sur disque ne sont pas
reecrits, et l'index n'est jamais duplique sur un meme id.

Contraintes RGPD : aucun texte de decision n'est ecrit dans les logs
ni dans l'index (seulement des metadonnees publiques et des compteurs).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret, JudilibreClient, JudilibreError  # noqa: E402

logger = logging.getLogger("build_corpus")

NOM_INDEX = "_index.csv"
COLONNES_INDEX = ["id", "passe", "annee", "publication", "sommaire_present"]


def _annee(arret: Arret) -> str:
    """Annee sur 4 chiffres depuis `arret.date_decision`, ou chaine vide."""
    if not arret.date_decision:
        return ""
    valeur = arret.date_decision[:4]
    return valeur if valeur.isdigit() else ""


def _publication_serialisee(arret: Arret) -> str:
    """Serialise `arret.publication` (liste) en une chaine `a|b|c`."""
    if not arret.publication:
        return ""
    return "|".join(str(p) for p in arret.publication)


def _charger_ids_index(chemin: Path) -> set[str]:
    if not chemin.exists():
        return set()
    with chemin.open(encoding="utf-8", newline="") as fichier:
        return {ligne["id"] for ligne in csv.DictReader(fichier) if ligne.get("id")}


def _ajouter_a_index(chemin: Path, ligne: dict[str, str]) -> None:
    """Ajout append-only ; ecrit l'entete si le fichier n'existait pas."""
    existait = chemin.exists()
    with chemin.open("a", encoding="utf-8", newline="") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=COLONNES_INDEX)
        if not existait:
            writer.writeheader()
        writer.writerow(ligne)


def collecter_corpus(
    client: JudilibreClient,
    dossier: Path,
    n_max: int,
    passe: str,
    *,
    exiger_sommaire: bool = False,
    date_start: str | None = None,
    date_end: str | None = None,
    chamber: list[str] | None = None,
    publication: list[str] | None = None,
    type_decision: list[str] | None = None,
    batch_size: int = 100,
    pause_sec: float = 0.2,
) -> dict[str, int]:
    """Sauvegarde jusqu'a `n_max` nouveaux arrets et met a jour l'index.

    Retourne `{recus, sauves, skippes, filtres_sans_sommaire, erreurs}`.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    index_path = dossier / NOM_INDEX
    ids_indexes = _charger_ids_index(index_path)

    stats = {
        "recus": 0,
        "sauves": 0,
        "skippes": 0,
        "filtres_sans_sommaire": 0,
        "erreurs": 0,
    }

    for arret in client.iter_export(
        date_start=date_start,
        date_end=date_end,
        chamber=chamber,
        publication=publication,
        type_decision=type_decision,
        batch_size=batch_size,
        pause_sec=pause_sec,
        limite=None,
    ):
        stats["recus"] += 1
        a_sommaire = bool(arret.sommaire)

        if exiger_sommaire and not a_sommaire:
            stats["filtres_sans_sommaire"] += 1
            continue

        destination = dossier / f"{arret.id}.json"
        if destination.exists():
            stats["skippes"] += 1
        else:
            try:
                JudilibreClient.sauvegarde_json(arret, destination)
                stats["sauves"] += 1
            except OSError as exc:
                stats["erreurs"] += 1
                logger.warning("Ecriture en echec pour %s : %s", arret.id, exc)
                continue
            if stats["sauves"] % 25 == 0 and stats["sauves"] > 0:
                logger.info(
                    "Progression : sauves=%d skippes=%d filtres=%d recus=%d",
                    stats["sauves"],
                    stats["skippes"],
                    stats["filtres_sans_sommaire"],
                    stats["recus"],
                )

        if arret.id not in ids_indexes:
            _ajouter_a_index(
                index_path,
                {
                    "id": arret.id,
                    "passe": passe,
                    "annee": _annee(arret),
                    "publication": _publication_serialisee(arret),
                    "sommaire_present": "1" if a_sommaire else "0",
                },
            )
            ids_indexes.add(arret.id)

        if stats["sauves"] >= n_max:
            break
    return stats


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recupere un corpus d'arrets Cour de cassation via Judilibre.",
    )
    p.add_argument(
        "--passe",
        required=True,
        choices=["a", "b"],
        help="Identifiant de passe (a: supervisee/sommaire, b: representative).",
    )
    p.add_argument("-n", type=int, default=300, help="Nombre d'arrets nouveaux vises.")
    p.add_argument(
        "--exiger-sommaire",
        action="store_true",
        help="Ne compte et ne sauvegarde que les arrets ayant un sommaire non vide.",
    )
    p.add_argument(
        "--dossier",
        type=Path,
        default=RACINE / "data" / "corpus",
        help="Dossier de sortie (defaut: data/corpus).",
    )
    p.add_argument("--date-start", default=None, help="Date minimale (YYYY-MM-DD).")
    p.add_argument("--date-end", default=None, help="Date maximale (YYYY-MM-DD).")
    p.add_argument("--chamber", nargs="+", default=None, help="Filtre chambres.")
    p.add_argument(
        "--publication",
        nargs="+",
        default=None,
        help="Filtre publication (cles b/r/l/c). Passe A recommandee : b r.",
    )
    p.add_argument("--type", dest="type_decision", nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--pause-ms", type=int, default=200)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "Collecte passe=%s n=%d exiger_sommaire=%s dossier=%s "
        "date=[%s..%s] chamber=%s publication=%s type=%s",
        args.passe, args.n, args.exiger_sommaire, args.dossier,
        args.date_start, args.date_end,
        args.chamber, args.publication, args.type_decision,
    )
    try:
        client = JudilibreClient()
    except RuntimeError as exc:
        logger.error("Config .env invalide : %s", exc)
        return 2

    try:
        stats = collecter_corpus(
            client,
            args.dossier,
            args.n,
            passe=args.passe,
            exiger_sommaire=args.exiger_sommaire,
            date_start=args.date_start,
            date_end=args.date_end,
            chamber=args.chamber,
            publication=args.publication,
            type_decision=args.type_decision,
            batch_size=args.batch_size,
            pause_sec=args.pause_ms / 1000.0,
        )
    except JudilibreError as exc:
        logger.error("Echec acquisition : %s", exc)
        return 1
    logger.info("Termine : %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
