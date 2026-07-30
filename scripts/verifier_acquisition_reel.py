"""Test manuel bout-en-bout du maillon `acquisition`.

Enchaine :
  1. `healthcheck`  -> API disponible ?
  2. `rechercher(query="licenciement")`  -> premier id de resultat.
  3. `get_decision(id)`                  -> mapping vers `Arret`.
  4. `sauvegarde_json`                   -> `data/echantillon/<id>.json`.
  5. Impression d'un resume metier (sans jamais afficher le texte).

Aucun texte de decision n'est logge ni imprime : seuls les metadonnees
et un booleen "sommaire present" sortent sur stdout.

Usage :
    venv/bin/python scripts/verifier_acquisition_reel.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret, JudilibreClient, JudilibreError  # noqa: E402


DESTINATION = RACINE / "data" / "echantillon"


def _longueur_zone(segments: list[tuple[int, int]]) -> int:
    """Longueur cumulee (en caracteres) des segments d'une zone."""
    return sum(fin - debut for debut, fin in segments)


def _afficher_resume(arret: Arret) -> None:
    """Impression conforme RGPD : jamais le texte, jamais le sommaire brut."""
    print("--- Resume metier ---")
    print(f"  id             : {arret.id}")
    print(f"  formation      : {arret.formation!r}")
    print(f"  date_decision  : {arret.date_decision}")
    print(f"  sens_solution  : {arret.sens_solution!r}")
    print(f"  sommaire ?     : {'oui' if arret.sommaire else 'non'}")
    print("  zones (longueur en caracteres) :")
    for nom, segments in arret.zones.items():
        marqueur = "-" if not segments else " "
        print(
            f"    {marqueur} {nom:<12} : {len(segments)} segment(s), "
            f"{_longueur_zone(segments)} car."
        )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("verifier_acquisition")

    try:
        client = JudilibreClient()
    except RuntimeError as exc:
        log.error("Config invalide : %s", exc)
        return 2

    log.info("1/4 healthcheck...")
    try:
        dispo = client.healthcheck()
    except JudilibreError as exc:
        log.error("Echec healthcheck : %s", exc)
        return 1
    log.info("   API disponible : %s", dispo)
    if not dispo:
        return 1

    log.info("2/4 recherche query='licenciement' (page_size=1)...")
    try:
        page = client.rechercher("licenciement", page=0, page_size=1)
    except JudilibreError as exc:
        log.error("Echec recherche : %s", exc)
        return 1
    resultats = page.get("results") or []
    log.info(
        "   total=%s, retournes=%d", page.get("total"), len(resultats)
    )
    if not resultats:
        log.error("Aucun resultat pour 'licenciement'.")
        return 1
    premier_id = resultats[0].get("id")
    if not premier_id:
        log.error("Premier resultat sans id.")
        return 1
    log.info("   premier id : %s", premier_id)

    log.info("3/4 recuperation de la decision integrale...")
    try:
        arret = client.get_decision(premier_id)
    except JudilibreError as exc:
        log.error("Echec get_decision : %s", exc)
        return 1

    log.info("4/4 sauvegarde JSON dans %s ...", DESTINATION)
    destination = DESTINATION / f"{arret.id}.json"
    JudilibreClient.sauvegarde_json(arret, destination)
    log.info("   ecrit : %s", destination.relative_to(RACINE))

    _afficher_resume(arret)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
