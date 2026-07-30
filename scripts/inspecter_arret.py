"""Script d'investigation manuelle sur un arret du corpus local.

Affiche a l'ecran (sans persister ni logger) :
  * les metadonnees publiques : type de decision, chambre, formation,
    publication, date, presence de zones API, longueur du texte ;
  * un apercu de 200 caracteres du texte, uniquement pour aider a
    identifier la nature de l'arret (avis, ordonnance, etc.) quand il
    est classe `indetermine` par la segmentation.

Contrainte RGPD : l'apercu est destine a l'investigation manuelle,
n'est pas stocke ni loggue. Rien n'est ecrit sur disque.

Usage :
    venv/bin/python scripts/inspecter_arret.py <id_arret> [<id_arret> ...]
    venv/bin/python scripts/inspecter_arret.py --indetermine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import segmenter  # noqa: E402

DOSSIER_CORPUS_DEFAUT = RACINE / "data" / "corpus"


def _charger_arret(chemin: Path) -> Arret:
    return Arret.from_dict(json.loads(chemin.read_text(encoding="utf-8")))


def _ids_indetermines(dossier: Path) -> list[str]:
    """Parcourt le dossier, applique `segmenter()`, renvoie les ids en `indetermine`."""
    ids: list[str] = []
    for chemin in sorted(dossier.glob("*.json")):
        if chemin.name.startswith("_"):
            continue
        try:
            arret = _charger_arret(chemin)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        seg = segmenter(arret)
        if seg.origine == "indetermine":
            ids.append(arret.id)
    return ids


def inspecter(arret_id: str, dossier: Path) -> None:
    chemin = dossier / f"{arret_id}.json"
    if not chemin.exists():
        print(f"[!] {arret_id} : fichier introuvable dans {dossier}")
        return
    arret = _charger_arret(chemin)
    seg = segmenter(arret)
    texte = arret.texte or ""
    zones_non_vides = {
        nom: len(segs)
        for nom, segs in (arret.zones or {}).items()
        if segs
    }

    print("=" * 72)
    print(f"id             : {arret.id}")
    print(f"type_decision  : {arret.type_decision!r}")
    print(f"juridiction    : {arret.juridiction!r}")
    print(f"chambre        : {arret.chambre!r}")
    print(f"formation      : {arret.formation!r}")
    print(f"publication    : {arret.publication}")
    print(f"date_decision  : {arret.date_decision}")
    print(f"numero_pourvoi : {arret.numero_pourvoi}")
    print(f"sens_solution  : {arret.sens_solution!r}")
    print(f"sommaire ?     : {'oui' if arret.sommaire else 'non'}")
    print(f"texte (len)    : {len(texte)} car.")
    print(f"zones API      : {zones_non_vides or 'aucune'}")
    print(f"segmentation   : origine={seg.origine}, "
          f"segments={len(seg.segments)}, "
          f"avertissements={len(seg.avertissements)}")
    for avert in seg.avertissements:
        extrait = avert if len(avert) <= 100 else avert[:97] + "..."
        print(f"    ! {extrait}")
    print(f"\napercu (200 premiers caracteres, non persiste) :")
    apercu = texte[:200].replace("\n", " ").replace("\r", " ")
    print(f"    {apercu!r}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspecte un ou plusieurs arrets du corpus.")
    p.add_argument("ids", nargs="*", help="Un ou plusieurs ids d'arret.")
    p.add_argument(
        "--indetermine",
        action="store_true",
        help="Inspecte automatiquement tous les arrets en `indetermine`.",
    )
    p.add_argument(
        "--dossier",
        type=Path,
        default=DOSSIER_CORPUS_DEFAUT,
        help="Dossier corpus (defaut: data/corpus).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dossier = args.dossier
    if not dossier.exists():
        print(f"[!] Dossier introuvable : {dossier}")
        return 2

    ids = list(args.ids)
    if args.indetermine:
        ids.extend(i for i in _ids_indetermines(dossier) if i not in ids)

    if not ids:
        print("Aucun id fourni. Utilise `--indetermine` ou passe des ids en argument.")
        return 1

    for arret_id in ids:
        inspecter(arret_id, dossier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
