"""CLI d'analyse : segmente un corpus et produit un rapport agrege.

Parcourt `data/corpus/*.json`, applique `segmenter()` a chaque arret,
produit :
  * un CSV horodate dans `reports/segmentation_YYYY-MM-DD_HHMMSS.csv`
    (une ligne par arret, uniquement des chiffres et des metadonnees
    publiques) ;
  * un resume console (repartitions, taux, paires exploitables pour
    l'entrainement, taux de compression, taux de sommaire par
    publication, etc.).

Le CSV ne contient JAMAIS de texte de decision (regle .cursorrules :
seuls des agregats chiffres et metadonnees publiques sortent de `data/`).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import segmenter  # noqa: E402

logger = logging.getLogger("rapport_segmentation")

ZONES_FICHE = ("faits_procedure", "moyens", "motivations", "dispositif")

COLONNES_CSV = [
    "arret_id",
    "annee",
    "chambre",
    "publication",
    "origine",
    "taille_texte",
    "len_sommaire",
    "nb_segments",
    "has_sommaire",
    "has_faits_procedure",
    "has_moyens",
    "has_motivations",
    "has_dispositif",
    "len_faits_procedure",
    "len_moyens",
    "len_motivations",
    "len_dispositif",
    "nb_avertissements",
    "invariant_offsets_ok",
]


def _publication_serialisee(arret: Arret) -> str:
    if not arret.publication:
        return ""
    return "|".join(str(p) for p in arret.publication)


def analyser_arret(arret: Arret) -> dict[str, Any]:
    """Applique `segmenter()` et calcule les stats d'un seul arret."""
    seg = segmenter(arret)
    texte = arret.texte or ""
    sommaire = arret.sommaire or ""
    invariant_ok = all(
        0 <= s.debut <= s.fin <= len(texte)
        and texte[s.debut: s.fin] == s.texte
        for s in seg.segments
    )
    annee: int | None = None
    if arret.date_decision:
        try:
            annee = int(arret.date_decision[:4])
        except ValueError:
            annee = None

    zones = {
        "faits_procedure": seg.faits,
        "moyens":          seg.moyens,
        "motivations":     seg.motivations,
        "dispositif":      seg.dispositif,
    }
    return {
        "arret_id": arret.id,
        "annee": annee,
        "chambre": arret.chambre or "",
        "publication": _publication_serialisee(arret),
        "origine": seg.origine,
        "taille_texte": len(texte),
        "len_sommaire": len(sommaire),
        "nb_segments": len(seg.segments),
        "has_sommaire": int(bool(sommaire)),
        **{f"has_{z}": int(bool(zones[z])) for z in ZONES_FICHE},
        **{f"len_{z}": len(zones[z]) for z in ZONES_FICHE},
        "nb_avertissements": len(seg.avertissements),
        "invariant_offsets_ok": int(invariant_ok),
        # Champs internes : uses uniquement pour agregation, jamais ecrits.
        "_avertissements": list(seg.avertissements),
    }


def _agreger(lignes: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduction des lignes en agregats (dict pret pour affichage / test)."""
    n = len(lignes)
    origines = Counter(l["origine"] for l in lignes)
    zones_presentes = {
        z: sum(l[f"has_{z}"] for l in lignes) for z in ZONES_FICHE
    }
    with_sommaire = sum(l["has_sommaire"] for l in lignes)
    with_avert = sum(1 for l in lignes if l["nb_avertissements"] > 0)

    top_avert: Counter[str] = Counter()
    for l in lignes:
        for a in l["_avertissements"]:
            top_avert[a] += 1

    annees = Counter(l["annee"] for l in lignes if l["annee"] is not None)
    moy_longueur = {
        z: (sum(l[f"len_{z}"] for l in lignes) / n) if n else 0.0
        for z in ZONES_FICHE
    }
    invariant_echecs = [l["arret_id"] for l in lignes if not l["invariant_offsets_ok"]]

    # --- Paires exploitables (arret, sommaire) ---
    paires = [l for l in lignes if l["has_sommaire"] and l["taille_texte"] > 0]
    n_paires = len(paires)
    paires_par_annee = Counter(l["annee"] for l in paires if l["annee"] is not None)
    paires_par_chambre = Counter(l["chambre"] for l in paires if l["chambre"])

    if paires:
        len_textes = [l["taille_texte"] for l in paires]
        len_sommaires = [l["len_sommaire"] for l in paires]
        moy_texte = statistics.mean(len_textes)
        moy_sommaire = statistics.mean(len_sommaires)
        taux_compression = [
            l["len_sommaire"] / l["taille_texte"] for l in paires
        ]
        moy_compression = statistics.mean(taux_compression)
        med_compression = statistics.median(taux_compression)
    else:
        moy_texte = moy_sommaire = moy_compression = med_compression = 0.0

    # --- Taux de sommaire par valeur de publication ---
    total_par_pub: Counter[str] = Counter()
    avec_som_par_pub: Counter[str] = Counter()
    for l in lignes:
        cle_pub = l["publication"] or "(vide)"
        total_par_pub[cle_pub] += 1
        if l["has_sommaire"]:
            avec_som_par_pub[cle_pub] += 1
    taux_som_par_pub = [
        (
            cle,
            avec_som_par_pub[cle],
            total,
            (avec_som_par_pub[cle] / total) if total else 0.0,
        )
        for cle, total in total_par_pub.most_common()
    ]

    return {
        "n": n,
        "origines": dict(origines),
        "zones_presentes": zones_presentes,
        "zones_taux": {z: v / n if n else 0.0 for z, v in zones_presentes.items()},
        "with_sommaire": with_sommaire,
        "sommaire_taux": with_sommaire / n if n else 0.0,
        "with_avert": with_avert,
        "top_avert": top_avert.most_common(5),
        "annees": dict(annees),
        "moy_longueur": moy_longueur,
        "invariant_echecs": invariant_echecs,
        "paires_exploitables": n_paires,
        "paires_par_annee": dict(paires_par_annee),
        "paires_par_chambre": dict(paires_par_chambre),
        "moy_len_texte_paires": moy_texte,
        "moy_len_sommaire_paires": moy_sommaire,
        "moy_taux_compression": moy_compression,
        "med_taux_compression": med_compression,
        "taux_sommaire_par_publication": taux_som_par_pub,
    }


def _ecrire_csv(lignes: list[dict[str, Any]], chemin: Path) -> None:
    """Ecrit le CSV en ne conservant QUE les colonnes agregees (pas de texte)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=COLONNES_CSV)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({k: ligne[k] for k in COLONNES_CSV})


def generer_rapport(
    dossier_corpus: Path,
    dossier_rapport: Path,
    *,
    ecrire_csv: bool = True,
) -> dict[str, Any]:
    """Charge tous les JSON de `dossier_corpus`, segmente et agrege."""
    dossier_corpus = Path(dossier_corpus)
    dossier_rapport = Path(dossier_rapport)

    lignes: list[dict[str, Any]] = []
    fichiers_echec: list[str] = []
    for chemin in sorted(dossier_corpus.glob("*.json")):
        if chemin.name.startswith("_"):
            continue   # ignore les fichiers d'index / meta
        try:
            donnees = json.loads(chemin.read_text(encoding="utf-8"))
            arret = Arret.from_dict(donnees)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            fichiers_echec.append(chemin.name)
            logger.warning("Lecture ignoree pour %s : %s", chemin.name, exc)
            continue
        lignes.append(analyser_arret(arret))

    stats = _agreger(lignes)
    stats["fichiers_illisibles"] = fichiers_echec

    if ecrire_csv and lignes:
        horodatage = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        chemin_csv = dossier_rapport / f"segmentation_{horodatage}.csv"
        _ecrire_csv(lignes, chemin_csv)
        stats["chemin_csv"] = str(chemin_csv)
    return stats


def afficher_console(stats: dict[str, Any]) -> None:
    """Affiche un resume lisible du rapport (aucun texte de decision)."""
    n = stats["n"]
    print()
    print(f"Corpus analyse : {n} arret(s)")
    if n == 0:
        return

    print()
    print("Repartition par origine de segmentation :")
    for origine, occurrences in sorted(stats["origines"].items()):
        print(f"  {origine:<14} : {occurrences:>5}  ({occurrences / n:.1%})")

    print()
    print("Presence des zones (chemin API ou fallback confondus) :")
    for zone in ZONES_FICHE:
        presence = stats["zones_presentes"][zone]
        taux = stats["zones_taux"][zone]
        moy = stats["moy_longueur"][zone]
        print(
            f"  {zone:<20} : {presence:>5} ({taux:.1%})   "
            f"longueur moyenne : {moy:>7.0f} car."
        )

    print()
    print(f"Sommaire officiel : {stats['with_sommaire']}/{n} "
          f"({stats['sommaire_taux']:.1%})")

    # --- Paires exploitables (arret, sommaire) ---
    print()
    print(f"Paires (arret, sommaire) exploitables : {stats['paires_exploitables']}")
    if stats["paires_exploitables"] > 0:
        print(f"  longueur moyenne texte    : "
              f"{stats['moy_len_texte_paires']:>8.0f} car.")
        print(f"  longueur moyenne sommaire : "
              f"{stats['moy_len_sommaire_paires']:>8.0f} car.")
        print(f"  taux compression (som/txt) : "
              f"moy {stats['moy_taux_compression']:.3%}, "
              f"med {stats['med_taux_compression']:.3%}")

        if stats["paires_par_annee"]:
            print("  Top annees (paires) :")
            top_annees_paires = sorted(
                stats["paires_par_annee"].items(),
                key=lambda kv: -kv[1],
            )[:8]
            for annee, occurrences in top_annees_paires:
                print(f"    {annee} : {occurrences}")

        if stats["paires_par_chambre"]:
            print("  Top chambres (paires) :")
            top_ch = sorted(
                stats["paires_par_chambre"].items(),
                key=lambda kv: -kv[1],
            )[:8]
            for chambre, occurrences in top_ch:
                extrait = chambre if len(chambre) <= 50 else chambre[:47] + "..."
                print(f"    {occurrences:>5}  {extrait}")

    # --- Taux de sommaire par publication ---
    if stats["taux_sommaire_par_publication"]:
        print()
        print("Taux de sommaire officiel par valeur de publication :")
        for cle_pub, avec, total, taux in stats["taux_sommaire_par_publication"]:
            extrait = cle_pub if len(cle_pub) <= 60 else cle_pub[:57] + "..."
            print(f"  {avec:>5}/{total:<5} ({taux:6.1%})  {extrait}")

    print()
    print(f"Arrets avec avertissement(s) : {stats['with_avert']}/{n}")
    if stats["top_avert"]:
        print("  Top 5 des messages d'avertissement :")
        for msg, occurrences in stats["top_avert"]:
            extrait = msg if len(msg) <= 90 else msg[:87] + "..."
            print(f"    x{occurrences:<4} {extrait}")

    print()
    print("Repartition par annee (top 10) :")
    top_annees = sorted(stats["annees"].items(), key=lambda kv: -kv[1])[:10]
    for annee, occurrences in top_annees:
        print(f"  {annee} : {occurrences}")

    print()
    print(f"Echecs invariant offsets : {len(stats['invariant_echecs'])}")
    for arret_id in stats["invariant_echecs"][:5]:
        print(f"  - {arret_id}")

    if stats.get("fichiers_illisibles"):
        print()
        print(f"Fichiers JSON illisibles : {len(stats['fichiers_illisibles'])}")
        for nom in stats["fichiers_illisibles"][:5]:
            print(f"  - {nom}")

    if stats.get("chemin_csv"):
        print()
        print(f"CSV ecrit : {stats['chemin_csv']}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segmente un corpus et produit un rapport agrege.",
    )
    p.add_argument(
        "--dossier-corpus",
        type=Path,
        default=RACINE / "data" / "corpus",
    )
    p.add_argument(
        "--dossier-rapport",
        type=Path,
        default=RACINE / "reports",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Ne pas ecrire de CSV (utile pour un dry-run).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stats = generer_rapport(
        args.dossier_corpus,
        args.dossier_rapport,
        ecrire_csv=not args.no_csv,
    )
    if stats.get("n", 0) == 0:
        logger.error("Corpus vide dans %s : rien a analyser.", args.dossier_corpus)
        return 1
    afficher_console(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
