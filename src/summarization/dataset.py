"""Construction du jeu de donnees supervise pour le maillon 3.

Etape A du plan :
  1. Construit les paires (entree, cible=sommaire) a partir de `data/corpus/`
     selon 3 strategies d'entree, pour choisir celle qui tient dans la
     fenetre du modele.
  2. Mesure les longueurs en caracteres et en tokens BARThez, calcule la
     fraction d'exemples au-dessus de 512 et 1024 tokens.
  3. Cree un split train/val/test gele (stratifie par chambre regroupee),
     sauvegarde dans `data/splits/*.json` **en ids uniquement**.

Contraintes :
  * Aucun texte de decision n'est loggue ni ecrit hors de `data/`.
  * Les splits sont deterministes (seed fixe) et ne doivent plus changer
    apres validation : `data/splits/` sert de reference figee.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.segmentation import segmenter  # noqa: E402

logger = logging.getLogger("summarization.dataset")

STRATEGIES: tuple[str, ...] = ("motivations", "motivations+expose", "texte_integral")

SEUILS_TOKENS = (512, 1024)


@dataclass(frozen=True)
class Exemple:
    """Une paire (entree, cible=sommaire) prete pour l'entrainement.

    `chambre` et `annee` sont conserves *hors* du texte pour permettre
    la stratification du split sans jamais persister de contenu de
    decision aux cotes des ids.
    """

    id: str
    entree: str
    cible: str
    strategie: str
    chambre: str
    annee: int | None


# ----------------------------------------------------------------------
# Chargement + construction des paires
# ----------------------------------------------------------------------


def _charger_arret(chemin: Path) -> Arret:
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    return Arret.from_dict(donnees)


def _iterer_arrets(dossier_corpus: Path) -> Iterator[Arret]:
    for chemin in sorted(Path(dossier_corpus).glob("*.json")):
        if chemin.name.startswith("_"):
            continue
        try:
            yield _charger_arret(chemin)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Lecture ignoree pour %s : %s", chemin.name, exc)


def _construire_entree(arret: Arret, strategie: str) -> str | None:
    """Assemble l'entree textuelle pour une strategie donnee.

    Retourne `None` si l'entree serait vide : dans ce cas la paire est
    ecartee (principe : abstention plutot qu'invention).
    """
    if strategie == "texte_integral":
        texte = (arret.texte or "").strip()
        return texte or None

    seg = segmenter(arret)
    faits = seg.faits.strip()
    motivations = seg.motivations.strip()

    if strategie == "motivations":
        return motivations or None
    if strategie == "motivations+expose":
        # Ordre : contexte factuel puis raisonnement (proche du sommaire).
        morceaux = [m for m in (faits, motivations) if m]
        return "\n".join(morceaux) if morceaux else None
    raise ValueError(f"Strategie inconnue : {strategie!r}")


def _annee(arret: Arret) -> int | None:
    if not arret.date_decision:
        return None
    try:
        return int(arret.date_decision[:4])
    except ValueError:
        return None


def construire_paires(
    dossier_corpus: Path,
    strategie: str,
) -> Iterator[Exemple]:
    """Genere les paires exploitables (arret avec sommaire, entree non vide)."""
    if strategie not in STRATEGIES:
        raise ValueError(f"Strategie inconnue : {strategie!r} (attendu {STRATEGIES})")

    for arret in _iterer_arrets(Path(dossier_corpus)):
        cible = (arret.sommaire or "").strip()
        if not cible:
            continue
        entree = _construire_entree(arret, strategie)
        if not entree:
            continue
        yield Exemple(
            id=arret.id,
            entree=entree,
            cible=cible,
            strategie=strategie,
            chambre=arret.chambre or "",
            annee=_annee(arret),
        )


# ----------------------------------------------------------------------
# Mesure des longueurs (caracteres et tokens)
# ----------------------------------------------------------------------


def _quantiles(valeurs: list[int]) -> dict[str, float]:
    """Quantiles (min, p25, p50, p75, p95, max) robustes sur une liste."""
    if not valeurs:
        return {"min": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "max": 0}
    triees = sorted(valeurs)
    n = len(triees)

    def _q(p: float) -> float:
        # Interpolation lineaire minimaliste, hors sklearn/numpy.
        if n == 1:
            return float(triees[0])
        rang = p * (n - 1)
        bas = int(rang)
        haut = min(bas + 1, n - 1)
        frac = rang - bas
        return triees[bas] * (1 - frac) + triees[haut] * frac

    return {
        "min": float(triees[0]),
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "p95": _q(0.95),
        "max": float(triees[-1]),
    }


def mesurer_longueurs(
    exemples: Iterable[Exemple],
    tokenizer: Any,
) -> dict[str, Any]:
    """Compte caracteres et tokens (BARThez) pour entree et cible.

    Le tokenizer est appele avec `truncation=False` pour mesurer la
    longueur reelle, meme au-dela de `model_max_length`.
    """
    len_car_entree: list[int] = []
    len_car_cible: list[int] = []
    len_tok_entree: list[int] = []
    len_tok_cible: list[int] = []
    n = 0

    for ex in exemples:
        n += 1
        len_car_entree.append(len(ex.entree))
        len_car_cible.append(len(ex.cible))
        toks_entree = tokenizer(
            ex.entree, add_special_tokens=True, truncation=False
        ).input_ids
        toks_cible = tokenizer(
            ex.cible, add_special_tokens=True, truncation=False
        ).input_ids
        len_tok_entree.append(len(toks_entree))
        len_tok_cible.append(len(toks_cible))

    depassements = {
        seuil: sum(1 for l in len_tok_entree if l > seuil) for seuil in SEUILS_TOKENS
    }
    return {
        "n": n,
        "caracteres": {
            "entree": _quantiles(len_car_entree),
            "cible": _quantiles(len_car_cible),
            "moyenne_entree": statistics.mean(len_car_entree) if n else 0.0,
            "moyenne_cible": statistics.mean(len_car_cible) if n else 0.0,
        },
        "tokens_barthez": {
            "entree": _quantiles(len_tok_entree),
            "cible": _quantiles(len_tok_cible),
            "moyenne_entree": statistics.mean(len_tok_entree) if n else 0.0,
            "moyenne_cible": statistics.mean(len_tok_cible) if n else 0.0,
            "depassements_entree": depassements,
            "taux_depassement_entree": {
                seuil: (depassements[seuil] / n if n else 0.0)
                for seuil in SEUILS_TOKENS
            },
        },
    }


# ----------------------------------------------------------------------
# Split gele train / val / test
# ----------------------------------------------------------------------


def _cle_stratification(
    chambres: list[str],
    annees: list[int | None] | None = None,
    seuil_rare: int = 20,
) -> list[str]:
    """Construit la cle de stratification (chambre | decennie) regroupee.

    Si `annees` est fourni, la cle combine chambre et decennie (annee//10*10).
    Les combinaisons de moins de `seuil_rare` occurrences sont fusionnees
    dans `__rare__` pour garantir la stratification 80/10/10.
    """
    if annees is None:
        cles_brutes = list(chambres)
    else:
        cles_brutes = []
        for chambre, annee in zip(chambres, annees):
            if annee is None:
                cles_brutes.append(f"{chambre}|inconnu")
            else:
                cles_brutes.append(f"{chambre}|{(annee // 10) * 10}s")
    compte = Counter(cles_brutes)
    return [(c if compte[c] >= seuil_rare else "__rare__") for c in cles_brutes]


def construire_splits(
    ids: list[str],
    chambres: list[str],
    *,
    annees: list[int | None] | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seuil_rare_stratif: int = 20,
) -> dict[str, list[str]]:
    """Genere un split train/val/test stratifie (chambre + decennie).

    Retourne `{"train": [...], "val": [...], "test": [...]}` (ids seulement).
    Deterministe pour une seed donnee.
    """
    if len(ids) != len(chambres):
        raise ValueError("ids et chambres doivent avoir la meme longueur.")
    if annees is not None and len(annees) != len(ids):
        raise ValueError("annees doit avoir la meme longueur que ids si fourni.")
    if not ids:
        return {"train": [], "val": [], "test": []}
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Les ratios doivent sommer a 1.")

    # Import local : sklearn est deja dans requirements mais on evite
    # d'imposer un cout d'import en top-level du module.
    from sklearn.model_selection import train_test_split

    strat = _cle_stratification(chambres, annees=annees, seuil_rare=seuil_rare_stratif)
    r_train, r_val, r_test = ratios

    # 1. train vs. (val + test)
    ids_train, ids_temp, _, strat_temp = train_test_split(
        ids,
        strat,
        test_size=r_val + r_test,
        random_state=seed,
        stratify=strat,
        shuffle=True,
    )
    # 2. val vs test au sein du reste
    proportion_test = r_test / (r_val + r_test)
    ids_val, ids_test = train_test_split(
        ids_temp,
        test_size=proportion_test,
        random_state=seed,
        stratify=strat_temp,
        shuffle=True,
    )
    return {"train": sorted(ids_train), "val": sorted(ids_val), "test": sorted(ids_test)}


def sauvegarder_splits(
    splits: dict[str, list[str]],
    dossier: Path,
    meta: dict[str, Any],
) -> dict[str, Path]:
    """Ecrit `train.json`, `val.json`, `test.json` (ids uniquement)."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemins: dict[str, Path] = {}
    for nom, ids in splits.items():
        chemin = dossier / f"{nom}.json"
        chemin.write_text(
            json.dumps(
                {**meta, "split": nom, "n": len(ids), "ids": ids},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        chemins[nom] = chemin
    return chemins


# ----------------------------------------------------------------------
# CLI : mesures des 3 strategies + generation du split
# ----------------------------------------------------------------------


def _afficher_mesures(strategie: str, mesures: dict[str, Any]) -> None:
    print(f"\n=== Strategie : {strategie} ===")
    print(f"  n exemples exploitables : {mesures['n']}")
    if mesures["n"] == 0:
        return
    print(f"  Caracteres entree : moy={mesures['caracteres']['moyenne_entree']:.0f}   "
          f"p50={mesures['caracteres']['entree']['p50']:.0f}   "
          f"p95={mesures['caracteres']['entree']['p95']:.0f}   "
          f"max={mesures['caracteres']['entree']['max']:.0f}")
    print(f"  Caracteres cible  : moy={mesures['caracteres']['moyenne_cible']:.0f}   "
          f"p50={mesures['caracteres']['cible']['p50']:.0f}   "
          f"p95={mesures['caracteres']['cible']['p95']:.0f}   "
          f"max={mesures['caracteres']['cible']['max']:.0f}")
    print(f"  Tokens entree     : moy={mesures['tokens_barthez']['moyenne_entree']:.0f}   "
          f"p50={mesures['tokens_barthez']['entree']['p50']:.0f}   "
          f"p95={mesures['tokens_barthez']['entree']['p95']:.0f}   "
          f"max={mesures['tokens_barthez']['entree']['max']:.0f}")
    print(f"  Tokens cible      : moy={mesures['tokens_barthez']['moyenne_cible']:.0f}   "
          f"p50={mesures['tokens_barthez']['cible']['p50']:.0f}   "
          f"p95={mesures['tokens_barthez']['cible']['p95']:.0f}   "
          f"max={mesures['tokens_barthez']['cible']['max']:.0f}")
    for seuil in SEUILS_TOKENS:
        n_dep = mesures["tokens_barthez"]["depassements_entree"][seuil]
        taux = mesures["tokens_barthez"]["taux_depassement_entree"][seuil]
        print(f"  Entrees > {seuil} tokens : {n_dep} ({taux:.1%})")


def _charger_tokenizer(nom_modele: str) -> Any:
    # On mesure les longueurs reelles au-dela de `model_max_length` : on
    # neutralise donc le warning "Token indices sequence length is longer..."
    # emis par transformers, qui est benin dans notre contexte de mesure.
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(nom_modele)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Etape A : mesures + split gele.")
    p.add_argument("--dossier-corpus", type=Path, default=RACINE / "data" / "corpus")
    p.add_argument("--dossier-splits", type=Path, default=RACINE / "data" / "splits")
    p.add_argument(
        "--modele-tokenizer",
        default="moussaKam/barthez",
        help="Nom HuggingFace du tokenizer (defaut: moussaKam/barthez).",
    )
    p.add_argument(
        "--strategie-split",
        default="motivations+expose",
        choices=list(STRATEGIES),
        help="Strategie utilisee pour definir les paires *eligibles* au split.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pas-de-split",
        action="store_true",
        help="Ne pas ecrire les splits (utile pour ne mesurer que les longueurs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Chargement du tokenizer %s ...", args.modele_tokenizer)
    tokenizer = _charger_tokenizer(args.modele_tokenizer)

    # --- Mesures pour les 3 strategies ---
    mesures_par_strategie: dict[str, dict[str, Any]] = {}
    exemples_split: list[Exemple] = []

    for strategie in STRATEGIES:
        logger.info("Strategie %s : construction des paires ...", strategie)
        exemples = list(construire_paires(args.dossier_corpus, strategie))
        logger.info("  %d paires exploitables", len(exemples))
        if not exemples:
            mesures_par_strategie[strategie] = {"n": 0}
            continue
        logger.info("  tokenisation en cours ...")
        mesures = mesurer_longueurs(exemples, tokenizer)
        mesures_par_strategie[strategie] = mesures
        _afficher_mesures(strategie, mesures)
        if strategie == args.strategie_split:
            exemples_split = exemples

    # --- Split gele (sur la strategie choisie) ---
    if args.pas_de_split or not exemples_split:
        return 0

    logger.info(
        "Construction du split sur %d exemples (strategie=%s, seed=%d) ...",
        len(exemples_split), args.strategie_split, args.seed,
    )
    ids = [ex.id for ex in exemples_split]
    chambres = [ex.chambre for ex in exemples_split]
    annees = [ex.annee for ex in exemples_split]
    splits = construire_splits(ids, chambres, annees=annees, seed=args.seed)

    # Petite verif : repartition par chambre dans chaque split
    par_split: dict[str, Counter[str]] = {
        nom: Counter() for nom in splits
    }
    chambre_par_id = dict(zip(ids, chambres))
    for nom, ids_split in splits.items():
        for _id in ids_split:
            par_split[nom][chambre_par_id[_id]] += 1

    print("\nRepartition des splits :")
    for nom in ("train", "val", "test"):
        total = sum(par_split[nom].values())
        print(f"  {nom:<5} : {total:>5} exemples")
    print("\nRepartition par chambre (train / val / test) :")
    chambres_uniques = sorted({c for c in chambres if c})
    for chambre in chambres_uniques:
        t = par_split["train"][chambre]
        v = par_split["val"][chambre]
        te = par_split["test"][chambre]
        libelle = chambre if len(chambre) <= 50 else chambre[:47] + "..."
        print(f"  {libelle:<52}  {t:>5}  {v:>4}  {te:>4}")

    meta = {
        "seed": args.seed,
        "ratios": [0.8, 0.1, 0.1],
        "strategie_stratification": "chambre_x_decennie_regroupee",
        "strategie_source": args.strategie_split,
    }
    chemins = sauvegarder_splits(splits, args.dossier_splits, meta)
    print("\nSplits ecrits (ids uniquement, aucun texte) :")
    for nom, chemin in chemins.items():
        print(f"  {nom:<5} : {chemin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
