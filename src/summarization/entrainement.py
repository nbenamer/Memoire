"""Fine-tuning abstractif BARThez sur corpus Cour de cassation.

Etape C du maillon 3. Le split est deja gele sur `motivations`
(seed=42, stratification chambre x decennie, cf. `data/splits/`).

Pipeline :
  1. Charge les paires exploitables selon la strategie d'entree.
  2. Tokenise et tronque (`queue` par defaut, `tete_queue` en option).
  3. Fine-tune BARThez avec Seq2SeqTrainer (eval + ROUGE a chaque epoch,
     load_best_model_at_end sur ROUGE-L, early stopping patience=2).
  4. Sauvegarde le meilleur checkpoint + un `hyperparametres.json` a
     cote pour la reproductibilite (config + versions + seed).

Contraintes RGPD :
  * Aucun texte de decision n'est ecrit dans les logs ni dans le JSON
    d'hyperparametres. Le meilleur checkpoint contient uniquement des
    poids et des metadonnees techniques.
  * `models/` et `data/` sont git-ignored.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parents[2]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.summarization.dataset import Exemple, construire_paires  # noqa: E402

logger = logging.getLogger("summarization.entrainement")

STRATEGIES_TRONCATURE = ("queue", "tete_queue")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class ConfigEntrainement:
    """Hyperparametres complets du fine-tuning. Serialisable en JSON."""

    # Modele et entree
    modele: str = "moussaKam/barthez"
    strategie_source: str = "motivations"
    max_source_length: int = 1024
    max_target_length: int = 384
    strategie_troncature: str = "queue"
    n_tete: int = 700              # tete_queue : nb de tokens conserves en tete
    n_queue: int = 300             # tete_queue : nb de tokens conserves en queue

    # Optimisation
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    label_smoothing_factor: float = 0.10
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # batch effectif = 16
    num_train_epochs: int = 8
    seed: int = 42

    # Precision (fp32 obligatoire sur MPS pour eviter les NaN)
    fp16: bool = False
    bf16: bool = False

    # Evaluation / generation
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2      # ne garde que 2 checkpoints sur disque
    early_stopping_patience: int = 2
    metric_pour_selection: str = "rougeL"
    num_beams: int = 4
    max_new_tokens: int = 256
    min_new_tokens: int = 40
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

    # Divers
    logging_steps: int = 20
    dossier_corpus: str = str(RACINE / "data" / "corpus")
    dossier_splits: str = str(RACINE / "data" / "splits")
    dossier_sortie: str = str(RACINE / "models" / "barthez_v1")

    # Smoke test : entrainement minimal pour valider la chaine
    smoke: bool = False
    smoke_n_train: int = 50
    smoke_n_val: int = 20

    def valider(self) -> None:
        if self.strategie_troncature not in STRATEGIES_TRONCATURE:
            raise ValueError(
                f"strategie_troncature doit etre dans {STRATEGIES_TRONCATURE}, "
                f"recu {self.strategie_troncature!r}"
            )
        if self.strategie_troncature == "tete_queue":
            if self.n_tete + self.n_queue > self.max_source_length:
                raise ValueError(
                    "n_tete + n_queue doit etre <= max_source_length "
                    f"({self.n_tete}+{self.n_queue} > {self.max_source_length})"
                )
        if self.max_new_tokens > self.max_target_length:
            raise ValueError(
                "max_new_tokens ne peut pas depasser max_target_length"
            )


# ----------------------------------------------------------------------
# Troncature
# ----------------------------------------------------------------------


def tronquer_ids(
    ids: list[int],
    max_length: int,
    strategie: str,
    *,
    bos_id: int | None,
    eos_id: int | None,
    n_tete: int = 700,
    n_queue: int = 300,
) -> list[int]:
    """Tronque une sequence d'ids en preservant BOS/EOS.

    - "queue"      : garde les `max_length` premiers tokens (comportement
                     par defaut de HuggingFace) ;
    - "tete_queue" : garde `n_tete` premiers + `n_queue` derniers tokens,
                     en s'assurant que BOS reste en tete et EOS en fin.

    Aucune modification si `len(ids) <= max_length` (ou <= n_tete+n_queue
    en tete_queue).
    """
    if strategie not in STRATEGIES_TRONCATURE:
        raise ValueError(f"Strategie inconnue : {strategie!r}")

    if strategie == "queue":
        if len(ids) <= max_length:
            return list(ids)
        tete = list(ids[:max_length])
        # Force EOS a la derniere position pour ne pas laisser une phrase
        # tronquee sans terminaison, si le modele en fournit un.
        if eos_id is not None and tete[-1] != eos_id:
            tete[-1] = eos_id
        return tete

    # strategie == "tete_queue"
    total_cible = n_tete + n_queue
    if len(ids) <= total_cible:
        return list(ids)

    debut = list(ids[:n_tete])
    fin = list(ids[-n_queue:])
    # Preserver BOS en tete et EOS en fin (si le tokenizer les fournit).
    if bos_id is not None and debut[0] != bos_id:
        debut.insert(0, bos_id)
    if eos_id is not None and fin[-1] != eos_id:
        fin.append(eos_id)
    return debut + fin


# ----------------------------------------------------------------------
# Preparation des paires
# ----------------------------------------------------------------------


def _lire_ids_split(chemin: Path) -> list[str]:
    donnees = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return list(donnees.get("ids", []))


def preparer_paires_pour_split(
    dossier_corpus: Path,
    ids_split: list[str],
    strategie_source: str,
) -> list[Exemple]:
    """Retourne les paires exploitables du corpus dont l'id est dans `ids_split`.

    Preserve l'ordre du split (permet un mode `--smoke` deterministe).
    """
    ids_split_set = set(ids_split)
    tous = {ex.id: ex for ex in construire_paires(dossier_corpus, strategie_source)}
    manquants = [i for i in ids_split if i not in tous]
    if manquants:
        logger.warning(
            "%d ids du split n'ont pas de paire exploitable pour la "
            "strategie %s (ex: %s)",
            len(manquants), strategie_source, manquants[:3],
        )
    return [tous[i] for i in ids_split if i in tous]


# ----------------------------------------------------------------------
# Encodage pour le Trainer
# ----------------------------------------------------------------------


def encoder_exemples(
    exemples: list[Exemple],
    tokenizer,
    config: ConfigEntrainement,
) -> Any:
    """Convertit les Exemples en `datasets.Dataset` tokenises.

    * Entree tokenisee sans troncature puis passee dans `tronquer_ids`
      (necessaire pour supporter la strategie `tete_queue`).
    * Cible tokenisee avec troncature simple (queue) a `max_target_length`.
    """
    from datasets import Dataset

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def _construire(ex: Exemple) -> dict[str, Any]:
        ids_source = tokenizer(
            ex.entree, add_special_tokens=True, truncation=False,
        ).input_ids
        input_ids = tronquer_ids(
            ids_source,
            max_length=config.max_source_length,
            strategie=config.strategie_troncature,
            bos_id=bos_id,
            eos_id=eos_id,
            n_tete=config.n_tete,
            n_queue=config.n_queue,
        )
        attention_mask = [1] * len(input_ids)
        labels = tokenizer(
            ex.cible,
            add_special_tokens=True,
            truncation=True,
            max_length=config.max_target_length,
        ).input_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    donnees = [_construire(ex) for ex in exemples]
    return Dataset.from_list(donnees)


# ----------------------------------------------------------------------
# Metriques ROUGE
# ----------------------------------------------------------------------


def construire_compute_metrics(tokenizer):
    """Retourne une fonction `compute_metrics` pour `Seq2SeqTrainer`.

    Requiert `predict_with_generate=True` : predictions sont des token ids
    generes, labels sont les token ids cibles (avec -100 pour le padding).
    """
    from rouge_score import rouge_scorer
    import numpy as np

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )

    def _fn(eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        # Predictions peuvent contenir des valeurs negatives si le beam a
        # produit des tokens de padding : on les remplace par pad_token_id.
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        predictions = np.where(np.asarray(predictions) < 0, pad_id, predictions)
        labels = np.where(np.asarray(labels) == -100, pad_id, labels)

        textes_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        textes_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(textes_preds, textes_labels):
            resultats = scorer.score(ref.strip(), pred.strip())
            for metrique, valeur in resultats.items():
                acc[metrique].append(valeur.fmeasure)

        return {
            "rouge1": float(sum(acc["rouge1"]) / len(acc["rouge1"])) if acc["rouge1"] else 0.0,
            "rouge2": float(sum(acc["rouge2"]) / len(acc["rouge2"])) if acc["rouge2"] else 0.0,
            "rougeL": float(sum(acc["rougeL"]) / len(acc["rougeL"])) if acc["rougeL"] else 0.0,
        }

    return _fn


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


def _fixer_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def _resoudre_device(prefere: str) -> str:
    """Retourne 'mps' si dispo, sinon 'cpu' avec log clair."""
    try:
        import torch
    except ImportError:
        logger.warning("torch indisponible : fallback CPU logique")
        return "cpu"
    if prefere == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefere == "mps":
        logger.warning("MPS demande mais indisponible : fallback CPU")
    return "cpu"


def _ecrire_hyperparametres(chemin: Path, config: ConfigEntrainement,
                            device: str, extra: dict[str, Any]) -> None:
    """Serialise les hyperparametres + versions pour reproductibilite."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for nom in ("torch", "transformers", "datasets", "accelerate", "rouge_score"):
        try:
            mod = __import__(nom)
            versions[nom] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[nom] = "absent"
    payload = {
        "config": asdict(config),
        "device": device,
        "versions": versions,
        **extra,
    }
    chemin.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")


def entrainer(config: ConfigEntrainement) -> dict[str, Any]:
    """Point d'entree principal du fine-tuning.

    Retourne un dict avec les metriques finales et le chemin du meilleur
    checkpoint.
    """
    config.valider()
    _fixer_seeds(config.seed)

    # Imports lourds a l'interieur pour ne pas les payer en tests offline.
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    device = _resoudre_device("mps")
    logger.info("Device selectionne : %s", device)

    # --- Tokenizer + modele ---
    logger.info("Chargement du tokenizer et du modele %s ...", config.modele)
    tokenizer = AutoTokenizer.from_pretrained(config.modele)
    modele = AutoModelForSeq2SeqLM.from_pretrained(config.modele)

    # --- Datasets ---
    dossier_corpus = Path(config.dossier_corpus)
    dossier_splits = Path(config.dossier_splits)
    ids_train = _lire_ids_split(dossier_splits / "train.json")
    ids_val = _lire_ids_split(dossier_splits / "val.json")

    if config.smoke:
        logger.info("Mode SMOKE : reduction train=%d val=%d, epochs=1",
                    config.smoke_n_train, config.smoke_n_val)
        ids_train = ids_train[: config.smoke_n_train]
        ids_val = ids_val[: config.smoke_n_val]

    logger.info("Preparation des paires : train=%d, val=%d",
                len(ids_train), len(ids_val))
    exemples_train = preparer_paires_pour_split(
        dossier_corpus, ids_train, config.strategie_source,
    )
    exemples_val = preparer_paires_pour_split(
        dossier_corpus, ids_val, config.strategie_source,
    )
    # Copie legere de la config pour permettre les overrides du smoke
    # sans mutation de la config d'origine (elle est deja loggee).
    config_encodage = dataclasses.replace(config)

    # --- Arguments d'entrainement ---
    dossier_sortie = Path(config.dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    epochs = 1 if config.smoke else config.num_train_epochs

    # En mode smoke : entrainement volontairement leger pour valider la
    # chaine (tokenisation -> forward -> backward -> eval -> save) en
    # quelques minutes sur MPS. Ces overrides ne modifient PAS la config
    # d'origine (qui est deja loggee), grace a `dataclasses.replace`.
    if config.smoke:
        num_beams = 1
        max_new_tokens = 60
        per_device_batch = 1
        grad_accum = 2
        config_encodage = dataclasses.replace(
            config, max_source_length=512, max_target_length=192,
        )
    else:
        num_beams = config.num_beams
        max_new_tokens = config.max_new_tokens
        per_device_batch = config.per_device_train_batch_size
        grad_accum = config.gradient_accumulation_steps

    logger.info("Encodage BARThez en cours (source=%d, target=%d) ...",
                config_encodage.max_source_length,
                config_encodage.max_target_length)
    ds_train = encoder_exemples(exemples_train, tokenizer, config_encodage)
    ds_val = encoder_exemples(exemples_val, tokenizer, config_encodage)

    args = Seq2SeqTrainingArguments(
        output_dir=str(dossier_sortie),
        overwrite_output_dir=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing_factor,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_pour_selection,
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=max_new_tokens,
        generation_num_beams=num_beams,
        logging_steps=1 if config.smoke else config.logging_steps,
        disable_tqdm=False,
        seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        report_to=[],   # PAS de tensorboard/wandb : pas de fuite de texte
        remove_unused_columns=False,
    )

    # Generation config pour l'evaluation (beams, min/max tokens, ...)
    # Le mode smoke reduit num_beams a 1 et max_new_tokens a 80 pour tenir
    # en quelques minutes ; les autres parametres restent normaux.
    modele.generation_config.num_beams = num_beams
    modele.generation_config.max_new_tokens = max_new_tokens
    modele.generation_config.min_new_tokens = (
        min(config.min_new_tokens, max_new_tokens - 1) if config.smoke
        else config.min_new_tokens
    )
    modele.generation_config.length_penalty = config.length_penalty
    modele.generation_config.no_repeat_ngram_size = config.no_repeat_ngram_size
    modele.generation_config.early_stopping = True

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=modele,
        label_pad_token_id=-100,
        padding="longest",
    )

    callbacks = []
    if not config.smoke:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
        ))

    trainer = Seq2SeqTrainer(
        model=modele,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=construire_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # --- Ecriture des hyperparametres AVANT l'entrainement ---
    _ecrire_hyperparametres(
        dossier_sortie / "hyperparametres.json",
        config,
        device,
        extra={
            "n_train": len(ds_train),
            "n_val": len(ds_val),
        },
    )

    # --- Entrainement ---
    logger.info("Lancement du fine-tuning (%d epochs) ...", epochs)
    resultats_train = trainer.train()

    # --- Evaluation finale (best model deja rechargee) ---
    logger.info("Evaluation finale sur val ...")
    metriques_val = trainer.evaluate()

    # --- Sauvegarde meilleur modele ---
    chemin_best = dossier_sortie / "best_model"
    trainer.save_model(str(chemin_best))
    tokenizer.save_pretrained(str(chemin_best))
    logger.info("Meilleur modele sauvegarde dans %s", chemin_best)

    # --- Enrichissement du JSON avec les metriques finales ---
    payload = json.loads(
        (dossier_sortie / "hyperparametres.json").read_text(encoding="utf-8")
    )
    payload["metriques_val_final"] = {
        k: v for k, v in metriques_val.items()
        if isinstance(v, (int, float))
    }
    payload["train_runtime_sec"] = float(getattr(resultats_train, "metrics", {}).get(
        "train_runtime", 0.0
    )) if hasattr(resultats_train, "metrics") else 0.0
    (dossier_sortie / "hyperparametres.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "chemin_best": str(chemin_best),
        "metriques_val": metriques_val,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tuning BARThez pour fiche d'arret.")
    p.add_argument("--modele", default="moussaKam/barthez")
    p.add_argument("--strategie-source", default="motivations",
                   choices=["motivations", "motivations+expose", "texte_integral"])
    p.add_argument("--troncature", default="queue",
                   choices=list(STRATEGIES_TRONCATURE),
                   help="Strategie de troncature de l'entree.")
    p.add_argument("--n-tete", type=int, default=700)
    p.add_argument("--n-queue", type=int, default=300)
    p.add_argument("--max-source-length", type=int, default=1024)
    p.add_argument("--max-target-length", type=int, default=384)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dossier-sortie", type=Path,
                   default=RACINE / "models" / "barthez_v1")
    p.add_argument("--dossier-corpus", type=Path,
                   default=RACINE / "data" / "corpus")
    p.add_argument("--dossier-splits", type=Path,
                   default=RACINE / "data" / "splits")
    p.add_argument("--smoke", action="store_true",
                   help="Test de fumee : 50 exemples / 1 epoch pour valider "
                        "la chaine de bout en bout.")
    p.add_argument("--smoke-n-train", type=int, default=50)
    p.add_argument("--smoke-n-val", type=int, default=20)
    return p


def _config_depuis_args(args: argparse.Namespace) -> ConfigEntrainement:
    return ConfigEntrainement(
        modele=args.modele,
        strategie_source=args.strategie_source,
        strategie_troncature=args.troncature,
        n_tete=args.n_tete,
        n_queue=args.n_queue,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_beams=args.num_beams,
        seed=args.seed,
        dossier_corpus=str(args.dossier_corpus),
        dossier_splits=str(args.dossier_splits),
        dossier_sortie=str(args.dossier_sortie),
        smoke=args.smoke,
        smoke_n_train=args.smoke_n_train,
        smoke_n_val=args.smoke_n_val,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence transformers verbeux sur les longueurs de tokens.
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = _config_depuis_args(args)
    resultats = entrainer(config)

    print()
    print(f"Best checkpoint : {resultats['chemin_best']}")
    print(f"n_train         : {resultats['n_train']}")
    print(f"n_val           : {resultats['n_val']}")
    print("Metriques val finales :")
    for cle, valeur in sorted(resultats["metriques_val"].items()):
        if isinstance(valeur, (int, float)):
            print(f"  {cle:<24} : {valeur:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
