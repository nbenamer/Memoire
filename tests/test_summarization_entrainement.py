"""Tests offline du module `entrainement` : troncature, config, preparation.

Aucun telechargement de modele, aucun entrainement lance ici.
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
from src.summarization.entrainement import (  # noqa: E402
    STRATEGIES_TRONCATURE,
    ConfigEntrainement,
    preparer_paires_pour_split,
    tronquer_ids,
)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_config_valeurs_par_defaut() -> None:
    config = ConfigEntrainement()
    assert config.modele == "moussaKam/barthez"
    assert config.strategie_troncature == "queue"
    assert config.max_source_length == 1024
    assert config.max_target_length == 384
    assert config.num_beams == 4
    assert config.max_new_tokens == 256
    assert config.min_new_tokens == 40
    assert config.per_device_train_batch_size == 2
    assert config.gradient_accumulation_steps == 8
    assert config.num_train_epochs == 8
    assert config.fp16 is False
    assert config.bf16 is False
    config.valider()   # ne doit pas lever


def test_config_troncature_inconnue_leve() -> None:
    config = ConfigEntrainement(strategie_troncature="milieu")
    with pytest.raises(ValueError):
        config.valider()


def test_config_tete_queue_depasse_max_source_leve() -> None:
    config = ConfigEntrainement(
        strategie_troncature="tete_queue", n_tete=800, n_queue=300,
        max_source_length=1024,   # 800+300=1100 > 1024
    )
    with pytest.raises(ValueError):
        config.valider()


def test_config_max_new_tokens_depasse_target_leve() -> None:
    config = ConfigEntrainement(max_new_tokens=500, max_target_length=384)
    with pytest.raises(ValueError):
        config.valider()


# ----------------------------------------------------------------------
# Troncature
# ----------------------------------------------------------------------


def test_tronquer_ids_queue_court_inchange() -> None:
    ids = list(range(10, 20))   # 10 tokens
    out = tronquer_ids(ids, max_length=100, strategie="queue",
                       bos_id=1, eos_id=2)
    assert out == ids


def test_tronquer_ids_queue_long_garde_tete_et_force_eos() -> None:
    # 2000 ids, BOS=1, EOS=2 en fin
    ids = [1] + list(range(3, 2001)) + [2]
    out = tronquer_ids(ids, max_length=1024, strategie="queue",
                       bos_id=1, eos_id=2)
    assert len(out) == 1024
    assert out[0] == 1              # BOS preserve
    assert out[-1] == 2              # EOS force en derniere position


def test_tronquer_ids_tete_queue_court_inchange() -> None:
    ids = list(range(10, 100))   # 90 tokens
    out = tronquer_ids(ids, max_length=1024, strategie="tete_queue",
                       bos_id=1, eos_id=2, n_tete=700, n_queue=300)
    assert out == ids


def test_tronquer_ids_tete_queue_long_concatene_debut_et_fin() -> None:
    # 2000 tokens dont BOS(=1) au debut et EOS(=2) a la fin
    ids = [1] + list(range(3, 2001)) + [2]
    out = tronquer_ids(ids, max_length=1024, strategie="tete_queue",
                       bos_id=1, eos_id=2, n_tete=700, n_queue=300)
    assert len(out) == 1000                # 700 + 300
    assert out[0] == 1                     # BOS en tete
    assert out[-1] == 2                    # EOS en fin
    assert out[:700] == ids[:700]          # tete inchangee
    assert out[-300:] == ids[-300:]        # queue inchangee


def test_tronquer_ids_tete_queue_sans_bos_eos() -> None:
    """Si le tokenizer ne fournit pas BOS/EOS, on ne casse pas la troncature."""
    ids = list(range(1, 3000))
    out = tronquer_ids(ids, max_length=1024, strategie="tete_queue",
                       bos_id=None, eos_id=None,
                       n_tete=700, n_queue=300)
    assert len(out) == 1000
    assert out[:700] == ids[:700]
    assert out[-300:] == ids[-300:]


def test_tronquer_ids_strategie_inconnue_leve() -> None:
    with pytest.raises(ValueError):
        tronquer_ids([1, 2, 3], max_length=10, strategie="ailes",
                     bos_id=None, eos_id=None)


def test_toutes_strategies_declarees() -> None:
    assert set(STRATEGIES_TRONCATURE) == {"queue", "tete_queue"}


# ----------------------------------------------------------------------
# Preparation des paires
# ----------------------------------------------------------------------


def _arret(id_: str, motivations_texte: str, sommaire: str) -> Arret:
    """Fabrique un arret avec une zone motivations non vide."""
    zones = {
        "introduction": [],
        "expose":       [],
        "moyens":       [],
        "motivations":  [(0, len(motivations_texte))],
        "dispositif":   [],
        "annexes":      [],
    }
    return Arret(
        id=id_,
        juridiction="cc",
        chambre="civ1",
        formation=None,
        numero_pourvoi="00-00.000",
        texte=motivations_texte,
        zones=zones,
        sommaire=sommaire,
        date_decision="2024-01-15",
    )


def _persister(arret: Arret, dossier: Path) -> None:
    (dossier / f"{arret.id}.json").write_text(
        json.dumps(arret.to_dict(), ensure_ascii=False), encoding="utf-8"
    )


def test_preparer_paires_pour_split_respecte_ordre_du_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for id_ in ("a", "b", "c"):
        _persister(_arret(id_, "motivations non vides.", f"sommaire {id_}"), corpus)

    # Split dans un ordre precis, avec un id manquant intercale
    ids_split = ["b", "manquant", "a"]
    exemples = preparer_paires_pour_split(corpus, ids_split, "motivations")
    assert [ex.id for ex in exemples] == ["b", "a"]
    for ex in exemples:
        assert ex.cible.startswith("sommaire ")
        assert "motivations non vides" in ex.entree
