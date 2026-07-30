"""Tests offline du module dataset (aucun reseau, aucun tokenizer HF)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import Arret  # noqa: E402
from src.summarization.dataset import (  # noqa: E402
    STRATEGIES,
    Exemple,
    _cle_stratification,
    _quantiles,
    construire_paires,
    construire_splits,
    mesurer_longueurs,
    sauvegarder_splits,
)


class TokenizerFactice:
    """Tokenizer factice : renvoie 1 token par caractere non-espace."""

    def __call__(self, texte, add_special_tokens=True, truncation=False):
        ids = [ord(c) for c in texte if not c.isspace()]

        class _Out:
            input_ids = ids
        return _Out()


def _arret(
    id_: str,
    texte: str,
    zones: dict[str, list[tuple[int, int]]] | None = None,
    sommaire: str | None = None,
    chambre: str = "civ1",
    date_decision: str = "2024-05-01",
) -> Arret:
    zones_completes = {
        "introduction": [],
        "expose": [],
        "moyens": [],
        "motivations": [],
        "dispositif": [],
        "annexes": [],
    }
    if zones:
        zones_completes.update(zones)
    return Arret(
        id=id_,
        juridiction="cc",
        chambre=chambre,
        formation=None,
        numero_pourvoi="00-00.000",
        texte=texte,
        zones=zones_completes,
        sommaire=sommaire,
        date_decision=date_decision,
    )


def _persister(arret: Arret, dossier: Path) -> None:
    (dossier / f"{arret.id}.json").write_text(
        json.dumps(arret.to_dict(), ensure_ascii=False), encoding="utf-8"
    )


def test_construire_paires_toutes_strategies(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    # Arret A : zones API renseignees + sommaire
    texte_a = "prefixe expose_faits_facon_procedure motivations_du_raisonnement suffixe"
    zones_a = {
        "expose":      [(texte_a.index("expose"),      texte_a.index("expose") + 6)],
        "motivations": [(texte_a.index("motivations"), texte_a.index("motivations") + 11)],
    }
    _persister(_arret("A", texte_a, zones_a, sommaire="som A"), corpus)

    # Arret B : pas de sommaire -> filtre
    _persister(_arret("B", texte_a, zones_a, sommaire=None), corpus)

    # Arret C : sommaire mais aucune motivation -> exclu de la strategie "motivations"
    zones_c = {
        "expose": [(0, 5)],
    }
    _persister(_arret("C", "abcde reste", zones_c, sommaire="som C"), corpus)

    ids_mot = [ex.id for ex in construire_paires(corpus, "motivations")]
    ids_mot_expose = [ex.id for ex in construire_paires(corpus, "motivations+expose")]
    ids_texte = [ex.id for ex in construire_paires(corpus, "texte_integral")]

    assert ids_mot == ["A"]                          # B filtre (pas de sommaire), C aussi (pas de motivations)
    assert set(ids_mot_expose) == {"A", "C"}         # C a un expose non vide + sommaire
    assert set(ids_texte) == {"A", "C"}              # texte + sommaire suffisent

    # Structure de l'Exemple
    ex_a = next(construire_paires(corpus, "motivations"))
    assert ex_a.id == "A"
    assert ex_a.cible == "som A"
    assert ex_a.strategie == "motivations"
    assert ex_a.chambre == "civ1"
    assert ex_a.annee == 2024


def test_construire_paires_strategie_inconnue(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    try:
        list(construire_paires(corpus, "inexistante"))
    except ValueError:
        return
    raise AssertionError("strategie inconnue doit lever ValueError")


def test_mesurer_longueurs_compte_tokens_et_depassements() -> None:
    """Le tokenizer factice permet de verifier la mecanique sans HF."""
    exemples = [
        Exemple(id="x", entree="a" * 100, cible="b" * 20,
                strategie="motivations", chambre="civ1", annee=2024),
        Exemple(id="y", entree="c" * 600, cible="d" * 200,
                strategie="motivations", chambre="civ1", annee=2024),
        Exemple(id="z", entree="e" * 1100, cible="f" * 500,
                strategie="motivations", chambre="soc",  annee=2020),
    ]
    m = mesurer_longueurs(exemples, TokenizerFactice())
    assert m["n"] == 3
    assert m["caracteres"]["moyenne_entree"] == 600.0
    # 1 exemple > 512 tokens (le 3e), 1 exemple > 1024 tokens (le 3e)
    assert m["tokens_barthez"]["depassements_entree"][512] == 2
    assert m["tokens_barthez"]["depassements_entree"][1024] == 1


def test_quantiles_robustes_liste_vide_et_singleton() -> None:
    q_vide = _quantiles([])
    assert q_vide["p50"] == 0
    q_un = _quantiles([42])
    assert q_un["p50"] == 42


def test_cle_stratification_regroupe_les_rares() -> None:
    ch = ["A"] * 30 + ["B"] * 25 + ["C"] * 5 + ["D"] * 3
    strat = _cle_stratification(ch, seuil_rare=20)
    # A, B assez grandes ; C, D fusionnees en __rare__
    assert strat.count("A") == 30
    assert strat.count("B") == 25
    assert strat.count("__rare__") == 8


def test_construire_splits_deterministe(tmp_path: Path) -> None:
    ids = [f"id_{i}" for i in range(200)]
    chambres = ["A" if i < 100 else "B" for i in range(200)]
    s1 = construire_splits(ids, chambres, seed=42)
    s2 = construire_splits(ids, chambres, seed=42)
    assert s1 == s2
    # Volumes ~= 80/10/10
    assert 155 <= len(s1["train"]) <= 165
    assert 15 <= len(s1["val"]) <= 25
    assert 15 <= len(s1["test"]) <= 25
    # Aucun doublon inter-split
    total = set(s1["train"]) | set(s1["val"]) | set(s1["test"])
    assert len(total) == 200

    # Stratification : chaque split contient les 2 chambres
    for nom in ("train", "val", "test"):
        ch_split = {("A" if int(_id.split("_")[1]) < 100 else "B") for _id in s1[nom]}
        assert ch_split == {"A", "B"}


def test_sauvegarder_splits_ecrit_ids_uniquement(tmp_path: Path) -> None:
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    meta = {"seed": 42, "strategie_stratification": "chambre_regroupee"}
    chemins = sauvegarder_splits(splits, tmp_path, meta)
    assert set(chemins) == {"train", "val", "test"}
    for nom, chemin in chemins.items():
        donnees = json.loads(chemin.read_text(encoding="utf-8"))
        assert donnees["split"] == nom
        assert donnees["seed"] == 42
        assert donnees["ids"] == splits[nom]
        assert "texte" not in json.dumps(donnees)   # RGPD : pas de texte
