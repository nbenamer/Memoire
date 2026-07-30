"""Tests unitaires du client Judilibre : mapping et endpoints (mockes).

Contraintes du projet :
  * aucun appel reseau reel (mock via `monkeypatch` sur `requests.Session`) ;
  * aucun vrai credential dans le code (config et token forges sur place) ;
  * aucun texte de decision reel (payloads neutres, non juridiques).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import requests

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.acquisition import (  # noqa: E402
    Arret,
    ConfigJudilibre,
    ErreurToken,
    FournisseurToken,
    JudilibreClient,
    JudilibreError,
)
from src.acquisition.judilibre import _mapper_arret  # noqa: E402


PAYLOAD_FACTICE: dict = {
    "id": "abc123",
    "jurisdiction": "cc",
    "chamber": "civ1",
    "formation": "fs",
    "number": "22-10.000",
    "numbers": ["22-10.000", "22-10.001"],
    "ecli": "ECLI:FR:CCASS:2024:C100001",
    "decision_date": "2024-01-10",
    "type": "arret",
    "solution": "rejet",
    "solution_alt": None,
    "text": "TEXTE_FICTIF",
    "summary": "SOMMAIRE_FICTIF",
    "themes": ["Theme 1", "Theme 2"],
    "publication": ["b"],
    "visa": [{"title": "Texte fictif", "url": "https://example.test/1"}],
    "zones": {
        "introduction": [{"start": 0, "end": 10}],
        "expose": [{"start": 10, "end": 50}],
        "moyens": [{"start": 50, "end": 90}],
        "motivations": [{"start": 90, "end": 130}],
        "dispositif": [{"start": 130, "end": 150}],
        "annexes": [],
    },
    "partial": False,
    "particularInterest": True,
    "contested": {"title": "Juridiction fictive", "number": "20-01234"},
}


class _ReponseFactice:
    """Simulacre minimal de `requests.Response`."""

    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.ok = 200 <= status < 300

    def json(self) -> dict:
        return self._payload


def _config_test() -> ConfigJudilibre:
    """Config totalement factice."""
    return ConfigJudilibre(
        client_id="ID_FICTIF",
        client_secret="SECRET_FICTIF",
        base_url="http://fake.local/v1",
        oauth_token_url="http://fake.local/oauth/token",
    )


class _TokenFige(FournisseurToken):
    """Fournisseur qui renvoie toujours le meme jeton (aucun appel reseau)."""

    def __init__(self, valeur: str = "JETON_FICTIF") -> None:
        self.valeur = valeur
        self.invalidations = 0

    def token(self) -> str:  # type: ignore[override]
        return self.valeur

    def invalider(self) -> None:  # type: ignore[override]
        self.invalidations += 1


def _client_mocke() -> tuple[JudilibreClient, _TokenFige]:
    tok = _TokenFige()
    return JudilibreClient(config=_config_test(), fournisseur_token=tok), tok


def test_mapper_arret_tous_les_champs() -> None:
    arret = _mapper_arret(PAYLOAD_FACTICE)

    assert isinstance(arret, Arret)
    assert arret.id == "abc123"
    assert arret.juridiction == "cc"
    assert arret.chambre == "civ1"
    assert arret.formation == "fs"
    assert arret.numero_pourvoi == "22-10.000"
    assert arret.numeros == ["22-10.000", "22-10.001"]
    assert arret.ecli == "ECLI:FR:CCASS:2024:C100001"
    assert arret.date_decision == "2024-01-10"
    assert arret.type_decision == "arret"
    assert arret.sens_solution == "rejet"
    assert arret.solution_libelle is None
    assert arret.texte == "TEXTE_FICTIF"
    assert arret.sommaire == "SOMMAIRE_FICTIF"
    assert arret.themes == ["Theme 1", "Theme 2"]
    assert arret.publication == ["b"]
    assert arret.visa == [{"title": "Texte fictif", "url": "https://example.test/1"}]
    assert arret.zones["introduction"] == [(0, 10)]
    assert arret.zones["annexes"] == []
    assert arret.partielle is False
    assert arret.interet_particulier is True
    assert arret.decision_attaquee == {
        "title": "Juridiction fictive",
        "number": "20-01234",
    }


def test_mapper_arret_champs_absents_tolere() -> None:
    minimal = {
        "id": "xyz",
        "jurisdiction": "cc",
        "chamber": "com",
        "number": "23-99.999",
    }
    arret = _mapper_arret(minimal)

    assert arret.sommaire is None
    assert arret.texte is None
    assert arret.zones == {
        "introduction": [],
        "expose": [],
        "moyens": [],
        "motivations": [],
        "dispositif": [],
        "annexes": [],
    }


def test_mapper_arret_sommaire_dans_titles_and_summaries() -> None:
    payload = {
        "id": "id1",
        "jurisdiction": "cc",
        "chamber": "civ1",
        "number": "0",
        "titlesAndSummaries": {"summary": "SOMMAIRE_ALTERNATIF"},
    }
    assert _mapper_arret(payload).sommaire == "SOMMAIRE_ALTERNATIF"


def test_mapper_arret_payload_invalide_leve() -> None:
    with pytest.raises(JudilibreError):
        _mapper_arret({"jurisdiction": "cc"})


def test_get_decision_envoie_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le client vise `/decision`, passe `resolve_references=true` et Bearer."""
    appels: list[dict] = []

    def _fake_get(self, url, params=None, headers=None, timeout=None):
        appels.append({"url": url, "params": params, "headers": headers})
        return _ReponseFactice(PAYLOAD_FACTICE)

    monkeypatch.setattr(requests.Session, "get", _fake_get)

    client, _ = _client_mocke()
    arret = client.get_decision("abc123")

    assert len(appels) == 1
    assert appels[0]["url"] == "http://fake.local/v1/decision"
    assert appels[0]["params"] == {"id": "abc123", "resolve_references": "true"}
    assert appels[0]["headers"]["Authorization"] == "Bearer JETON_FICTIF"
    assert "KeyId" not in appels[0]["headers"]
    assert arret.id == "abc123"


def test_rechercher_appelle_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(self, url, params=None, headers=None, timeout=None):
        assert url.endswith("/search")
        assert params == {"query": "licenciement", "page": 0, "page_size": 5}
        assert headers["Authorization"] == "Bearer JETON_FICTIF"
        return _ReponseFactice({"total": 1, "results": [{"id": "abc"}]})

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()
    page = client.rechercher("licenciement", page_size=5)
    assert page["results"][0]["id"] == "abc"


def test_rechercher_query_vide_leve() -> None:
    client, _ = _client_mocke()
    with pytest.raises(ValueError):
        client.rechercher("")


def test_healthcheck_disponible(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(self, url, params=None, headers=None, timeout=None):
        assert url.endswith("/healthcheck")
        return _ReponseFactice({"status": "disponible"})

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()
    assert client.healthcheck() is True


def test_retry_apres_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sur 401, le token est invalide et un second appel est tente."""
    statuts = iter([401, 200])

    def _fake_get(self, url, params=None, headers=None, timeout=None):
        code = next(statuts)
        return _ReponseFactice({"status": "disponible"}, status=code)

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, tok = _client_mocke()
    assert client.healthcheck() is True
    assert tok.invalidations == 1


def test_erreur_http_leve(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(self, url, params=None, headers=None, timeout=None):
        return _ReponseFactice({"error": "not found"}, status=404)

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()
    with pytest.raises(JudilibreError):
        client.get_decision("inconnu")


def test_erreur_reseau_leve(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(self, url, params=None, headers=None, timeout=None):
        raise requests.ConnectionError("panne simulee")

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()
    with pytest.raises(JudilibreError):
        client.healthcheck()


def test_iter_export_pagine_jusqua_limite(monkeypatch: pytest.MonkeyPatch) -> None:
    """`iter_export` doit paginer via `batch` et respecter `limite`."""
    lots = [
        {"results": [dict(PAYLOAD_FACTICE, id=f"id_a{i}") for i in range(3)]},
        {"results": [dict(PAYLOAD_FACTICE, id=f"id_b{i}") for i in range(3)]},
        {"results": []},
    ]
    appels: list[dict] = []

    def _fake_get(self, url, params=None, headers=None, timeout=None):
        appels.append(dict(params or {}))
        idx = params.get("batch", 0) if params else 0
        payload = lots[idx] if idx < len(lots) else {"results": []}
        return _ReponseFactice(payload)

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()

    obtenus = list(client.iter_export(batch_size=3, limite=5, pause_sec=0))

    assert [a.id for a in obtenus] == [
        "id_a0", "id_a1", "id_a2", "id_b0", "id_b1",
    ]
    assert appels[0]["batch"] == 0
    assert appels[0]["batch_size"] == 3
    assert appels[0]["resolve_references"] == "true"
    assert appels[1]["batch"] == 1


def test_iter_export_transmet_les_filtres(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les filtres date/chambre/publication doivent passer en params HTTP."""
    appels: list[dict] = []

    def _fake_get(self, url, params=None, headers=None, timeout=None):
        appels.append(dict(params or {}))
        return _ReponseFactice({"results": []})

    monkeypatch.setattr(requests.Session, "get", _fake_get)
    client, _ = _client_mocke()

    list(client.iter_export(
        date_start="2020-01-01",
        date_end="2020-12-31",
        chamber=["civ1", "civ2"],
        publication=["b"],
        type_decision=["arret"],
        pause_sec=0,
    ))

    assert len(appels) == 1
    p = appels[0]
    assert p["date_start"] == "2020-01-01"
    assert p["date_end"] == "2020-12-31"
    assert p["chamber"] == ["civ1", "civ2"]
    assert p["publication"] == ["b"]
    assert p["type"] == ["arret"]


def test_sauvegarde_json_ecrit_le_fichier(tmp_path: Path) -> None:
    arret = _mapper_arret(PAYLOAD_FACTICE)
    destination = tmp_path / "sous_dossier" / "arret.json"

    resultat = JudilibreClient.sauvegarde_json(arret, destination)

    assert resultat == destination
    contenu = json.loads(destination.read_text(encoding="utf-8"))
    assert contenu["id"] == "abc123"
    assert contenu["zones"]["introduction"] == [[0, 10]]


# -------- Tests du FournisseurToken --------


def test_token_utilise_le_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deux appels a `token()` ne provoquent qu'un seul POST OAuth."""
    appels: list[dict] = []

    def _fake_post(self, url, data=None, headers=None, timeout=None):
        appels.append({"url": url, "data": data})
        return _ReponseFactice({"access_token": "T1", "expires_in": 3600})

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    fournisseur = FournisseurToken(_config_test())
    assert fournisseur.token() == "T1"
    assert fournisseur.token() == "T1"
    assert len(appels) == 1
    assert appels[0]["url"] == "http://fake.local/oauth/token"
    assert appels[0]["data"]["grant_type"] == "client_credentials"
    assert appels[0]["data"]["client_id"] == "ID_FICTIF"
    assert appels[0]["data"]["client_secret"] == "SECRET_FICTIF"


def test_token_renouvelle_apres_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valeurs = iter(["T1", "T2"])

    def _fake_post(self, url, data=None, headers=None, timeout=None):
        return _ReponseFactice(
            {"access_token": next(valeurs), "expires_in": 3600}
        )

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    fournisseur = FournisseurToken(_config_test())
    assert fournisseur.token() == "T1"
    fournisseur.invalider()
    assert fournisseur.token() == "T2"


def test_token_renouvelle_si_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un token dont l'expiration est passee est renouvele automatiquement."""
    valeurs = iter(["T1", "T2"])

    def _fake_post(self, url, data=None, headers=None, timeout=None):
        return _ReponseFactice(
            {"access_token": next(valeurs), "expires_in": 1}
        )

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    fournisseur = FournisseurToken(_config_test())
    assert fournisseur.token() == "T1"
    # On force l'expiration en manipulant l'horloge percue par le cache.
    fournisseur._cache.expire_a = time.time() - 1  # type: ignore[attr-defined]
    assert fournisseur.token() == "T2"


def test_token_erreur_http(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_post(self, url, data=None, headers=None, timeout=None):
        return _ReponseFactice({"error": "invalid_client"}, status=401)

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    fournisseur = FournisseurToken(_config_test())
    with pytest.raises(ErreurToken):
        fournisseur.token()
