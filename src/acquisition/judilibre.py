"""Client HTTP pour l'API JUDILIBRE (portail PISTE, Cour de cassation).

Trois responsabilites, et rien de plus :
  * `healthcheck`     : verifier la disponibilite du service ;
  * `rechercher`      : requeter `/search` (renvoie le JSON brut) ;
  * `get_decision`    : recuperer une decision integrale puis la mapper
                        vers la dataclass `Arret` (aucune logique metier
                        au-dela du mapping) ;
  * `sauvegarde_json` : ecrire un `Arret` dans `data/` (git-ignored),
                        pour le cache local du pipeline aval.

Authentification : OAuth2 `client_credentials` (PISTE). Le jeton Bearer
est fourni par un `FournisseurToken` injectable, ce qui rend le client
testable sans reseau.

Contraintes RGPD respectees ici : jamais de log du texte de la decision,
jamais de log du jeton ; aucune sortie vers un service cloud tiers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from .config import ConfigJudilibre, charger_config
from .models import Arret
from .oauth import ErreurToken, FournisseurToken

logger = logging.getLogger(__name__)

TIMEOUT_SEC = 15
NOMS_ZONES = (
    "introduction",
    "expose",
    "moyens",
    "motivations",
    "dispositif",
    "annexes",
)


class JudilibreError(RuntimeError):
    """Erreur specifique au client JUDILIBRE (reseau, HTTP, format)."""


class JudilibreClient:
    """Client minimal Judilibre : healthcheck, rechercher, get_decision."""

    def __init__(
        self,
        config: ConfigJudilibre | None = None,
        session: requests.Session | None = None,
        fournisseur_token: FournisseurToken | None = None,
    ) -> None:
        self._config = config or charger_config()
        self._session = session or requests.Session()
        self._token = fournisseur_token or FournisseurToken(
            self._config, session=self._session
        )

    def healthcheck(self) -> bool:
        """Retourne `True` si l'API declare `status == 'disponible'`."""
        data = self._get("/healthcheck")
        return data.get("status") == "disponible"

    def rechercher(
        self,
        query: str,
        *,
        page: int = 0,
        page_size: int = 10,
    ) -> dict[str, Any]:
        """Recherche libre : retourne le JSON brut `searchPage` de l'API.

        On ne mappe pas encore vers `Arret` : `/search` ne renvoie ni le
        texte ni les zones (spec Swagger). Le mapping metier reste reserve
        a `get_decision`, appele avec l'`id` de chaque resultat pertinent.
        """
        if not query:
            raise ValueError("Parametre 'query' vide.")
        return self._get(
            "/search",
            params={"query": query, "page": page, "page_size": page_size},
        )

    def iter_export(
        self,
        *,
        date_start: str | None = None,
        date_end: str | None = None,
        chamber: list[str] | None = None,
        publication: list[str] | None = None,
        type_decision: list[str] | None = None,
        batch_size: int = 100,
        limite: int | None = None,
        pause_sec: float = 0.2,
        resolve_references: bool = True,
    ) -> Iterator[Arret]:
        """Iterateur pagine sur `/export`, un `Arret` a la fois.

        Chaque `decisionFull` retourne par l'API est mappe vers `Arret`
        via le meme code que `get_decision` (invariant unique). Utilise
        pour construire un corpus. `limite` borne le nombre total
        d'arrets rendus ; `pause_sec` est appliquee entre appels HTTP
        pour respecter les quotas PISTE (typ. 20 req/s max).
        """
        params_base: dict[str, Any] = {
            "batch_size": batch_size,
            "resolve_references": "true" if resolve_references else "false",
        }
        if date_start:
            params_base["date_start"] = date_start
        if date_end:
            params_base["date_end"] = date_end
        if chamber:
            params_base["chamber"] = list(chamber)
        if publication:
            params_base["publication"] = list(publication)
        if type_decision:
            params_base["type"] = list(type_decision)

        rendus = 0
        batch = 0
        while True:
            params = dict(params_base, batch=batch)
            payload = self._get("/export", params=params)
            resultats = payload.get("results") or []
            if not resultats:
                return
            for donnees in resultats:
                try:
                    arret = _mapper_arret(donnees)
                except JudilibreError as exc:
                    logger.warning("Arret ignore : %s", exc)
                    continue
                yield arret
                rendus += 1
                if limite is not None and rendus >= limite:
                    return
            batch += 1
            if pause_sec > 0:
                time.sleep(pause_sec)

    def get_decision(self, id: str) -> Arret:
        """Recupere une decision integrale et la mappe vers `Arret`.

        Utilise `resolve_references=true` pour recevoir les intitules
        complets (juridiction, chambre, solution, ...) plutot que les
        clefs de taxonomie. Toujours plus lisible pour le pipeline aval.
        """
        if not id:
            raise ValueError("Identifiant de decision manquant.")
        payload = self._get(
            "/decision",
            params={"id": id, "resolve_references": "true"},
        )
        return _mapper_arret(payload)

    @staticmethod
    def sauvegarde_json(arret: Arret, chemin: str | Path) -> Path:
        """Ecrit un `Arret` en JSON UTF-8, avec creation du dossier parent.

        Le pipeline persiste sous `data/` qui est git-ignored : conformite
        RGPD par construction (aucun texte de decision ne remonte au depot).
        """
        destination = Path(chemin)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(arret.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    def _get(self, chemin: str, params: dict[str, Any] | None = None) -> dict:
        """GET generique avec Bearer OAuth2 et une tentative de retry sur 401.

        Un 401 peut arriver si le token a ete revoque cote PISTE avant
        son expiration locale : on invalide le cache et on retente une
        seule fois.
        """
        url = f"{self._config.base_url}{chemin}"
        reponse = self._appel_bearer(url, params)
        if reponse.status_code == 401:
            logger.info("Judilibre : 401, rafraichissement du token puis retry.")
            self._token.invalider()
            reponse = self._appel_bearer(url, params)

        if not reponse.ok:
            logger.warning(
                "Judilibre : HTTP %s sur %s", reponse.status_code, chemin
            )
            raise JudilibreError(
                f"Erreur HTTP {reponse.status_code} sur {chemin}"
            )
        try:
            return reponse.json()
        except ValueError as exc:
            raise JudilibreError(f"Reponse non JSON sur {chemin}") from exc

    def _appel_bearer(
        self, url: str, params: dict[str, Any] | None
    ) -> requests.Response:
        """Un appel GET avec `Authorization: Bearer <token>`."""
        try:
            jeton = self._token.token()
        except ErreurToken as exc:
            raise JudilibreError(str(exc)) from exc
        headers = {
            "Authorization": f"Bearer {jeton}",
            "Accept": "application/json",
        }
        try:
            return self._session.get(
                url, params=params, headers=headers, timeout=TIMEOUT_SEC
            )
        except requests.RequestException as exc:
            raise JudilibreError(f"Erreur reseau sur {url}") from exc


def _extraire_zones(brut: Any) -> dict[str, list[tuple[int, int]]]:
    """Normalise le bloc `zones` en `{nom: [(start, end), ...]}`.

    Toujours retourne les 6 clefs attendues (listes vides si absentes),
    ce qui rend le maillon `segmentation/` plus simple : pas besoin de
    tester la presence des clefs.
    """
    zones: dict[str, list[tuple[int, int]]] = {nom: [] for nom in NOMS_ZONES}
    if not isinstance(brut, dict):
        return zones
    for nom in NOMS_ZONES:
        segments = brut.get(nom) or []
        zones[nom] = [
            (int(seg["start"]), int(seg["end"]))
            for seg in segments
            if isinstance(seg, dict) and "start" in seg and "end" in seg
        ]
    return zones


def _extraire_sommaire(payload: dict) -> str | None:
    """Sommaire officiel : soit `summary`, soit `titlesAndSummaries.summary`.

    L'API expose historiquement `summary` a plat, mais les decisions
    Cour de cassation peuvent aussi le porter dans `titlesAndSummaries`.
    """
    sommaire = payload.get("summary")
    if sommaire:
        return sommaire
    titres = payload.get("titlesAndSummaries")
    if isinstance(titres, dict):
        valeur = titres.get("summary")
        if isinstance(valeur, str) and valeur:
            return valeur
    return None


def _mapper_arret(payload: dict) -> Arret:
    """Mappe une reponse `decisionFull` (JSON) sur la dataclass `Arret`.

    Isole le pipeline des noms de champs API : tout renommage se fait
    ici, et nulle part ailleurs.
    """
    if not isinstance(payload, dict) or "id" not in payload:
        raise JudilibreError(
            "Reponse Judilibre invalide : champ 'id' manquant."
        )
    return Arret(
        id=payload["id"],
        juridiction=payload.get("jurisdiction") or "",
        chambre=payload.get("chamber") or "",
        formation=payload.get("formation"),
        numero_pourvoi=payload.get("number") or "",
        numeros=list(payload.get("numbers") or []),
        ecli=payload.get("ecli"),
        date_decision=payload.get("decision_date"),
        type_decision=payload.get("type"),
        sens_solution=payload.get("solution"),
        solution_libelle=payload.get("solution_alt"),
        texte=payload.get("text"),
        sommaire=_extraire_sommaire(payload),
        themes=list(payload.get("themes") or []),
        publication=list(payload.get("publication") or []),
        visa=list(payload.get("visa") or []),
        zones=_extraire_zones(payload.get("zones")),
        partielle=payload.get("partial"),
        interet_particulier=payload.get("particularInterest"),
        decision_attaquee=payload.get("contested"),
    )
