"""OAuth2 client_credentials pour PISTE (obtention et cache du jeton).

Un `FournisseurToken` cache le jeton en memoire, le rafraichit
automatiquement quelques secondes avant expiration, et expose une
methode `invalider()` pour le cas ou le serveur renverrait 401 avec un
token encore juge valide localement.

Aucun secret ni jeton n'est jamais logge ; en cas d'echec, seul le
code HTTP est trace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from .config import ConfigJudilibre

logger = logging.getLogger(__name__)

TIMEOUT_SEC = 15
MARGE_EXPIRATION_SEC = 30  # on renouvelle un peu avant l'echeance


class ErreurToken(RuntimeError):
    """Impossible d'obtenir un jeton OAuth2 (config, reseau, credentials)."""


@dataclass
class _TokenCache:
    valeur: str
    expire_a: float  # timestamp Unix

    def encore_valide(self, marge: float = MARGE_EXPIRATION_SEC) -> bool:
        return time.time() + marge < self.expire_a


class FournisseurToken:
    """Recupere et cache un token `client_credentials` PISTE."""

    def __init__(
        self,
        config: ConfigJudilibre,
        session: requests.Session | None = None,
    ) -> None:
        self._config = config
        self._session = session or requests.Session()
        self._cache: _TokenCache | None = None

    def token(self) -> str:
        """Retourne un token valide, en le renouvelant si necessaire."""
        if self._cache is None or not self._cache.encore_valide():
            self._cache = self._demander_nouveau_token()
        return self._cache.valeur

    def invalider(self) -> None:
        """Force le prochain appel a `token()` a demander un nouveau jeton."""
        self._cache = None

    def _demander_nouveau_token(self) -> _TokenCache:
        donnees = {
            "grant_type": "client_credentials",
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "scope": self._config.scope,
        }
        try:
            reponse = self._session.post(
                self._config.oauth_token_url,
                data=donnees,
                headers={"Accept": "application/json"},
                timeout=TIMEOUT_SEC,
            )
        except requests.RequestException as exc:
            raise ErreurToken("Erreur reseau sur le endpoint OAuth") from exc

        if not reponse.ok:
            logger.warning(
                "OAuth PISTE : HTTP %s sur %s",
                reponse.status_code,
                self._config.oauth_token_url,
            )
            raise ErreurToken(
                f"Echec OAuth2 (HTTP {reponse.status_code}). "
                "Verifier client_id / client_secret et l'abonnement PISTE."
            )
        try:
            payload = reponse.json()
        except ValueError as exc:
            raise ErreurToken("Reponse OAuth non JSON") from exc

        valeur = payload.get("access_token")
        expires_in = payload.get("expires_in")
        if not isinstance(valeur, str) or not isinstance(expires_in, (int, float)):
            raise ErreurToken(
                "Reponse OAuth invalide : 'access_token' ou 'expires_in' manquant."
            )
        return _TokenCache(valeur=valeur, expire_a=time.time() + float(expires_in))
