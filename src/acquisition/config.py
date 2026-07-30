"""Configuration du client JUDILIBRE : credentials OAuth2 + URLs.

Les secrets restent hors du code source : lecture via `.env`
(python-dotenv). La sandbox PISTE est utilisee par defaut ; les
variables `JUDILIBRE_BASE_URL` et `JUDILIBRE_OAUTH_TOKEN_URL` permettent
de basculer sur la production sans modifier le code.

En pratique, la sandbox PISTE refuse l'auth par en-tete `KeyId` et
exige un jeton Bearer obtenu via OAuth2 `client_credentials` : le
Swagger declare bien ce flow (`SecurityProfile.OAuth2 Application`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

URL_SANDBOX = "https://sandbox-api.piste.gouv.fr/cassation/judilibre/v1.0"
URL_OAUTH_SANDBOX = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"


@dataclass(frozen=True)
class ConfigJudilibre:
    """Configuration immuable transmise au client HTTP."""

    client_id: str
    client_secret: str
    base_url: str
    oauth_token_url: str
    scope: str = "openid"


def charger_config() -> ConfigJudilibre:
    """Charge la configuration depuis l'environnement (`.env` inclus).

    Leve `RuntimeError` avec un message explicite si `JUDILIBRE_CLIENT_ID`
    ou `JUDILIBRE_CLIENT_SECRET` sont absents : sans eux, aucun token ne
    peut etre obtenu et il vaut mieux echouer tot.
    """
    load_dotenv()
    client_id = os.getenv("JUDILIBRE_CLIENT_ID")
    client_secret = os.getenv("JUDILIBRE_CLIENT_SECRET")
    manquants = [
        nom
        for nom, val in (
            ("JUDILIBRE_CLIENT_ID", client_id),
            ("JUDILIBRE_CLIENT_SECRET", client_secret),
        )
        if not val
    ]
    if manquants:
        raise RuntimeError(
            "Variables d'environnement manquantes : "
            + ", ".join(manquants)
            + ". Renseigner client_id et client_secret PISTE dans .env "
            "(portail piste.gouv.fr, application -> onglet Details/Securite)."
        )
    base = os.getenv("JUDILIBRE_BASE_URL", URL_SANDBOX).rstrip("/")
    token_url = os.getenv("JUDILIBRE_OAUTH_TOKEN_URL", URL_OAUTH_SANDBOX)
    return ConfigJudilibre(
        client_id=client_id,          # type: ignore[arg-type]
        client_secret=client_secret,  # type: ignore[arg-type]
        base_url=base,
        oauth_token_url=token_url,
    )
