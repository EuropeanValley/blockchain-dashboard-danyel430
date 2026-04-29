# """
# Blockchain API client.

# Provides helper functions to fetch blockchain data from public APIs.
# """

# import requests

# BASE_URL = "https://blockchain.info"


# def get_latest_block() -> dict:
#     """Return the latest block summary."""
#     response = requests.get(f"{BASE_URL}/latestblock", timeout=10)
#     response.raise_for_status()
#     return response.json()


# def get_block(block_hash: str) -> dict:
#     """Return full details for a block identified by *block_hash*."""
#     response = requests.get(
#         f"{BASE_URL}/rawblock/{block_hash}", timeout=10
#     )
#     response.raise_for_status()
#     return response.json()


# def get_difficulty_history(n_points: int = 100) -> list[dict]:
#     """Return the last *n_points* difficulty values as a list of dicts."""
#     response = requests.get(
#         f"{BASE_URL}/charts/difficulty",
#         params={"timespan": "1year", "format": "json", "sampled": "true"},
#         timeout=10,
#     )
#     response.raise_for_status()
#     data = response.json()
#     return data.get("values", [])[-n_points:]

"""
blockchain_client.py
--------------------
Centraliza todas las llamadas a las APIs públicas de Bitcoin.
APIs usadas (sin clave, sin registro):
  - Blockstream : https://blockstream.info/api
  - Blockchain.info : https://api.blockchain.info
  - Mempool.space : https://mempool.space/api
"""

import time
import requests

# ── Base URLs ──────────────────────────────────────────────────────────────────
BASE_BS  = "https://blockstream.info/api"       # Blockstream
BASE_BC  = "https://api.blockchain.info"        # Blockchain.info
BASE_MEM = "https://mempool.space/api"          # Mempool.space

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoChainAnalyzer/1.0"})


# ── Helper ─────────────────────────────────────────────────────────────────────
def _get(url, params=None, retries=3):
    """GET con timeout y reintentos exponenciales."""
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise ConnectionError(f"Error al conectar con {url}: {exc}") from exc
            time.sleep(2 ** attempt)


# ── Blockstream endpoints ──────────────────────────────────────────────────────
def get_tip_height() -> int:
    """Altura del bloque más reciente."""
    return _get(f"{BASE_BS}/blocks/tip/height").json()


def get_tip_hash() -> str:
    """Hash del bloque más reciente."""
    return _get(f"{BASE_BS}/blocks/tip/hash").text.strip()


def get_block(block_hash: str) -> dict:
    """Datos JSON de un bloque dado su hash."""
    return _get(f"{BASE_BS}/block/{block_hash}").json()


def get_block_by_height(height: int) -> dict:
    """Datos JSON de un bloque dado su altura."""
    block_hash = _get(f"{BASE_BS}/block-height/{height}").text.strip()
    return get_block(block_hash)


def get_recent_blocks(n: int = 100) -> list[dict]:
    """
    Devuelve los últimos n bloques de forma eficiente.
    Blockstream permite pedir 10 bloques por llamada (/blocks/:height).
    """
    tip_height = get_tip_height()
    blocks: list[dict] = []
    height = tip_height

    while len(blocks) < n:
        batch = _get(f"{BASE_BS}/blocks/{height}").json()  # lista de 10 bloques
        if not batch:
            break
        blocks.extend(batch)
        height = batch[-1]["height"] - 1  # siguiente batch empieza aquí

    return blocks[:n]


# ── Blockchain.info endpoints ──────────────────────────────────────────────────
def get_difficulty_history() -> dict:
    """
    Historial de dificultad de los últimos 2 años.
    Devuelve {"values": [{"x": unix_ts, "y": difficulty}, ...]}
    """
    return _get(
        f"{BASE_BC}/charts/difficulty",
        params={"timespan": "2years", "format": "json", "cors": "true"},
    ).json()


# ── Mempool.space endpoints ────────────────────────────────────────────────────
def get_mempool_stats() -> dict:
    """Estadísticas actuales del mempool."""
    return _get(f"{BASE_MEM}/mempool").json()