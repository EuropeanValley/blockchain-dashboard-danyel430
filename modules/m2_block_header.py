# """Starter file for module M2."""

# import streamlit as st

# from api.blockchain_client import get_block


# def render() -> None:
#     """Render the M2 panel."""
#     st.header("M2 - Block Header Analyzer")
#     st.write("Use this module to inspect the fields of one block header.")

#     block_hash = st.text_input(
#         "Block hash",
#         placeholder="Enter a block hash",
#         key="m2_hash",
#     )

#     if st.button("Look up block", key="m2_lookup") and block_hash:
#         with st.spinner("Fetching data..."):
#             try:
#                 block = get_block(block_hash)
#                 st.subheader("Block header fields")
#                 header_fields = {
#                     "Hash": block.get("hash"),
#                     "Height": block.get("height"),
#                     "Time": block.get("time"),
#                     "Nonce": block.get("nonce"),
#                     "Bits": block.get("bits"),
#                     "Merkle root": block.get("mrkl_root"),
#                     "Previous block": block.get("prev_block"),
#                 }
#                 for label, value in header_fields.items():
#                     st.write(f"**{label}:** {value}")
#             except Exception as exc:
#                 st.error(f"Error fetching block: {exc}")
#     elif not block_hash:
#         st.info("Enter a block hash and click Look up block.")


"""
m2_block_header.py
------------------
Módulo M2 — Block Header Analyzer.

Muestra la estructura de 80 bytes del cabecero del último bloque y verifica
el Proof of Work localmente con Python puro (hashlib).

Campos del cabecero (80 bytes en total, todos en little-endian excepto indicado):
  [4]  version       – versión del bloque
  [32] prev_hash     – hash del bloque anterior (big-endian en el JSON, invertir)
  [32] merkle_root   – raíz del árbol de Merkle de las transacciones
  [4]  timestamp     – Unix timestamp (little-endian)
  [4]  bits          – target compacto
  [4]  nonce         – número arbitrario ajustado por el minero

Verificación:
  hash = SHA256(SHA256(cabecero_80_bytes))    ← doble SHA-256
  PoW válido ⟺ hash < target(bits)
"""

import hashlib
import struct
import time as _time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import get_block, get_tip_hash


# ── Funciones criptográficas ───────────────────────────────────────────────────

def bits_to_target(bits_value: int | str) -> int:
    """
    Convierte el campo 'bits' al target de 256 bits.
    Acepta un entero o un string hexadecimal.
    """
    bits = int(bits_value, 16) if isinstance(bits_value, str) else int(bits_value)
    exp  = bits >> 24
    coef = bits & 0x00FFFFFF
    return coef * (2 ** (8 * (exp - 3)))


def build_block_header(block: dict) -> bytes:
    """
    Serializa el cabecero de 80 bytes en el formato exacto usado por Bitcoin.

    Atención al byte order:
      - version, timestamp, bits, nonce → little-endian (pack '<I')
      - prev_hash, merkle_root          → los hashes en el JSON son big-endian;
                                          se invierten con [::-1] para obtener
                                          el orden interno de Bitcoin (little-endian).
    """
    version     = struct.pack("<I", block["version"])                          # 4 bytes LE
    prev_hash   = bytes.fromhex(block["previousblockhash"])[::-1]             # 32 bytes → invertir
    merkle_root = bytes.fromhex(block["merkle_root"])[::-1]                   # 32 bytes → invertir
    timestamp   = struct.pack("<I", block["timestamp"])                        # 4 bytes LE
    bits_value  = block["bits"]
    bits_int    = int(bits_value, 16) if isinstance(bits_value, str) else int(bits_value)
    bits        = struct.pack("<I", bits_int)                                  # 4 bytes LE
    nonce       = struct.pack("<I", block["nonce"])                            # 4 bytes LE

    header = version + prev_hash + merkle_root + timestamp + bits + nonce
    assert len(header) == 80, f"Cabecero debe ser 80 bytes, got {len(header)}"
    return header


def double_sha256(data: bytes) -> bytes:
    """SHA256(SHA256(data)) — el hash usado en Bitcoin."""
    first  = hashlib.sha256(data).digest()
    second = hashlib.sha256(first).digest()
    return second


def count_leading_zero_bits(hash_bytes: bytes) -> int:
    """Cuenta los bits cero iniciales en un hash de 32 bytes."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            # bin() da '0b101...' → ignorar los 2 primeros caracteres
            count += 8 - len(bin(byte)) + 2
            break
    return count


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render(tip_hash: str | None = None) -> dict:
    """Renderiza el módulo M2. Devuelve el bloque analizado."""
    st.header("🔍 M2 — Block Header Analyzer")

    # Carga del bloque
    if tip_hash is None:
        with st.spinner("Obteniendo hash del último bloque…"):
            try:
                tip_hash = get_tip_hash()
            except Exception as exc:
                st.error(f"Error al obtener el tip hash: {exc}")
                return {}

    with st.spinner(f"Cargando bloque {tip_hash[:16]}…"):
        try:
            block = get_block(tip_hash)
        except Exception as exc:
            st.error(f"Error al obtener el bloque: {exc}")
            return {}

    # ── Tabla: 6 campos del cabecero ──────────────────────────────────────────
    st.subheader("Estructura del cabecero (80 bytes)")

    block_time = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)
    fields = {
        "Campo":        ["version", "prev_hash (32 B)", "merkle_root (32 B)", "timestamp", "bits", "nonce"],
        "Tamaño":       ["4 bytes", "32 bytes", "32 bytes", "4 bytes", "4 bytes", "4 bytes"],
        "Valor":        [
            hex(block["version"]),
            block["previousblockhash"],
            block["merkle_root"],
            f'{block["timestamp"]} ({block_time.strftime("%Y-%m-%d %H:%M:%S UTC")})',
            f'{block["bits"]} (0x{int(block["bits"]):08x})',
            f'{block["nonce"]:,}',
        ],
        "Descripción":  [
            "Versión del protocolo Bitcoin",
            "Hash del bloque anterior — crea la cadena",
            "Raíz del árbol de Merkle de todas las txs",
            "Momento en que el minero cerró el bloque",
            "Target compacto (codifica dificultad)",
            "Número ajustado por el minero hasta obtener hash válido",
        ],
    }
    st.dataframe(pd.DataFrame(fields), use_container_width=True, hide_index=True)

    # ── Verificación PoW con hashlib ──────────────────────────────────────────
    st.subheader("Verificación local del Proof of Work")

    try:
        header_bytes = build_block_header(block)
        computed_hash_le = double_sha256(header_bytes)          # little-endian (interno BTC)
        computed_hash_be = computed_hash_le[::-1]               # big-endian (visualización)
        computed_hash_hex = computed_hash_be.hex()

        target = bits_to_target(block["bits"])
        target_hex = f"{target:064x}"

        hash_int = int(computed_hash_hex, 16)
        pow_valid = hash_int < target

        leading_zeros = count_leading_zero_bits(computed_hash_le)  # se cuenta en LE

        # Comparación visual
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Hash calculado localmente (hashlib):**")
            st.code(computed_hash_hex, language=None)
        with col2:
            st.markdown("**Hash reportado por la API:**")
            st.code(block["id"], language=None)

        match = computed_hash_hex == block["id"]
        if match:
            st.success(" El hash calculado localmente **coincide** con el de la API.")
        else:
            st.error(" Los hashes no coinciden — revisa el byte order.")

        # Resultado PoW
        st.markdown("---")
        st.markdown("**Comprobación PoW:**")
        st.code(
            f"Hash    = {computed_hash_hex}\n"
            f"Target  = {target_hex}\n"
            f"¿Hash < Target? {'SÍ  — Proof of Work VÁLIDO' if pow_valid else 'NO '}"
        )

        col_a, col_b = st.columns(2)
        col_a.metric("Bits cero iniciales en el hash", leading_zeros)
        col_b.metric("Nonce (intentos del minero)", f"{block['nonce']:,}")

        # Visualización del hash con ceros marcados
        st.markdown("**Hash con ceros iniciales destacados:**")
        zero_count = len(computed_hash_hex) - len(computed_hash_hex.lstrip("0"))
        highlighted = (
            f"<span style='background:#ff6b6b;color:white;padding:2px 0'>"
            f"{'0' * zero_count}</span>"
            f"{computed_hash_hex[zero_count:]}"
        )
        st.markdown(
            f"<code style='font-size:12px;word-break:break-all'>{highlighted}</code>",
            unsafe_allow_html=True,
        )

        st.info(
            "💡 **Cómo funciona el Proof of Work:** El minero modifica el **nonce** (4 bytes = "
            f"hasta 4.294.967.296 combinaciones) y recalcula SHA256(SHA256(cabecero)) en cada intento. "
            "Para este bloque, el nonce ganador fue "
            f"**{block['nonce']:,}**. El hash resultante tiene **{leading_zeros} bits cero** iniciales, "
            "lo que lo coloca por debajo del target. La dificultad actual implica que la red "
            "realizó del orden de 2⁷² intentos para encontrarlo."
        )

    except Exception as exc:
        st.error(f"Error durante la verificación PoW: {exc}")

    return block
