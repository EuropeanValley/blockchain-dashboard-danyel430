# """Starter file for module M1."""

# import streamlit as st

# from api.blockchain_client import get_latest_block


# def render() -> None:
#     """Render the M1 panel."""
#     st.header("M1 - Proof of Work Monitor")
#     st.write("Use this module to show live Bitcoin mining data.")
#     st.write("Suggested ideas:")
#     st.write("- latest block height")
#     st.write("- block hash")
#     st.write("- difficulty")
#     st.write("- nonce")
#     st.write("- number of transactions")

#     if st.button("Fetch latest block", key="m1_fetch"):
#         with st.spinner("Fetching data..."):
#             try:
#                 block = get_latest_block()
#                 st.success(f"Block height: {block.get('height')}")
#                 st.json(block)
#             except Exception as exc:
#                 st.error(f"Error fetching data: {exc}")
#     else:
#         st.info("Click the button to test the API connection.")

"""
m1_pow_monitor.py
-----------------
Módulo M1 — Proof of Work Monitor.

Muestra en tiempo real:
  - Dificultad actual y su representación visual como umbral de ceros en SHA-256.
  - Distribución de tiempos entre los últimos N bloques (histograma).
  - Estimación de la tasa de hash actual de la red.

Conceptos criptográficos aplicados (Topic 7):
  - El campo 'bits' del bloque codifica el TARGET de 256 bits.
  - difficulty = genesis_target / current_target.
  - hashrate ≈ difficulty × 2³² / 600  (bloques/s × hashes/bloque).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import get_recent_blocks


# ── Funciones criptográficas ───────────────────────────────────────────────────

def bits_to_target(bits_value: int | str) -> int:
    """
    Convierte el campo 'bits' compacto al target de 256 bits.
    Formato: 0xAABBBBBB  →  target = 0xBBBBBB × 2^(8×(0xAA − 3))
    """
    bits = int(bits_value, 16) if isinstance(bits_value, str) else int(bits_value)
    exponent   = bits >> 24          # byte más significativo
    coefficient = bits & 0x00FFFFFF  # 3 bytes menos significativos
    return coefficient * (2 ** (8 * (exponent - 3)))


def target_to_difficulty(target: int) -> float:
    """
    difficulty = genesis_target / current_target
    El genesis_target corresponde a bits = 0x1d00ffff.
    """
    genesis_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return genesis_target / target


def estimate_hashrate(difficulty: float) -> float:
    """
    Hashrate estimado en hashes/segundo.
    Un minero tarda, de media, difficulty × 2³² intentos para encontrar un bloque.
    Con un bloque cada 600 s:  H = D × 2³² / 600
    """
    return difficulty * (2 ** 32) / 600


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render(blocks: list[dict] | None = None) -> list[dict]:
    """
    Renderiza el módulo M1 en la pestaña activa.
    Devuelve la lista de bloques para reutilizarla en otros módulos.
    """
    st.header("⛏️ M1 — Proof of Work Monitor")

    # Carga de datos
    if blocks is None:
        with st.spinner("Obteniendo los últimos 100 bloques…"):
            try:
                blocks = get_recent_blocks(100)
            except Exception as exc:
                st.error(f"Error al obtener datos: {exc}")
                return []

    latest = blocks[0]
    bits_value = latest["bits"]
    target     = bits_to_target(bits_value)
    difficulty = target_to_difficulty(target)
    hashrate   = estimate_hashrate(difficulty)

    # ── Métricas principales ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Altura actual",     f"{latest['height']:,}")
    col2.metric("Dificultad actual", f"{difficulty:,.0f}")
    col3.metric("Hash rate estimado", f"{hashrate / 1e18:.2f} EH/s")
    col4.metric("Transacciones en último bloque", f"{latest.get('tx_count', 'N/A'):,}")

    # ── Target como umbral de ceros ────────────────────────────────────────────
    st.subheader("Target SHA-256 como umbral de ceros")
    target_hex = f"{target:064x}"
    block_hash = latest["id"]
    leading_hex_zeros = len(target_hex) - len(target_hex.lstrip("0"))
    leading_bit_zeros = leading_hex_zeros * 4  # 1 hex digit = 4 bits

    st.code(f"Target      : {target_hex}\nHash actual : {block_hash}")
    st.info(
        f"El target exige que los **{leading_hex_zeros} primeros dígitos hex** "
        f"(≈ {leading_bit_zeros} bits) sean cero en el espacio SHA-256 de 256 bits. "
        f"La probabilidad de lograrlo en un intento es **1 / {2**leading_bit_zeros:,}**."
    )

    # Barra de progreso visual: % del espacio SHA-256 que es válido
    valid_fraction = target / (2**256)
    st.progress(
        min(valid_fraction * 1e12, 1.0),
        text=f"Fracción del espacio hash que es válida: {valid_fraction:.2e}",
    )

    # ── Distribución de tiempos entre bloques ─────────────────────────────────
    st.subheader("Distribución de tiempos entre bloques (últimos 100 bloques)")

    timestamps = sorted([b["timestamp"] for b in blocks], reverse=True)
    inter_times = [
        timestamps[i] - timestamps[i + 1]
        for i in range(len(timestamps) - 1)
        if 0 < timestamps[i] - timestamps[i + 1] < 7200  # filtra datos corruptos
    ]

    if inter_times:
        mean_time = np.mean(inter_times)
        median_time = np.median(inter_times)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tiempo medio entre bloques", f"{mean_time:.0f} s")
        col_b.metric("Mediana", f"{median_time:.0f} s")
        col_c.metric("Target teórico", "600 s")

        # Histograma con curva exponencial teórica superpuesta
        lam = 1 / 600  # parámetro de la distribución exponencial
        x_range = np.linspace(0, max(inter_times), 300)
        pdf_values = lam * np.exp(-lam * x_range) * len(inter_times) * (max(inter_times) / 30)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=inter_times, nbinsx=30, name="Observado",
            marker_color="steelblue", opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf_values,
            mode="lines", name="Exp(λ=1/600) teórica",
            line=dict(color="red", width=2, dash="dash"),
        ))
        fig.add_vline(
            x=600, line_dash="dot", line_color="orange",
            annotation_text="Target: 600 s", annotation_position="top right",
        )
        fig.update_layout(
            title="Tiempos entre bloques — distribución esperada: Exponencial(λ = 1/600)",
            xaxis_title="Segundos entre bloques",
            yaxis_title="Frecuencia",
            legend_title="Serie",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 **Teoría (Topic 7):** Los mineros realizan intentos de hash de forma independiente "
            "y continua — proceso de Poisson. Por ello, el tiempo hasta encontrar un bloque válido "
            "sigue una distribución **Exponencial con media 600 s** (λ = 1/600 ≈ 0.00167). "
            "La falta de memoria de esta distribución significa que el tiempo restante hasta el "
            "siguiente bloque siempre es, en expectativa, 600 s, independientemente de cuánto se lleve esperando."
        )

    return blocks
