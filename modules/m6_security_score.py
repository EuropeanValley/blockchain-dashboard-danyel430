"""
m6_security_score.py
--------------------
Módulo M6 — Security Score: coste de un ataque del 51%.

Basado en Nakamoto (2008), §11 "Calculations":
  - Un atacante con fracción q del hashrate de la red necesita superar
    a los mineros honestos (fracción p = 1 - q).
  - La probabilidad de que el atacante alcance al nodo honesto desde
    z bloques de retraso es:
        P(z, q) = 1  si q >= p
        P(z, q) = 1 - Σ_{k=0}^{z} [e^{-λ} λ^k / k!] × (1 - (q/p)^(z-k+1))
    aproximada por la fórmula de Nakamoto:
        P ≈ 1 - Φ(z, λ) donde λ = z(q/p)

    En la práctica se usa la fórmula binomial de Poisson:
        P(attack_succeeds | z confirmations) =
            1 - Σ_{k=0}^{z} poisson(k, λ=z*q/p) × (1-(q/p)^(z-k))

Coste del ataque (USD/hora):
  - Para controlar el 51% del hashrate necesitas ~H/2 hashes/s adicionales,
    donde H es el hashrate actual de la red.
  - El hardware más eficiente actualmente son ASICs tipo Antminer S21 Pro:
      ~234 TH/s con ~3500 W de consumo → ~0.015 J/TH
  - Coste eléctrico: ~0.05 USD/kWh (coste industrial medio global).
  - Coste por hora ≈ (H/2) / (234e12) × 3500 W × 0.05 USD/kWh / 1000
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

from api.blockchain_client import get_recent_blocks
from modules.m1_pow_monitor import bits_to_target, target_to_difficulty, estimate_hashrate


# ── Constantes de hardware (Antminer S21 Pro, 2024) ───────────────────────────
ASIC_HASHRATE_THS  = 234          # TH/s por unidad
ASIC_POWER_W       = 3_500        # vatios por unidad
ELECTRICITY_USD_KWH = 0.05        # USD por kWh (coste industrial)


# ── Fórmulas ───────────────────────────────────────────────────────────────────

def attack_cost_usd_per_hour(hashrate_hs: float, attacker_fraction: float = 0.51) -> dict:
    """
    Calcula el coste por hora de controlar `attacker_fraction` del hashrate.
    Devuelve dict con hashrate necesario, número de ASICs, potencia y coste USD/h.
    """
    target_hashrate = hashrate_hs * attacker_fraction
    asic_hashrate_hs = ASIC_HASHRATE_THS * 1e12

    n_asics   = target_hashrate / asic_hashrate_hs
    power_kw  = n_asics * ASIC_POWER_W / 1000
    cost_hour = power_kw * ELECTRICITY_USD_KWH

    return {
        "attacker_hashrate_ehs": target_hashrate / 1e18,
        "n_asics":               n_asics,
        "power_mw":              power_kw / 1000,
        "cost_usd_hour":         cost_hour,
        "cost_usd_day":          cost_hour * 24,
    }


def nakamoto_attack_probability(z: int, q: float) -> float:
    """
    Probabilidad de que un atacante con fracción q del hashrate
    revierta z confirmaciones (Nakamoto 2008, §11).
    Usa la aproximación de Poisson.
    """
    if q >= 0.5:
        return 1.0
    p = 1.0 - q
    lam = z * q / p

    total = 0.0
    for k in range(z + 1):
        pk = poisson.pmf(k, lam)
        total += pk * (1.0 - (q / p) ** (z - k + 1))

    return max(0.0, min(1.0, 1.0 - total))


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render(blocks: list[dict] | None = None) -> None:
    st.header("🛡️ M6 — Security Score: Coste del ataque del 51%")

    st.markdown(
        """
        Estima cuánto costaría atacar la red Bitcoin y visualiza cómo las
        confirmaciones reducen la probabilidad de éxito del atacante.
        Basado en **Nakamoto (2008), §11**.
        """
    )

    # ── Obtener hashrate actual ────────────────────────────────────────────────
    if blocks is None:
        with st.spinner("Obteniendo datos de la red…"):
            try:
                blocks = get_recent_blocks(20)
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

    latest  = blocks[0]
    target  = bits_to_target(latest["bits"])
    diff    = target_to_difficulty(target)
    hashrate_hs = estimate_hashrate(diff)

    col1, col2 = st.columns(2)
    col1.metric("Hash rate de la red", f"{hashrate_hs / 1e18:.2f} EH/s")
    col2.metric("Dificultad actual",   f"{diff:,.0f}")

    # ── Coste del ataque ───────────────────────────────────────────────────────
    st.subheader("Coste estimado de controlar el 51% del hashrate")

    attacker_pct = st.slider(
        "Fracción del hashrate del atacante (%)",
        min_value=1, max_value=60, value=51,
        help="Para revertir transacciones confirmadas se necesita >50%"
    ) / 100

    costs = attack_cost_usd_per_hour(hashrate_hs, attacker_pct)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hashrate necesario",   f"{costs['attacker_hashrate_ehs']:.2f} EH/s")
    c2.metric("ASICs necesarios",     f"{costs['n_asics']:,.0f}")
    c3.metric("Coste eléctrico/hora", f"${costs['cost_usd_hour']:,.0f}")
    c4.metric("Coste eléctrico/día",  f"${costs['cost_usd_day']:,.0f}")

    st.caption(
        f"⚡ Asumiendo Antminer S21 Pro ({ASIC_HASHRATE_THS} TH/s, {ASIC_POWER_W} W) "
        f"a ${ELECTRICITY_USD_KWH}/kWh. No incluye coste de hardware ni operación."
    )

    # ── Probabilidad de ataque vs confirmaciones ───────────────────────────────
    st.subheader("Probabilidad de éxito del atacante vs número de confirmaciones")
    st.markdown("_(Nakamoto 2008 §11 — fórmula de Poisson)_")

    attacker_fractions = [0.10, 0.20, 0.30, 0.40, 0.49]
    confirmations      = list(range(0, 31))

    fig = go.Figure()
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

    for frac, color in zip(attacker_fractions, colors):
        probs = [nakamoto_attack_probability(z, frac) * 100 for z in confirmations]
        fig.add_trace(go.Scatter(
            x=confirmations, y=probs,
            mode="lines+markers",
            name=f"q = {int(frac*100)}% hashrate",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"q={int(frac*100)}%<br>z=%{{x}} conf.<br>P=%{{y:.4f}}%<extra></extra>",
        ))

    # Umbral de seguridad habitual: 6 confirmaciones
    fig.add_vline(
        x=6, line_dash="dash", line_color="white", opacity=0.5,
        annotation_text="6 confirmaciones\n(estándar exchanges)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=0.1, line_dash="dot", line_color="gray", opacity=0.4,
        annotation_text="0.1% prob.", annotation_position="right",
    )
    fig.update_layout(
        title="P(ataque exitoso) en función de las confirmaciones y el hashrate del atacante",
        xaxis_title="Número de confirmaciones (z)",
        yaxis_title="Probabilidad de éxito (%)",
        yaxis_type="log",
        legend_title="Fracción del atacante",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de referencia para 6 confirmaciones ──────────────────────────────
    st.subheader("Tabla de referencia: P(éxito) con 6 confirmaciones")
    table_data = []
    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.49]:
        p = nakamoto_attack_probability(6, frac)
        table_data.append({
            "Hashrate atacante": f"{int(frac*100)}%",
            "P(éxito) con 6 conf.": f"{p*100:.6f}%",
            "Coste/hora (USD)": f"${attack_cost_usd_per_hour(hashrate_hs, frac)['cost_usd_hour']:,.0f}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.info(
        "💡 **Conclusión:** Con 6 confirmaciones, incluso un atacante con el 40% del "
        "hashrate tiene una probabilidad de éxito ínfima. El coste económico hace "
        "que atacar Bitcoin sea rentablemente inviable: el coste operativo supera "
        "ampliamente cualquier beneficio posible de revertir transacciones."
    )

    with st.expander("📚 Referencia: Nakamoto (2008) §11"):
        st.markdown(
            """
            > *"An attacker can only try to change one of his own transactions to take back
            > money he recently spent. [...] We can calculate the probability he ever catches up
            > using a Gambler's ruin problem."*

            La fórmula exacta es un proceso de caminata aleatoria donde el atacante
            necesita ganar z pasos de ventaja a los mineros honestos. La distribución
            de Poisson aproxima el número de bloques que el atacante mina mientras
            la cadena honesta avanza z bloques.

            **Fuente:** Satoshi Nakamoto, *Bitcoin: A Peer-to-Peer Electronic Cash System* (2008).
            https://bitcoin.org/bitcoin.pdf
            """
        )