# """Starter file for module M4."""

# import streamlit as st


# def render() -> None:
#     """Render the M4 panel."""
#     st.header("M4 - AI Component")
#     st.info("Use this module for your AI idea.")

#     st.subheader("Suggested steps")
#     st.markdown(
#         """
#         1. Choose one AI approach.
#         2. Explain it in the README.
#         3. Show the result in this tab.
#         """
#     )
"""
m4_ai_anomaly.py
----------------
Módulo M4 — AI Component: Anomaly Detector.

Modelo elegido: Detector estadístico de anomalías en tiempos de llegada entre bloques.

Justificación:
  - El proceso de minado es un proceso de Poisson homogéneo (intentos independientes,
    tasa constante). Por ello, el tiempo entre bloques sigue una distribución
    Exponencial con media μ = 600 s (λ = 1/600).
  - Cualquier desviación significativa puede indicar: comportamiento de pools de
    minería, fluctuaciones repentinas de hash rate, o ataques de selfish mining.
  - No requiere datos etiquetados (aprendizaje no supervisado), lo que es apropiado
    dado que no disponemos de un conjunto de entrenamiento con anomalías conocidas.

Método:
  1. Ajustar Exp(λ) a los tiempos observados con scipy (MLE).
  2. Calcular el p-valor de cada observación: P(T ≤ t | Exp(λ)).
     - P muy bajo  (t muy pequeño) → bloque llegó extremadamente rápido (sospechoso).
     - P muy alto  (t muy grande)  → bloque tardó muchísimo (sospechoso).
  3. Umbral: anomalía si p < alpha/2 o p > 1 - alpha/2  (test bilateral, α = 0.01).

Métricas de evaluación:
  - Test de Kolmogorov-Smirnov (KS): contrasta si los datos siguen Exp(λ̂).
  - Tasa de anomalías detectadas.
  - MAE entre la media muestral y μ = 600 s.
  - Visualización: scatter de tiempos con anomalías marcadas + CDF comparativa.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from api.blockchain_client import get_recent_blocks


# ── Constantes ─────────────────────────────────────────────────────────────────
ALPHA = 0.01           # nivel de significancia (bilateral → cada cola 0.5%)
TARGET_MEAN = 600.0    # media teórica en segundos


# ── Modelo ─────────────────────────────────────────────────────────────────────

def fit_exponential(inter_times: list[float]) -> float:
    """
    Ajusta Exp(λ) a los datos por Máxima Verosimilitud.
    Para la distribución exponencial, el estimador MLE es λ̂ = 1 / media_muestral.
    Devuelve λ̂.
    """
    mean = np.mean(inter_times)
    return 1.0 / mean  # MLE para Exp


def detect_anomalies(inter_times: list[float], lam: float, alpha: float = ALPHA) -> list[bool]:
    """
    Marca como anomalía cualquier tiempo cuyo p-valor bilateral sea < alpha.
    p_bilateral(t) = 2 × min(CDF(t), 1 - CDF(t))
    """
    anomalies = []
    for t in inter_times:
        cdf = 1 - np.exp(-lam * t)          # CDF de Exp(λ) en t
        p_bilateral = 2 * min(cdf, 1 - cdf)
        anomalies.append(p_bilateral < alpha)
    return anomalies


def ks_test(inter_times: list[float], lam: float) -> tuple[float, float]:
    """
    Test de Kolmogorov-Smirnov contra Exp(λ̂).
    Devuelve (estadístico D, p-valor).
    H0: los datos provienen de Exp(λ).
    """
    result = stats.kstest(inter_times, "expon", args=(0, 1 / lam))
    return result.statistic, result.pvalue


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render(blocks: list[dict] | None = None) -> None:
    """Renderiza el módulo M4."""
    st.header(" M4 — AI Component: Anomaly Detector")

    st.markdown(
        """
        **Modelo:** Detector de anomalías basado en distribución exponencial (no supervisado).

        **Hipótesis:** Los tiempos entre bloques de Bitcoin siguen `Exp(λ = 1/600)` bajo
        condiciones normales de minería. Bloques cuyo tiempo de llegada se aleja
        significativamente del modelo teórico son marcados como **anomalías** (α = 1%).

        **Caso de uso:** Detectar comportamientos atípicos que pueden indicar actividad
        coordinada de pools de minería o fluctuaciones bruscas del hash rate global.
        """
    )

    # Carga de datos
    if blocks is None:
        with st.spinner("Obteniendo datos de los últimos 200 bloques para el modelo…"):
            try:
                blocks = get_recent_blocks(200)
            except Exception as exc:
                st.error(f"Error al obtener bloques: {exc}")
                return

    # Calcular tiempos entre bloques
    timestamps = sorted([b["timestamp"] for b in blocks], reverse=True)
    inter_times_raw = [
        timestamps[i] - timestamps[i + 1]
        for i in range(len(timestamps) - 1)
    ]
    # Filtrar tiempos improbables (datos corruptos o reorgs)
    inter_times = [t for t in inter_times_raw if 30 <= t <= 7200]
    heights = sorted([b["height"] for b in blocks])[1:]  # alinear con inter_times

    if len(inter_times) < 10:
        st.warning("Datos insuficientes para ajustar el modelo.")
        return

    # ── Ajuste del modelo ──────────────────────────────────────────────────────
    lam_hat    = fit_exponential(inter_times)
    mean_hat   = 1 / lam_hat
    anomalies  = detect_anomalies(inter_times, lam_hat)
    ks_stat, ks_pval = ks_test(inter_times, lam_hat)

    mae = abs(mean_hat - TARGET_MEAN)
    anomaly_rate = sum(anomalies) / len(anomalies) * 100

    # ── Métricas ───────────────────────────────────────────────────────────────
    st.subheader("Métricas del modelo")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("λ̂ ajustado (MLE)",   f"{lam_hat:.5f} s⁻¹")
    col2.metric("Media muestral",      f"{mean_hat:.1f} s",
                delta=f"{mean_hat - TARGET_MEAN:+.1f} s vs target")
    col3.metric("MAE vs μ=600 s",      f"{mae:.1f} s")
    col4.metric("Tasa de anomalías",   f"{anomaly_rate:.1f}%")

    # ── Test KS ───────────────────────────────────────────────────────────────
    st.subheader("Test de Kolmogorov-Smirnov (bondad de ajuste)")
    ks_df = pd.DataFrame({
        "Estadístico":  ["KS statistic (D)", "p-valor", "Resultado H0"],
        "Valor":        [
            f"{ks_stat:.4f}",
            f"{ks_pval:.4f}",
            "No rechazada  — datos compatibles con Exp(λ̂)"
            if ks_pval > 0.05
            else "Rechazada  — los datos se desvían de Exp(λ̂)",
        ],
    })
    st.dataframe(ks_df, use_container_width=True, hide_index=True)
    st.caption("H₀: los tiempos entre bloques siguen Exp(λ̂). Se rechaza si p < 0.05.")

    # ── Scatter: tiempos con anomalías marcadas ────────────────────────────────
    st.subheader("Tiempos entre bloques con anomalías detectadas")

    # Alinear heights (pueden ser más largos que inter_times tras el filtro)
    min_len = min(len(inter_times), len(heights))
    inter_arr = np.array(inter_times[:min_len])
    height_arr = np.array(heights[:min_len])
    anom_arr   = np.array(anomalies[:min_len])

    normal_mask = ~anom_arr
    anom_mask   = anom_arr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=height_arr[normal_mask], y=inter_arr[normal_mask],
        mode="markers", name="Normal",
        marker=dict(color="steelblue", size=6, opacity=0.7),
        hovertemplate="Bloque %{x}<br>Δt = %{y:.0f} s<extra></extra>",
    ))
    if anom_mask.any():
        fig.add_trace(go.Scatter(
            x=height_arr[anom_mask], y=inter_arr[anom_mask],
            mode="markers", name=" Anomalía",
            marker=dict(color="red", size=10, symbol="x"),
            hovertemplate="Bloque %{x}<br>Δt = %{y:.0f} s — ANOMALÍA<extra></extra>",
        ))
    fig.add_hline(y=TARGET_MEAN, line_dash="dash", line_color="orange",
                  annotation_text="Target 600 s")
    fig.add_hline(y=mean_hat, line_dash="dot", line_color="green",
                  annotation_text=f"Media muestral {mean_hat:.0f} s")

    # Umbral de anomalía (bilateral)
    lower_threshold = stats.expon.ppf(ALPHA / 2,    scale=1 / lam_hat)
    upper_threshold = stats.expon.ppf(1 - ALPHA / 2, scale=1 / lam_hat)
    fig.add_hrect(y0=0, y1=lower_threshold, fillcolor="red", opacity=0.07,
                  line_width=0, annotation_text="Zona anómala (muy rápido)")
    fig.add_hrect(y0=upper_threshold, y1=max(inter_arr) * 1.1,
                  fillcolor="red", opacity=0.07, line_width=0,
                  annotation_text="Zona anómala (muy lento)")

    fig.update_layout(
        title=f"Δt entre bloques — anomalías al nivel α = {ALPHA} (bilateral)",
        xaxis_title="Altura del bloque",
        yaxis_title="Tiempo entre bloques (s)",
        legend_title="Tipo",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CDF empírica vs teórica ────────────────────────────────────────────────
    st.subheader("CDF empírica vs CDF teórica Exp(λ̂)")
    sorted_times = np.sort(inter_times)
    empirical_cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    theoretical_cdf = 1 - np.exp(-lam_hat * sorted_times)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sorted_times, y=empirical_cdf,
        mode="lines", name="CDF empírica",
        line=dict(color="steelblue", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=sorted_times, y=theoretical_cdf,
        mode="lines", name=f"CDF Exp(λ̂={lam_hat:.5f})",
        line=dict(color="red", width=2, dash="dash"),
    ))
    fig2.update_layout(
        title="Comparación CDF empírica vs teórica",
        xaxis_title="Tiempo entre bloques (s)",
        yaxis_title="Probabilidad acumulada",
        legend_title="Serie",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Tabla de anomalías detectadas ─────────────────────────────────────────
    anomaly_indices = [i for i, a in enumerate(anomalies[:min_len]) if a]
    if anomaly_indices:
        st.subheader(f"Bloques anómalos detectados ({len(anomaly_indices)} total)")
        anom_data = []
        for i in anomaly_indices:
            t = inter_times[i]
            cdf = 1 - np.exp(-lam_hat * t)
            p = 2 * min(cdf, 1 - cdf)
            tipo = "Muy rápido " if t < lower_threshold else "Muy lento "
            anom_data.append({
                "Bloque":    heights[i] if i < len(heights) else "?",
                "Δt (s)":   f"{t:.0f}",
                "Tipo":      tipo,
                "p-valor":   f"{p:.4f}",
            })
        st.dataframe(pd.DataFrame(anom_data), use_container_width=True, hide_index=True)
    else:
        st.success("No se detectaron anomalías en la muestra actual.")

    # ── Limitaciones ───────────────────────────────────────────────────────────
    with st.expander(" Limitaciones del modelo y trabajo futuro"):
        st.markdown(
            """
            **Limitaciones:**
            - El modelo asume λ constante en toda la muestra. En la práctica, el hash rate
              de la red fluctúa (entrada/salida de mineros, halving events).
            - Con sólo ~200 bloques (~1.4 días de datos) la muestra es pequeña para
              detectar anomalías sutiles con alta precisión.
            - No distingue entre anomalías causadas por factores externos (apagón)
              vs. comportamiento estratégico de mineros (selfish mining).

            **Trabajo futuro:**
            - Usar ventana deslizante para detectar cambios en λ (CUSUM, PELT).
            - Añadir features adicionales (tamaño del bloque, fee medio) para un
              modelo supervisado si se dispone de etiquetas.
            - Comparar con un modelo LSTM para detectar patrones secuenciales.
            """
        )