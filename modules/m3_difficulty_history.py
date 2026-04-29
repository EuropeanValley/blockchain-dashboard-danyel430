# """Starter file for module M3."""

# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from api.blockchain_client import get_difficulty_history


# def render() -> None:
#     """Render the M3 panel."""
#     st.header("M3 - Difficulty History")
#     st.write("Use this module to plot the history of Bitcoin difficulty.")

#     n_points = st.slider("Number of data points", min_value=10, max_value=365, value=100, key="m3_n")

#     if st.button("Load difficulty chart", key="m3_load"):
#         with st.spinner("Fetching data..."):
#             try:
#                 values = get_difficulty_history(n_points)
#                 df = pd.DataFrame(values)
#                 df["x"] = pd.to_datetime(df["x"], unit="s")
#                 df = df.rename(columns={"x": "Date", "y": "Difficulty"})

#                 fig = px.line(df, x="Date", y="Difficulty", title="Bitcoin Mining Difficulty")
#                 st.plotly_chart(fig, use_container_width=True)
#             except Exception as exc:
#                 st.error(f"Error loading chart: {exc}")
#     else:
#         st.info("Click Load difficulty chart to display the chart.")

"""
m3_difficulty_history.py
------------------------
Módulo M3 — Difficulty History.

Muestra la evolución de la dificultad a lo largo de los últimos ~2 años:
  - Gráfico de dificultad vs tiempo.
  - Marcadores en cada ajuste (cada 2016 bloques ≈ 2 semanas).
  - Ratio tiempo_real / 600 s para cada periodo de ajuste.

Fórmula de ajuste (Section 6.1 de las notas):
  new_difficulty = old_difficulty × (tiempo_real_2016_bloques / 1_209_600)
  donde 1_209_600 s = 2016 × 600 s (target teórico para 2016 bloques)

Los datos provienen de Blockchain.info /charts/difficulty que devuelve
series temporales de dificultad pre-calculadas.
"""
# """Starter file for module M3."""

# import pandas as pd
# import plotly.express as px
# import streamlit as st

# from api.blockchain_client import get_difficulty_history


# def render() -> None:
#     """Render the M3 panel."""
#     st.header("M3 - Difficulty History")
#     st.write("Use this module to plot the history of Bitcoin difficulty.")

#     n_points = st.slider("Number of data points", min_value=10, max_value=365, value=100, key="m3_n")

#     if st.button("Load difficulty chart", key="m3_load"):
#         with st.spinner("Fetching data..."):
#             try:
#                 values = get_difficulty_history(n_points)
#                 df = pd.DataFrame(values)
#                 df["x"] = pd.to_datetime(df["x"], unit="s")
#                 df = df.rename(columns={"x": "Date", "y": "Difficulty"})

#                 fig = px.line(df, x="Date", y="Difficulty", title="Bitcoin Mining Difficulty")
#                 st.plotly_chart(fig, use_container_width=True)
#             except Exception as exc:
#                 st.error(f"Error loading chart: {exc}")
#     else:
#         st.info("Click Load difficulty chart to display the chart.")

"""
m3_difficulty_history.py
------------------------
Módulo M3 — Difficulty History.

Muestra la evolución de la dificultad a lo largo de los últimos ~2 años:
  - Gráfico de dificultad vs tiempo.
  - Marcadores en cada ajuste (cada 2016 bloques ≈ 2 semanas).
  - Ratio tiempo_real / 600 s para cada periodo de ajuste.

Fórmula de ajuste (Section 6.1 de las notas):
  new_difficulty = old_difficulty × (tiempo_real_2016_bloques / 1_209_600)
  donde 1_209_600 s = 2016 × 600 s (target teórico para 2016 bloques)

Los datos provienen de Blockchain.info /charts/difficulty que devuelve
series temporales de dificultad pre-calculadas.
"""

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import get_difficulty_history


# ── Constantes ─────────────────────────────────────────────────────────────────
ADJUSTMENT_PERIOD_BLOCKS = 2016
TARGET_SECS_PER_PERIOD   = ADJUSTMENT_PERIOD_BLOCKS * 600  # 1_209_600 s ≈ 14 días


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render() -> None:
    """Renderiza el módulo M3."""
    st.header(" M3 — Difficulty History (últimos 2 años)")

    with st.spinner("Descargando historial de dificultad…"):
        try:
            raw = get_difficulty_history()
        except Exception as exc:
            st.error(f"Error al obtener historial: {exc}")
            return

    # ── Procesar datos ─────────────────────────────────────────────────────────
    values = raw.get("values", [])
    if not values:
        st.warning("No se obtuvieron datos del historial.")
        return

    # La API devuelve lista de dicts {"x": unix_ts, "y": difficulty}
    if isinstance(values[0], dict):
        df = pd.DataFrame(values).rename(columns={"x": "timestamp", "y": "difficulty"})
    else:
        df = pd.DataFrame(values, columns=["timestamp", "difficulty"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Calcular ajustes: tomamos puntos donde la dificultad cambia (cada ~2 semanas)
    # La API de Blockchain.info ya devuelve un punto por ajuste aproximadamente.
    df["pct_change"] = df["difficulty"].pct_change() * 100
    adjustment_rows = df[df["pct_change"].abs() > 0.1].copy()  # filtro de ruido

    # Ratio tiempo real vs target por periodo
    df["time_delta_s"] = df["timestamp"].diff()
    # Filtramos deltas coherentes (entre 1 y 30 días en segundos)
    valid_delta = df["time_delta_s"].between(86_400, 2_592_000)
    df["ratio"] = None
    df.loc[valid_delta, "ratio"] = df.loc[valid_delta, "time_delta_s"] / TARGET_SECS_PER_PERIOD

    # ── Gráfico principal: dificultad vs tiempo ────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["difficulty"],
        mode="lines",
        name="Dificultad",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Dificultad: %{y:,.0f}<extra></extra>",
    ))

    # Marcadores de ajuste (positivo = naranja, negativo = rojo)
    adj_up   = adjustment_rows[adjustment_rows["pct_change"] > 0]
    adj_down = adjustment_rows[adjustment_rows["pct_change"] < 0]

    fig.add_trace(go.Scatter(
        x=adj_up["date"], y=adj_up["difficulty"],
        mode="markers", name="Ajuste (+)",
        marker=dict(color="orange", size=9, symbol="triangle-up"),
        hovertemplate=(
            "<b>Ajuste positivo</b><br>Fecha: %{x|%Y-%m-%d}<br>"
            "Dificultad: %{y:,.0f}<br>Cambio: +"
            + adj_up["pct_change"].round(2).astype(str) + "%<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=adj_down["date"], y=adj_down["difficulty"],
        mode="markers", name="Ajuste (−)",
        marker=dict(color="red", size=9, symbol="triangle-down"),
        hovertemplate=(
            "<b>Ajuste negativo</b><br>Fecha: %{x|%Y-%m-%d}<br>"
            "Dificultad: %{y:,.0f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Evolución de la dificultad de minería de Bitcoin (últimos 2 años)",
        xaxis_title="Fecha",
        yaxis_title="Dificultad",
        legend_title="Serie",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabla de ajustes recientes ─────────────────────────────────────────────
    st.subheader("Últimos ajustes de dificultad")
    recent_adj = adjustment_rows.tail(15)[["date", "difficulty", "pct_change"]].copy()
    recent_adj.columns = ["Fecha (UTC)", "Dificultad", "Cambio (%)"]
    recent_adj["Fecha (UTC)"] = recent_adj["Fecha (UTC)"].dt.strftime("%Y-%m-%d")
    recent_adj["Dificultad"]  = recent_adj["Dificultad"].apply(lambda x: f"{x:,.0f}")
    recent_adj["Cambio (%)"]  = recent_adj["Cambio (%)"].apply(lambda x: f"{x:+.2f}%")
    st.dataframe(recent_adj, use_container_width=True, hide_index=True)

    # ── Ratio tiempo real / 600 s ──────────────────────────────────────────────
    ratio_df = df[valid_delta][["date", "ratio"]].dropna().tail(20).copy()
    if not ratio_df.empty:
        st.subheader("Ratio tiempo real / target (600 s) por periodo de ajuste")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=ratio_df["date"], y=ratio_df["ratio"],
            marker_color=[
                "green" if r < 1 else "red" for r in ratio_df["ratio"]
            ],
            name="Ratio",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ratio: %{y:.3f}<extra></extra>",
        ))
        fig2.add_hline(
            y=1.0, line_dash="dash", line_color="white",
            annotation_text="Target = 1.0 (600 s exactos)",
            annotation_position="top right",
        )
        fig2.update_layout(
            title="Ratio tiempo_real / 1_209_600 s por periodo (verde = bloques más rápidos que target)",
            xaxis_title="Fecha del ajuste",
            yaxis_title="Ratio",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Explicación de la fórmula ──────────────────────────────────────────────
    st.info(
        " **Fórmula de ajuste (Section 6.1):**\n\n"
        "```\n"
        "nueva_dificultad = dificultad_anterior × (T_real / T_target)\n"
        "T_target = 2016 × 600 s = 1 209 600 s ≈ 14 días\n"
        "```\n\n"
        "Si los mineros encontraron los 2016 bloques en menos de 14 días "
        "(ratio < 1), la dificultad **sube** para volver a los 10 min/bloque. "
        "Si tardaron más (ratio > 1), la dificultad **baja**. "
        "El ajuste está limitado a un factor máximo de ×4 ó ×1/4 por periodo."
    )