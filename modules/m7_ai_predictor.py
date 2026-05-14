"""
m7_ai_predictor.py
------------------
Módulo M7 — Second AI Approach: Predictor de dificultad.

Contraste con M4:
  M4 — Detector de anomalías (no supervisado, un solo bloque a la vez).
  M7 — Predictor de series temporales (supervisado, predice el próximo ajuste).

Modelo: Regresión lineal con features de ventana deslizante.
  - Variable objetivo (y): valor de dificultad en el siguiente ajuste.
  - Features (X): últimos k valores de dificultad + sus ratios de cambio.
  - Evaluación: MAE, RMSE, MAPE y R² sobre un conjunto de test (20% final).

Por qué regresión lineal sobre LSTM o Prophet:
  - La dificultad de Bitcoin muestra una tendencia casi log-lineal a largo plazo.
  - Con ~50 puntos de datos (2 años de ajustes quincenales) un modelo complejo
    se sobreajustaría fácilmente. La regresión lineal es interpretable y robusta.
  - Se puede comparar con la fórmula determinista real de ajuste para evaluar
    si el modelo "aprende" la lógica de la red.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from api.blockchain_client import get_difficulty_history


# ── Preparación de datos ───────────────────────────────────────────────────────

def load_difficulty_series() -> pd.DataFrame:
    """
    Descarga el historial de dificultad y devuelve un DataFrame limpio
    con columnas: date (datetime), difficulty (float).
    """
    raw    = get_difficulty_history()
    values = raw.get("values", [])
    if not values:
        raise ValueError("No se obtuvieron datos de dificultad.")

    if isinstance(values[0], dict):
        df = pd.DataFrame(values).rename(columns={"x": "timestamp", "y": "difficulty"})
    else:
        df = pd.DataFrame(values, columns=["timestamp", "difficulty"])

    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    # Filtrar puntos duplicados o con dificultad 0
    df = df[df["difficulty"] > 0].drop_duplicates("timestamp")
    return df


def build_features(series: np.ndarray, window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de features X e y usando ventana deslizante.
    Features por muestra: [d(t-k), ..., d(t-1), ratio(t-k), ..., ratio(t-1)]
    Target: d(t)
    """
    X, y = [], []
    ratios = np.diff(series) / series[:-1]  # pct_change

    for i in range(window, len(series)):
        feat_levels = series[i - window: i]
        feat_ratios = ratios[i - window: i - 1] if i - window < len(ratios) else ratios[-(window-1):]
        # Asegurar longitud fija
        if len(feat_ratios) < window - 1:
            continue
        X.append(np.concatenate([feat_levels, feat_ratios]))
        y.append(series[i])

    return np.array(X), np.array(y)


# ── Métricas ───────────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render() -> None:
    st.header("🧠 M7 — Second AI Approach: Predictor de Dificultad")

    st.markdown(
        """
        **Contraste con M4:** el módulo M4 detecta anomalías en bloques individuales
        (no supervisado). Este módulo **predice el valor del próximo ajuste de dificultad**
        usando regresión lineal sobre el historial de ajustes (supervisado).

        **¿Por qué regresión lineal?** Con ~50 puntos de datos (ajustes quincenales de
        los últimos 2 años), modelos complejos como LSTM se sobreajustarían. La regresión
        lineal sobre features en escala logarítmica captura la tendencia de crecimiento
        de la dificultad de forma interpretable.
        """
    )

    # ── Cargar datos ───────────────────────────────────────────────────────────
    with st.spinner("Descargando historial de dificultad…"):
        try:
            df = load_difficulty_series()
        except Exception as exc:
            st.error(f"Error al cargar datos: {exc}")
            return

    st.success(f"✅ {len(df)} puntos de datos cargados.")

    # ── Parámetros del modelo ──────────────────────────────────────────────────
    col_p1, col_p2, col_p3 = st.columns(3)
    window    = col_p1.slider("Ventana de features (k ajustes anteriores)", 2, 8, 4)
    test_pct  = col_p2.slider("Porcentaje de test (%)", 10, 30, 20)
    use_log   = col_p3.checkbox("Escala logarítmica", value=True,
                                help="Trabajar en log(dificultad) reduce heterocedasticidad")
    use_ridge = st.checkbox("Usar Ridge (regularización L2) en lugar de OLS",
                            value=True)

    if st.button("▶️ Entrenar y evaluar modelo"):
        series = df["difficulty"].values
        if use_log:
            series = np.log(series)

        X, y = build_features(series, window=window)

        if len(X) < 10:
            st.warning("Datos insuficientes para el tamaño de ventana seleccionado.")
            return

        # Split temporal (no aleatorio — series temporales)
        split = int(len(X) * (1 - test_pct / 100))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Normalización
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Entrenamiento
        model = Ridge(alpha=1.0) if use_ridge else LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Deshacer log si aplica
        if use_log:
            y_test_real = np.exp(y_test)
            y_pred_real = np.exp(y_pred)
        else:
            y_test_real = y_test
            y_pred_real = y_pred

        # ── Métricas ───────────────────────────────────────────────────────────
        mae_val  = mean_absolute_error(y_test_real, y_pred_real)
        rmse_val = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mape_val = mape(y_test_real, y_pred_real)
        r2_val   = r2_score(y_test, y_pred)  # R² en el espacio del modelo

        st.subheader("Métricas de evaluación (conjunto de test)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE",  f"{mae_val:,.0f}")
        m2.metric("RMSE", f"{rmse_val:,.0f}")
        m3.metric("MAPE", f"{mape_val:.2f}%")
        m4.metric("R²",   f"{r2_val:.4f}")

        st.caption(
            "MAE/RMSE en unidades de dificultad absoluta. "
            "MAPE es el error porcentual medio — más interpretable dado el crecimiento exponencial. "
            "R² calculado en el espacio del modelo (log si aplica)."
        )

        # ── Gráfico: predicciones vs real ──────────────────────────────────────
        st.subheader("Predicciones vs valores reales (conjunto de test)")

        # Fechas correspondientes al test set
        test_dates = df["date"].values[-(len(y_test)):]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_test_real,
            mode="lines+markers", name="Real",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_pred_real,
            mode="lines+markers", name="Predicción",
            line=dict(color="orange", width=2, dash="dash"),
            marker=dict(size=6, symbol="x"),
        ))
        fig.update_layout(
            title="Dificultad real vs predicha (ajustes quincenales)",
            xaxis_title="Fecha",
            yaxis_title="Dificultad",
            legend_title="Serie",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Residuos ───────────────────────────────────────────────────────────
        st.subheader("Distribución de residuos")
        residuals = y_test_real - y_pred_real
        fig2 = go.Figure(go.Histogram(
            x=residuals, nbinsx=15,
            marker_color="steelblue", opacity=0.8,
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text="Residuo = 0")
        fig2.update_layout(
            title="Histograma de residuos (real − predicho)",
            xaxis_title="Residuo",
            yaxis_title="Frecuencia",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Predicción del próximo ajuste ──────────────────────────────────────
        st.subheader("🔮 Predicción del próximo ajuste de dificultad")
        last_series = np.log(df["difficulty"].values) if use_log else df["difficulty"].values
        last_window = last_series[-window:]
        last_ratios = np.diff(last_series)[-(window - 1):]

        if len(last_ratios) == window - 1:
            x_next = np.concatenate([last_window, last_ratios]).reshape(1, -1)
            x_next_scaled = scaler.transform(x_next)
            pred_next = model.predict(x_next_scaled)[0]
            pred_next_real = np.exp(pred_next) if use_log else pred_next

            current_diff = df["difficulty"].values[-1]
            pct_change   = (pred_next_real - current_diff) / current_diff * 100

            col_a, col_b = st.columns(2)
            col_a.metric("Dificultad actual",       f"{current_diff:,.0f}")
            col_b.metric("Próximo ajuste predicho", f"{pred_next_real:,.0f}",
                         delta=f"{pct_change:+.2f}%")

        # ── Comparación con fórmula determinista ──────────────────────────────
        with st.expander("📊 Comparación M4 vs M7"):
            st.markdown(
                """
                | Criterio              | M4 — Anomaly Detector         | M7 — Difficulty Predictor      |
                |-----------------------|-------------------------------|--------------------------------|
                | **Tipo**              | No supervisado                | Supervisado                    |
                | **Granularidad**      | Por bloque (~10 min)          | Por ajuste (~2 semanas)        |
                | **Variable objetivo** | ¿Es este bloque anómalo?      | ¿Cuál será la próxima D?       |
                | **Modelo base**       | Distribución Exponencial      | Regresión lineal (Ridge)       |
                | **Evaluación**        | Test KS, tasa de anomalías    | MAE, RMSE, MAPE, R²            |
                | **Interpretabilidad** | Alta (p-valores)              | Alta (coeficientes lineales)   |
                | **Limitación**        | No predice, solo detecta      | Pocos puntos de entrenamiento  |

                Los dos modelos se complementan: M4 monitoriza en tiempo real el comportamiento
                de bloques individuales, mientras M7 anticipa el próximo ajuste de dificultad
                que afecta a toda la red.
                """
            )