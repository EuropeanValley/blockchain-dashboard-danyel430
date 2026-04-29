# """Simple Streamlit entry point for the student project."""

# import streamlit as st

# from modules.m1_pow_monitor import render as render_m1
# from modules.m2_block_header import render as render_m2
# from modules.m3_difficulty_history import render as render_m3
# from modules.m4_ai_component import render as render_m4

# st.set_page_config(page_title="Blockchain Dashboard", layout="wide")

# st.title("Blockchain Dashboard")

# tab1, tab2, tab3, tab4 = st.tabs(
#     ["M1 - PoW Monitor", "M2 - Block Header", "M3 - Difficulty History", "M4 - AI Component"]
# )

# with tab1:
#     render_m1()

# with tab2:
#     render_m2()

# with tab3:
#     render_m3()

# with tab4:
#     render_m4()
"""
app.py
------
Punto de entrada del CryptoChain Analyzer Dashboard.

Ejecutar con:
    streamlit run app.py

El dashboard se auto-refresca cada 60 segundos sin intervención del usuario.
Los módulos comparten datos entre sí para minimizar llamadas a la API.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Importar módulos del proyecto
from modules import m1_pow_monitor, m2_block_header, m3_difficulty_history, m4_ai_component

# ── Configuración de la página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoChain Analyzer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-refresco cada 60 segundos ────────────────────────────────────────────
# Devuelve el número de refresco actual (se puede usar para logs o depuración).
refresh_count = st_autorefresh(interval=60_000, limit=None, key="autorefresh")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/64px-Bitcoin.svg.png",
        width=48,
    )
    st.title("CryptoChain\nAnalyzer")
    st.caption("Hash Functions & Blockchain\nUniversidad Alfonso X el Sabio — 2025-26")
    st.divider()

    st.markdown("### Módulos")
    st.markdown("-  M1 · Proof of Work Monitor")
    st.markdown("-  M2 · Block Header Analyzer")
    st.markdown("-  M3 · Difficulty History")
    st.markdown("-  M4 · AI Anomaly Detector")
    st.divider()

    st.markdown("### Info")
    st.markdown("**APIs usadas:**")
    st.markdown("- [Blockstream](https://blockstream.info/api)")
    st.markdown("- [Blockchain.info](https://api.blockchain.info)")
    st.markdown("- [Mempool.space](https://mempool.space/api)")
    st.divider()

    st.caption(f"Auto-refresco: cada 60 s (refresco #{refresh_count})")
    if st.button(" Forzar refresco"):
        st.cache_data.clear()
        st.rerun()

# ── Header principal ───────────────────────────────────────────────────────────
st.title("₿ CryptoChain Analyzer Dashboard")
st.markdown(
    "**Dashboard en tiempo real** de métricas criptográficas de la red Bitcoin. "
    "Datos obtenidos de APIs públicas · Auto-refresco cada 60 s."
)
st.divider()

# ── Pestañas ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    " M1 · Proof of Work",
    " M2 · Block Header",
    " M3 · Difficulty History",
    " M4 · AI Anomaly Detector",
])

# Compartimos los bloques entre M1 y M4 para evitar doble llamada a la API.
# Usamos st.session_state como caché ligera entre pestañas.
if "blocks" not in st.session_state or refresh_count != st.session_state.get("last_refresh"):
    st.session_state["blocks"] = None
    st.session_state["last_refresh"] = refresh_count

with tab1:
    blocks = m1_pow_monitor.render(blocks=st.session_state["blocks"])
    if blocks:
        st.session_state["blocks"] = blocks

with tab2:
    m2_block_header.render()

with tab3:
    m3_difficulty_history.render()

with tab4:
    m4_ai_component.render(blocks=st.session_state.get("blocks"))