"""
m5_merkle_verifier.py
---------------------
Módulo M5 — Merkle Proof Verifier.

Dado un bloque reciente, el usuario elige una transacción y el módulo:
  1. Descarga todos los txids del bloque.
  2. Construye el árbol de Merkle completo capa a capa.
  3. Verifica que la raíz calculada coincide con la merkle_root del cabecero.
  4. Muestra el camino de prueba (proof path) para la tx elegida,
     con cada hash intermedio visible.

Concepto criptográfico:
  - El árbol de Merkle permite verificar que una transacción pertenece a un
    bloque sin descargar todas las transacciones: basta con O(log N) hashes.
  - Cada nodo padre = SHA256(SHA256(hijo_izq + hijo_der)).
  - Si el número de hojas es impar, el último nodo se duplica.
"""

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from api.blockchain_client import _get, BASE_BS, get_tip_hash, get_block


# ── Funciones criptográficas ───────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    """Doble SHA-256, la función hash de Bitcoin."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def get_block_txids(block_hash: str) -> list[str]:
    """Descarga la lista de txids de un bloque (big-endian, como aparecen en exploradores)."""
    return _get(f"{BASE_BS}/block/{block_hash}/txids").json()


def txid_to_internal(txid_hex: str) -> bytes:
    """
    Convierte un txid (big-endian, formato explorador) a bytes internos (little-endian).
    Bitcoin almacena los hashes de txs en little-endian en el árbol de Merkle.
    """
    return bytes.fromhex(txid_hex)[::-1]


def build_merkle_tree(txids: list[str]) -> list[list[bytes]]:
    """
    Construye el árbol de Merkle capa a capa.
    Devuelve lista de capas: capas[0] = hojas, capas[-1] = [raíz].
    Cada elemento es bytes en formato interno (little-endian).
    """
    layer = [txid_to_internal(txid) for txid in txids]
    layers = [layer]

    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer = layer + [layer[-1]]  # duplicar último si impar
        next_layer = []
        for i in range(0, len(layer), 2):
            combined = layer[i] + layer[i + 1]
            next_layer.append(sha256d(combined))
        layer = next_layer
        layers.append(layer)

    return layers


def get_merkle_proof(layers: list[list[bytes]], tx_index: int) -> list[dict]:
    """
    Extrae el proof path para la transacción en tx_index.
    Devuelve lista de pasos: {"sibling": hash_hex, "side": "left"/"right", "parent": hash_hex}
    """
    proof = []
    idx = tx_index

    for depth, layer in enumerate(layers[:-1]):
        # Asegurar paridad (igual que en build_merkle_tree)
        padded = layer if len(layer) % 2 == 0 else layer + [layer[-1]]
        sibling_idx = idx ^ 1  # XOR con 1 da el índice del hermano
        sibling_hash = padded[sibling_idx]
        parent_layer = layers[depth + 1]
        parent_hash  = parent_layer[idx // 2]

        proof.append({
            "depth":   depth,
            "sibling": sibling_hash[::-1].hex(),   # volver a big-endian para mostrar
            "side":    "derecha" if idx % 2 == 0 else "izquierda",
            "parent":  parent_hash[::-1].hex(),
        })
        idx //= 2

    return proof


def verify_proof(txid_hex: str, proof: list[dict], expected_root_hex: str) -> bool:
    """
    Verifica el proof path reconstruyendo la raíz desde la hoja.
    Devuelve True si la raíz reconstruida coincide con expected_root_hex.
    """
    current = txid_to_internal(txid_hex)

    for step in proof:
        sibling = bytes.fromhex(step["sibling"])[::-1]  # a little-endian
        if step["side"] == "derecha":
            # la tx actual está a la izquierda
            combined = current + sibling
        else:
            combined = sibling + current
        current = sha256d(combined)

    # current es la raíz en little-endian; invertir para comparar con el JSON
    computed_root = current[::-1].hex()
    return computed_root == expected_root_hex


# ── Renderizado Streamlit ──────────────────────────────────────────────────────

def render() -> None:
    st.header("🌿 M5 — Merkle Proof Verifier")

    st.markdown(
        """
        Verifica criptográficamente que una transacción pertenece al bloque
        reconstruyendo el camino de prueba hash a hash.
        Con **O(log₂ N)** hashes es suficiente — no necesitas descargar todo el bloque.
        """
    )

    # ── Cargar bloque ──────────────────────────────────────────────────────────
    with st.spinner("Obteniendo último bloque…"):
        try:
            tip_hash = get_tip_hash()
            block    = get_block(tip_hash)
        except Exception as exc:
            st.error(f"Error al obtener el bloque: {exc}")
            return

    n_tx = block.get("tx_count", "?")
    st.info(f"Bloque **#{block['height']}** — {n_tx} transacciones — hash: `{tip_hash[:20]}…`")

    # ── Descargar txids ────────────────────────────────────────────────────────
    with st.spinner(f"Descargando {n_tx} txids…"):
        try:
            txids = get_block_txids(tip_hash)
        except Exception as exc:
            st.error(f"Error al obtener txids: {exc}")
            return

    st.success(f"✅ {len(txids)} txids descargados.")

    # ── Selección de transacción ───────────────────────────────────────────────
    max_idx = len(txids) - 1
    tx_index = st.slider(
        "Selecciona el índice de la transacción a verificar",
        min_value=0, max_value=min(max_idx, 99), value=1,
        help="0 = coinbase (la primera tx del bloque, creada por el minero)",
    )

    selected_txid = txids[tx_index]
    st.markdown(f"**Txid seleccionado:** `{selected_txid}`")

    if st.button("🔎 Verificar prueba Merkle"):
        with st.spinner("Construyendo árbol de Merkle…"):
            layers = build_merkle_tree(txids)

        # Raíz calculada localmente
        computed_root = layers[-1][0][::-1].hex()
        api_root      = block["merkle_root"]

        # ── Resultado de la raíz ───────────────────────────────────────────────
        st.subheader("Comparación de la raíz de Merkle")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raíz calculada localmente:**")
            st.code(computed_root)
        with col2:
            st.markdown("**Raíz reportada por la API:**")
            st.code(api_root)

        if computed_root == api_root:
            st.success("✅ Las raíces coinciden — árbol construido correctamente.")
        else:
            st.error("❌ Las raíces no coinciden.")

        # ── Proof path ────────────────────────────────────────────────────────
        proof = get_merkle_proof(layers, tx_index)
        valid = verify_proof(selected_txid, proof, api_root)

        st.subheader(f"Proof path para tx[{tx_index}] — {len(proof)} pasos (profundidad del árbol)")

        proof_df = pd.DataFrame([{
            "Paso":         i + 1,
            "Hash hermano": step["sibling"][:32] + "…",
            "Posición":     step["side"],
            "Hash padre":   step["parent"][:32] + "…",
        } for i, step in enumerate(proof)])
        st.dataframe(proof_df, use_container_width=True, hide_index=True)

        if valid:
            st.success(
                f"✅ Prueba Merkle **VÁLIDA** — la transacción `{selected_txid[:20]}…` "
                f"pertenece al bloque #{block['height']}."
            )
        else:
            st.error("❌ Prueba Merkle inválida.")

        # ── Visualización del árbol (primeras capas) ───────────────────────────
        st.subheader("Estructura del árbol de Merkle")
        n_layers = len(layers)
        n_leaves = len(txids)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Transacciones (hojas)", n_leaves)
        col_b.metric("Profundidad del árbol", n_layers - 1)
        col_c.metric("Hashes de prueba necesarios", len(proof))

        # Gráfico de barras: nodos por capa
        layer_sizes = [len(l) for l in layers]
        fig = go.Figure(go.Bar(
            x=[f"Capa {i}" for i in range(n_layers)],
            y=layer_sizes,
            marker_color="steelblue",
            text=layer_sizes,
            textposition="outside",
        ))
        fig.update_layout(
            title="Número de nodos por capa del árbol de Merkle",
            xaxis_title="Capa (0 = hojas, última = raíz)",
            yaxis_title="Nodos",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"💡 Con {n_leaves} transacciones, necesitas solo **{len(proof)} hashes** "
            f"para probar que cualquier tx pertenece al bloque, en lugar de los {n_leaves} completos. "
            f"Eficiencia: O(log₂ {n_leaves}) ≈ {len(proof)} vs O(N) = {n_leaves}."
        )