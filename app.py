import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="TelecoBrain Pro", layout="wide")
API_URL = os.getenv("API_URL", "http://asistente-ia:8000")

# --- SIDEBAR (Aquí está lo que te falta) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Inicializar contador para resetear file_uploader
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    
    uploaded_files = st.file_uploader(
        "Sube tus PDFs", 
        type="pdf", 
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.reset_counter}"
    )
    
    if "use_ragas" not in st.session_state:
        st.session_state.use_ragas = False
    if "use_reasoning" not in st.session_state:
        st.session_state.use_reasoning = False
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()

    use_ragas = st.checkbox(
        "Evaluación RAGAS",
        value=st.session_state.use_ragas,
        help="Activa la evaluación de fidelidad y relevancia. Si está desactivado, la respuesta será más rápida.",
        key="use_ragas"
    )

    use_reasoning = st.checkbox(
        "Razonamiento interno (CoT)",
        value=st.session_state.use_reasoning,
        help="Activa el razonamiento interno en el prompt. Si está desactivado, la IA responderá de forma más directa.",
        key="use_reasoning"
    )

    st.caption("Los PDFs se indexan automáticamente al subirlos.")

    # --- Lógica de Sincronización Automática ---
    current_files_names = {f.name for f in uploaded_files} if uploaded_files else set()
    
    # 1. Detectar archivos ELIMINADOS del uploader
    files_to_remove = st.session_state.indexed_files - current_files_names
    if files_to_remove:
        for file_name in files_to_remove:
            try:
                requests.delete(f"{API_URL}/delete/{file_name}")
                st.session_state.indexed_files.remove(file_name)
                st.toast(f"🗑️ Eliminado: {file_name}")
            except Exception as e:
                st.error(f"Error al eliminar {file_name}: {e}")

    # 2. Detectar archivos NUEVOS en el uploader
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner(f"Indexando {len(new_files)} archivo(s)..."):
                for uploaded_file in new_files:
                    try:
                        files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        resp = requests.post(f"{API_URL}/ingest", files=files_payload, timeout=120)
                        resp.raise_for_status()
                        st.session_state.indexed_files.add(uploaded_file.name)
                        st.toast(f"✅ Indexado: {uploaded_file.name}")
                    except Exception as e:
                        st.warning(f"⚠️ {uploaded_file.name}: {str(e)[:50]}")
    
    # Botón de Limpieza Total (mantiene su utilidad para borrar historial)
    if st.button("🗑️ Limpiar Todo", use_container_width=True):
        requests.post(f"{API_URL}/reset")
        st.session_state.indexed_files.clear()
        st.session_state.messages = []
        st.session_state.reset_counter += 1
        st.success("✅ Todo limpiado")
        st.rerun()

    # Mostrar archivos indexados actualmente
    if st.session_state.indexed_files:
        st.divider()
        st.subheader(f"📄 Archivos Activos ({len(st.session_state.indexed_files)})")
        for idx_file in sorted(st.session_state.indexed_files):
            st.caption(f"✓ {idx_file}")
    else:
        # Asegurar que si el uploader está vacío, el estado también
        if not uploaded_files and st.session_state.indexed_files:
            requests.post(f"{API_URL}/reset")
            st.session_state.indexed_files.clear()

# --- CUERPO PRINCIPAL ---
st.title("🚀 TelecoBrain Professional")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat Input
if query := st.chat_input("Pregunta algo sobre los documentos..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Llamada al backend
    try:
        payload = {
            "query": query,
            "use_ragas": st.session_state.use_ragas,
            "use_reasoning": st.session_state.use_reasoning
        }
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        response.raise_for_status()
        res = response.json()
        full_text = res["answer"]
        
        with st.chat_message("assistant"):
            # 1. CoT (Pensamiento)
            if "<pensamiento>" in full_text:
                parts = full_text.split("</pensamiento>")
                thought = parts[0].replace("<pensamiento>", "").strip()
                answer = parts[1].replace("<respuesta>", "").replace("</respuesta>", "").strip()
                with st.expander("🧠 Razonamiento interno"):
                    st.write(thought)
            else:
                answer = full_text

            # 2. Respuesta Final
            st.markdown(answer)

            # 3. Métricas RAGAS en Sidebar dinámico
            if "evaluation" in res:
                evals = res["evaluation"]
                st.sidebar.divider()
                st.sidebar.subheader("📊 Calidad de Respuesta")
                st.sidebar.metric("Fidelidad", f"{evals['fidelidad']:.2f}")
                st.sidebar.metric("Relevancia", f"{evals['relevancia']:.2f}")

            # 4. Gráficas
            if "```python" in answer:
                try:
                    code = answer.split("```python")[1].split("```")[0].strip()
                    with st.expander("📊 Visualización"):
                        fig, ax = plt.subplots()
                        exec(code, {}, {"plt": plt, "np": np, "ax": ax, "fig": fig})
                        st.pyplot(fig)
                except:
                    pass

        st.session_state.messages.append({"role": "assistant", "content": full_text})
    except Exception as e:
        st.error(f"Error de conexión con el backend: {e}")