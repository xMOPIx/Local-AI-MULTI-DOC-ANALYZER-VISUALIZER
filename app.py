import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
import matplotlib.pyplot as plt
import numpy as np

# Configuración UI
st.set_page_config(page_title="TelecoBrain", layout="wide")

# Inicialización el historial
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_engines():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = OllamaLLM(model="llama3.1", temperature=0)
    return embeddings, llm

motores = get_engines()
embeddings_engine = motores[0]
llm = motores[1]

# Interfaz Lateral e ingesta
with st.sidebar:
    st.header("⚙️ Configuración")
    uploaded_files = st.file_uploader("Sube tus PDFs", type="pdf", accept_multiple_files=True)

    if st.button("🗑️ Limpiar Todo"):
        st.session_state.clear()
        st.rerun()  # CORRECTO: Era st.rerun(), no return()
    
# Ingesta
if uploaded_files:
    if "vector_db" not in st.session_state:   
        with st.spinner("Indexando..."): # CORRECTO: Era spinner con dos 'n'
            all_docs = []
            file_identities = {} 

            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                file_identities[uploaded_file.name] = docs[0].page_content[:1500]    

                for d in docs:
                    d.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)
                os.remove(temp_path) 

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            splits = text_splitter.split_documents(all_docs)

            st.session_state.vector_db = DocArrayInMemorySearch.from_documents(splits, embeddings_engine)
            st.session_state.file_identities = file_identities

# Chat e Interfaz Principal
st.title("🚀 TelecoBrain: Multi-Doc Analyzer")

chat_container = st.container()

with chat_container:
    for m in st.session_state.messages:
        # CORRECTO: Era st.chat_message (singular), no chat_messages
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# Entrada usuario
if query := st.chat_input("¿Qué quieres saber?"):
    st.session_state.messages.append({"role": "human", "content": query})
    with chat_container:
        with st.chat_message("human"):
            st.markdown(query)

    if "vector_db" in st.session_state:
        with st.chat_message("ai"):
            docs = st.session_state.vector_db.similarity_search(query, k=10) 
            
            # CORRECTO: He definido las variables que faltaban para el prompt
            context_text = "\n".join([f"[{d.metadata['source']}]: {d.page_content}" for d in docs])
            
            identities_str = ""
            for name, intro in st.session_state.file_identities.items():
                identities_str += f"\n[DOC: {name}]\n{intro[:300]}...\n"

            prompt = f"""Eres un asistente técnico avanzado. Tienes que analizar varios archivos PDF.
            
            IDENTIDAD DE LOS ARCHIVOS CARGADOS:
            {identities_str}
            
            DETALLES TÉCNICOS ENCONTRADOS:
            {context_text}
            
            PREGUNTA DEL USUARIO: {query}
            
            INSTRUCCIONES DE RESPUESTA:
            1. RIGOR: Usa solo la info de los PDFs.
            2. ESTRUCTURA: Diferencia bien de qué archivo sacas cada dato.
            3. VISUALIZACIÓN: Genera código Python (matplotlib) si hay datos numéricos.
            4. ESTILO: Profesional, directo.
            """
            
            response = llm.invoke(prompt)
            st.markdown(response)

            # Gráficas
            if "```python" in response:
                try:
                    # Extraemos el código limpiando espacios extra
                    code = response.split("```python")[1].split("```")[0].strip()
                    
                    with st.expander("📊 Visualización Técnica"):
                        # Creamos la figura explícitamente
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Definimos el entorno de ejecución
                        # Incluimos fig y ax para que la IA pueda usarlos si quiere
                        local_vars = {"plt": plt, "np": np, "ax": ax, "fig": fig}
                        
                        exec(code, {}, local_vars)
                        
                        # Forzamos que se dibuje la figura actual
                        st.pyplot(plt.gcf())
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Error en el código de la gráfica: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Primero sube un PDF.")

