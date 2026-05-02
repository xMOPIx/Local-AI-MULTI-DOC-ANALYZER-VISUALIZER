"""
Backend de Local AI (FastAPI).
Gestiona la ingesta de documentos, la base de datos vectorial Chroma, 
la generación de respuestas usando Ollama (RAG) y la evaluación con Ragas.
"""
# ==========================================
# 1. IMPORTACIONES
# ==========================================
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import asyncio
import os

# --- Imports de Langchain ---
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Imports de Evaluación (RAGAS) ---
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# ==========================================
# 2. MODELOS DE DATOS (Pydantic)
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatQuery(BaseModel):
    query: str
    history: list[ChatMessage] = []
    active_files: list[str] = []
    use_ragas: bool = False
    use_reasoning: bool = False

# ==========================================
# 3. CONFIGURACIÓN GLOBAL E INICIALIZACIÓN
# ==========================================
app = FastAPI(title="Local AI")
CHROMA_PATH = "db_data"
base_url = "http://host.docker.internal:11434"
ENABLE_RAGAS_EVAL = os.getenv("ENABLE_RAGAS_EVAL", "false").lower() in ("1", "true", "yes")

# Inicialización de Langchain y Chroma
embeddings_engine = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
llm = OllamaLLM(model="llama3.1", temperature=0, base_url=base_url)
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_engine)

# ==========================================
# 4. FUNCIONES AUXILIARES
# ==========================================
def load_document(file_path: str, filename: str):
    """
    Carga un documento seleccionando el cargador (Loader) adecuado según la extensión.
    Soporta PDF, DOCX, CSV y texto plano.
    Añade metadatos del archivo a cada fragmento.
    """
    ext = filename.lower().split('.')[-1]
    
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    elif ext == 'csv':
        loader = CSVLoader(file_path)
    elif ext in ['txt', 'md']:
        loader = TextLoader(file_path)
    else:
        # Default to TextLoader for unknown types
        loader = TextLoader(file_path)
    
    docs = loader.load()
    
    # Añadir nombre de archivo al contenido para ayudar al LLM
    for d in docs:
        d.metadata["source"] = filename
        d.page_content = f"ARCHIVO: {filename}\n{d.page_content}"
    
    return docs

# ==========================================
# 5. ENDPOINTS DE GESTIÓN DE ARCHIVOS
# ==========================================
@app.post("/reset")
async def reset_db():
    """Limpia la base de datos vectorial eliminando todos los documentos."""
    global vector_db
    # Extraer y eliminar todos los IDs
    ids = vector_db.get()["ids"]
    if ids:
        vector_db.delete(ids=ids)
    return {"status": "success", "message": "Base de datos limpiada"}

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    """Elimina únicamente los fragmentos asociados a un archivo específico."""
    try:
        result = vector_db.get(where={"source": filename})
        if result and result.get("ids"):
            vector_db.delete(ids=result["ids"])
    except Exception as e:
        print(f"Error borrando {filename}: {e}")
    return {"status": "success", "message": f"Archivo {filename} eliminado"}
    
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Recibe un archivo, lo guarda temporalmente, extrae el texto, 
    lo divide en fragmentos (chunks) y lo guarda en ChromaDB.
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 1. Cargar documento según extensión
    docs = load_document(temp_path, file.filename)
    
    # 2. Dividir texto en fragmentos solapados
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # 3. Guardar fragmentos en la BD Vectorial
    if hasattr(vector_db, "add_documents"):
        vector_db.add_documents(splits)
        if hasattr(vector_db, "persist"):
            vector_db.persist()
    else:
        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings_engine, 
            persist_directory=CHROMA_PATH
        )
    
    # Limpieza
    os.remove(temp_path)
    return {"status": "success", "filename": file.filename}

# ==========================================
# 6. ENDPOINT PRINCIPAL (CHAT & RAG)
# ==========================================
@app.post("/ask")
async def ask_ai(chat_query: ChatQuery):
    """
    Ejecuta el flujo completo de RAG:
    1. Búsqueda semántica (MMR) en Chroma.
    2. Construcción del Prompt (con o sin CoT).
    3. Generación de respuesta con LLM.
    4. Opcional: Evaluación de fidelidad y relevancia con Ragas.
    """
    # 1. Recuperación de Documentos
    docs = []
    
    # Si tenemos archivos activos y no son demasiados (para no reventar el tiempo/VRAM),
    # intentamos sacar fragmentos equitativamente de CADA archivo.
    if chat_query.active_files and len(chat_query.active_files) <= 5:
        k_per_file = max(2, 10 // len(chat_query.active_files))
        for file_name in chat_query.active_files:
            try:
                # Filtrar por metadato "source" para obligar a leer cada archivo
                file_docs = vector_db.similarity_search(
                    chat_query.query, 
                    k=k_per_file, 
                    filter={"source": file_name}
                )
                docs.extend(file_docs)
            except Exception as e:
                print(f"Error recuperando de {file_name}: {e}")
                
    # Fallback: Si no hay archivos o la búsqueda por filtro falló, usamos búsqueda general
    if not docs:
        if hasattr(vector_db, "max_marginal_relevance_search"):
            docs = vector_db.max_marginal_relevance_search(chat_query.query, k=10, fetch_k=35)
        else:
            docs = vector_db.similarity_search(chat_query.query, k=10)
    
    # Incluimos el nombre del archivo fuente en el contexto para que el LLM pueda diferenciarlos
    context_list = [f"Fuente ({d.metadata.get('source', 'Desconocido')}): {d.page_content}" for d in docs]
    
    # Ajustamos el contexto a 5000 caracteres para un balance óptimo entre info y VRAM
    max_context_chars = 5000
    context_text = "\n".join(context_list)
    if len(context_text) > max_context_chars:
        context_text = context_text[:max_context_chars] + "..."

    # 2. Construcción del historial para el prompt
    chat_history_text = ""
    for msg in chat_query.history[-5:]: # Tomamos los últimos 5 mensajes para no saturar
        role_name = "Usuario" if msg.role == "user" else "Asistente"
        chat_history_text += f"{role_name}: {msg.content}\n"
        
    archivos_activos_str = ", ".join(chat_query.active_files) if chat_query.active_files else "Ninguno"

    # 3. Generación de respuesta
    if chat_query.use_reasoning:
        prompt = f"""
        Eres un asistente técnico avanzado con capacidad de análisis de datos.
        
        ARCHIVOS ACTUALMENTE INDEXADOS (que el usuario ha subido):
        {archivos_activos_str}
        
        HISTORIAL DE CONVERSACIÓN (Contexto previo):
        {chat_history_text}

        CONTEXTO EXTRAÍDO DE LOS DOCUMENTOS:
        {context_text}

        PREGUNTA ACTUAL DEL USUARIO: {chat_query.query}

        INSTRUCCIONES CRÍTICAS:
        1. <pensamiento>: Analiza el historial y el contexto para planear la respuesta.
        2. <respuesta>: Responde con rigor. 
        3. CONDICIÓN PARA GRÁFICAS:
           - Si el usuario NO pide gráfica: responde solo con texto. No des excusas ni menciones que puedes graficar.
           - Si el usuario PIDE gráfica (usa palabras como "grafica", "dibuja", etc.): DEBES generar un bloque ```python con código matplotlib.
        4. REGLAS DE CÓDIGO PYTHON (Solo si generas código):
           - Usa `fig, ax = plt.subplots()`.
           - Extrae los datos del contexto y escríbelos a mano en variables. No leas archivos del disco.
        
        Formato: <pensamiento>...</pensamiento><respuesta>...</respuesta>
        """
    else:
        prompt = f"""
        Eres un asistente técnico avanzado.
        
        ARCHIVOS ACTUALMENTE INDEXADOS (que el usuario ha subido):
        {archivos_activos_str}
        
        HISTORIAL DE CONVERSACIÓN:
        {chat_history_text}

        CONTEXTO EXTRAÍDO DE LOS DOCUMENTOS:
        {context_text}

        PREGUNTA ACTUAL: {chat_query.query}

        INSTRUCCIONES:
        1. Responde de forma clara y directa usando el contexto y el historial.
        2. CONDICIÓN PARA GRÁFICAS:
           - Si el usuario NO pide gráfica: responde solo con texto. No des excusas ni menciones que puedes graficar.
           - Si el usuario PIDE gráfica (usa palabras como "grafica", "dibuja", etc.): DEBES generar un bloque ```python con código matplotlib.
        3. REGLAS DE CÓDIGO PYTHON (Solo si generas código):
           - Usa `fig, ax = plt.subplots()`.
           - Extrae los datos del contexto y escríbelos a mano en variables. No leas archivos del disco.
        """
    response = llm.invoke(prompt)

    # 3. --- IMPLEMENTACIÓN DE RAGAS ---
    # Creamos el diccionario con los datos de esta pregunta
    eval_data = {
        "question": [chat_query.query],
        "answer": [response],
        "contexts": [context_list],  # Ragas necesita una lista de listas
    }
    
    # Lo convertimos al formato que Ragas entiende
    dataset = Dataset.from_dict(eval_data)
    
    evaluation = None
    enable_ragas = ENABLE_RAGAS_EVAL or chat_query.use_ragas
    if enable_ragas:
        # Ejecutamos la auditoría
        # Nota: Esto hará que la API tarde un poco más porque Llama 3.1 tiene que "corregir"
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            None, 
            lambda: evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings_engine
            )
        )

        evaluation = {
            "fidelidad": float(score["faithfulness"][0]) if isinstance(score["faithfulness"], list) else float(score["faithfulness"]),
            "relevancia": float(score["answer_relevancy"][0]) if isinstance(score["answer_relevancy"], list) else float(score["answer_relevancy"])
        }

    # 4. Devolvemos todo al Frontend
    response_payload = {"answer": response}
    if evaluation is not None:
        response_payload["evaluation"] = evaluation
    return response_payload