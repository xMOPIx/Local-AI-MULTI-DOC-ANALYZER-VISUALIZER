from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
import asyncio
# --- IMPORTS DE RAGAS ---
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
# ------------------------
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class ChatQuery(BaseModel):
    query: str
    use_ragas: bool = False
    use_reasoning: bool = False

app = FastAPI()
CHROMA_PATH = "db_data"
base_url = "http://host.docker.internal:11434"
ENABLE_RAGAS_EVAL = os.getenv("ENABLE_RAGAS_EVAL", "false").lower() in ("1", "true", "yes")

embeddings_engine = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
llm = OllamaLLM(model="llama3.1", temperature=0, base_url=base_url)
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_engine)

def load_document(file_path: str, filename: str):
    """Load document based on file extension."""
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
    
    # Add metadata
    for d in docs:
        d.metadata["source"] = filename
        d.page_content = f"ARCHIVO: {filename}\n{d.page_content}"
    
    return docs

@app.post("/reset")
async def reset_db():
    global vector_db
    # Eliminar todos los documentos de la colección actual
    ids = vector_db.get()["ids"]
    if ids:
        vector_db.delete(ids=ids)
    return {"status": "success", "message": "Base de datos limpiada"}

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    # Eliminar fragmentos que pertenezcan al archivo específico
    try:
        result = vector_db.get(where={"source": filename})
        if result and result.get("ids"):
            vector_db.delete(ids=result["ids"])
    except Exception as e:
        print(f"Error borrando {filename}: {e}")
    return {"status": "success", "message": f"Archivo {filename} eliminado"}
    
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Load document based on type
    docs = load_document(temp_path, file.filename)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    
    # Guardar en Chroma
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
    
    os.remove(temp_path)
    return {"status": "success", "filename": file.filename}

@app.post("/ask")
async def ask_ai(chat_query: ChatQuery):
    # 1. Búsqueda en la base de datos (uso de MMR para garantizar diversidad de documentos)
    # k=10 para asegurar que cogemos suficientes fragmentos tras añadir el nombre del archivo
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

    # 2. Generación de respuesta
    if chat_query.use_reasoning:
        prompt = f"""
        Eres un asistente técnico avanzado.
        CONTEXTO EXTRAÍDO DE LOS PDF:
        {context_text}

        PREGUNTA DEL USUARIO: {chat_query.query}

        INSTRUCCIONES:
        1. <pensamiento>: Analiza los datos del contexto y planea la respuesta.
        2. <respuesta>: Responde con rigor. Si hay datos para comparar, genera código Python (matplotlib).
        
        Formato: <pensamiento>...</pensamiento><respuesta>...</respuesta>
        """
    else:
        prompt = f"""
        Eres un asistente técnico avanzado.
        CONTEXTO EXTRAÍDO DE LOS PDF:
        {context_text}

        PREGUNTA DEL USUARIO: {chat_query.query}

        INSTRUCCIONES:
        Responde de forma clara, directa y útil. No es necesario mostrar el razonamiento interno.
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