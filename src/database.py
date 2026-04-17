from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from .config import Config

def get_embeddings():
    """Inicializa el modelo de embeddings de Google"""
    return GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)

def get_vector_db():
    """Carga la base de datos vectorial existente"""
    return Chroma(
        persist_directory=Config.CHROMA_PATH, 
        embedding_function=get_embeddings()
    )

def save_to_db(chunks):
    """Crea una base de datos nueva o sobreescribe la actual de forma segura"""
    # Usamos from_documents para recrear el índice sin borrar archivos bloqueados
    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=Config.CHROMA_PATH
    )
    return db
