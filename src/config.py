import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Modelos (Corregidos)
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    LLM_MODEL = "gemini-2.5-flash"
    
    # Rutas
    CHROMA_PATH = os.path.join("src", "chroma_db")
    DOCUMENTS_PATH = "documents"
    
    # Seguridad
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
