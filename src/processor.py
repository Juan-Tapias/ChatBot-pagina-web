import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

def load_and_split_docs():
    """Busca archivos .txt y .pdf en la carpeta documents y los divide en fragmentos"""
    if not os.path.exists(Config.DOCUMENTS_PATH):
        os.makedirs(Config.DOCUMENTS_PATH)
        
    docs = []
    for file in os.listdir(Config.DOCUMENTS_PATH):
        file_path = os.path.join(Config.DOCUMENTS_PATH, file)
        if file.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error cargando {file}: {e}")
        elif file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error cargando {file}: {e}")
                
    if not docs:
        raise ValueError(f"No se encontraron archivos válidos (.txt o .pdf) en {Config.DOCUMENTS_PATH}")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    return text_splitter.split_documents(docs)

