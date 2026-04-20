import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

def load_and_split_docs():
    """Busca archivos .txt en la carpeta documents y los divide en fragmentos"""
    if not os.path.exists(Config.DOCUMENTS_PATH):
        os.makedirs(Config.DOCUMENTS_PATH)
        
    loader = DirectoryLoader(
        Config.DOCUMENTS_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    docs = loader.load()
    
    if not docs:
        raise ValueError(f"No se encontraron archivos .txt en {Config.DOCUMENTS_PATH}")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    return text_splitter.split_documents(docs)
