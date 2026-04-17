from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.processor import load_and_split_docs
from src.database import save_to_db
from src.chatbot import generate_response

app = FastAPI(
    title="Campuslands AI API",
    description="API para el ChatBot de Campuslands usando RAG con Gemini y ChromaDB"
)

# Configuración de CORS para permitir conexión con el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para permitir Private Network Access (soluciona ERR_BLOCKED_BY_CLIENT)
# Algunos navegadores y extensiones bloquean peticiones de páginas públicas a localhost
@app.middleware("http")
async def add_private_network_header(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

class MessageItem(BaseModel):
    role: str   # "user" | "orbit"
    text: str

class QuestionRequest(BaseModel):
    question: str
    history: list[MessageItem] = []   # Historial previo (opcional, viene del frontend)

@app.get("/")
def status():
    """Endpoint para verificar que la API esta online"""
    return {
        "status": "online", 
        "message": "Bienvenido al cerebro de Campuslands",
        "engine": "Gemini 1.5 Flash + RAG"
    }

@app.post("/index")
def index_documents():
    """
    Lee archivos .txt de la carpeta /documents, los procesa y los guarda 
    en la base de datos vectorial para que el chatbot pueda usarlos.
    """
    try:
        chunks = load_and_split_docs()
        save_to_db(chunks)
        return {"message": f"Indexación completada correctamente. {len(chunks)} fragmentos procesados."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en indexación: {str(e)}")

@app.post("/ask")
def ask(req: QuestionRequest):
    """
    Recibe una pregunta y busca en los documentos de Campuslands 
    para dar una respuesta precisa usando Inteligencia Artificial.
    """
    try:
        # Convertir los items del historial a dicts simples para el chatbot
        history = [{"role": m.role, "text": m.text} for m in req.history]
        answer = generate_response(req.question, history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {str(e)}")
