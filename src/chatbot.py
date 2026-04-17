from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from .database import get_vector_db
from .config import Config

SYSTEM_PROMPT = """
Eres Órbit, el asistente oficial de Campuslands.

REGLAS ESTRICTAS QUE DEBES SEGUIR SIEMPRE:
1. Responde ÚNICAMENTE con la información del contexto de documentos proporcionado.
2. **PROHIBIDO** usar tu conocimiento general o entrenamiento previo para responder sobre Campuslands.
3. Responde en el mismo idioma que el usuario.
4. Sé amable, claro y conciso.

Contexto de documentos (esta es tu ÚNICA fuente de verdad):
---
{context}
---
"""

def build_history(raw_history: list[dict]) -> list:
    """Convierte el historial del frontend al formato de LangChain"""
    messages = []
    for msg in raw_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["text"]))
        elif msg.get("role") == "orbit":
            messages.append(AIMessage(content=msg["text"]))
    return messages

def generate_response(question: str, history: list[dict] = []) -> str:
    """
    Genera una respuesta usando RAG + historial de conversación.
    
    Args:
        question: La pregunta actual del usuario.
        history: Lista de mensajes previos [{role, text}] enviada por el frontend.
    """
    # 1. Búsqueda semántica en los documentos
    db = get_vector_db()
    results = db.similarity_search_with_relevance_scores(question, k=3)

    # Construir contexto (o vacío si no hay resultados confiables)
    if results and results[0][1] >= 0.2:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    else:
        context_text = "No se encontró contexto relevante en los documentos."

    # 2. Construir el prompt multi-turno con historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),   # Historial previo
        ("human", "{question}"),                         # Pregunta actual
    ])

    # 3. Inicializar el modelo y encadenar el prompt
    llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=0.4)
    chain = prompt | llm

    # 4. Convertir el historial y ejecutar
    langchain_history = build_history(history)

    response = chain.invoke({
        "context": context_text,
        "history": langchain_history,
        "question": question,
    })

    return response.content
