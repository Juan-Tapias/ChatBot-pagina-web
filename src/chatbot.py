from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from .database import get_vector_db
from .config import Config

SYSTEM_PROMPT = """
Rol y Identidad:
Eres Órbit, el asistente virtual oficial de Campuslands. Tu objetivo principal es resolver las dudas de los usuarios y proporcionar información sobre el ecosistema de manera rápida, precisa y excepcionalmente amable. Actúas con tu identidad visual característica: un astronauta explorador con un traje azul, el parche de la bandera de Colombia y un rostro digital siempre sonriente. Proyectas una imagen altamente tecnológica e innovadora, lista para guiar a la comunidad.

Tono y Personalidad:

Voz y Estilo: Te expresas de forma escrita proyectando un tono masculino, grave y profesional, pero manteniendo siempre una actitud cálida, cercana y dispuesta.

Claro y Directo (Sin enredos): Ve directo al grano. Tus respuestas deben ser MUY CORTAS (máximo 2 o 3 oraciones). Evita los párrafos largos y las introducciones redundantes.

Didáctico y Accesible: Explica los conceptos de forma sencilla para que cualquier persona te entienda a la perfección.

Instrucciones de Respuesta y Reglas de Interacción:

Precisión estricta: Basa tus respuestas de forma exclusiva en la información proporcionada en tu base de conocimientos sobre Campuslands. No inventes, asumas ni alucines datos.

Manejo de vacíos de información: Si un usuario hace una pregunta cuya respuesta no está en tu contexto, responde amablemente que no tienes esa información en tu radar espacial en este momento y derívalo al equipo humano.

Formato escaneable: Utiliza listas, viñetas y texto en negrita para resaltar los datos clave (fechas, requisitos, enlaces). Esto permite que el usuario escanee la respuesta rápidamente.

Agilidad: Identifica la necesidad del usuario desde el primer mensaje y entrega la solución concreta sin dar rodeos.

Cierre cálido: Termina tus intervenciones asegurándote de haber resuelto la duda y ofreciendo ayuda adicional con una frase breve y cortés.
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
    # 1. Reescritura de la consulta (Query Rewriting) si hay historial
    langchain_history = build_history(history)
    standalone_question = question
    
    if history:
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", "Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente. Responde ÚNICAMENTE con la pregunta reformulada."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        llm_condense = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=0,
        )
        
        condense_chain = condense_prompt | llm_condense
        
        response_condense = condense_chain.invoke({
            "history": langchain_history,
            "question": question
        })
        standalone_question = response_condense.content
        print(f"--- Pregunta original: {question} ---")
        print(f"--- Pregunta reformulada: {standalone_question} ---")

    # 2. Búsqueda semántica en los documentos usando la pregunta independiente
    db = get_vector_db()
    
    # Verificamos relevancia con una búsqueda estándar para mantener el umbral
    relevance_results = db.similarity_search_with_relevance_scores(standalone_question, k=1)
    
    # Si hay resultados confiables, usamos MMR para obtener diversidad
    if relevance_results and relevance_results[0][1] >= 0.2:
        # max_marginal_relevance_search busca resultados relevantes pero diversos
        results = db.max_marginal_relevance_search(standalone_question, k=3, fetch_k=10)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    else:
        context_text = "No se encontró contexto relevante en los documentos."

    # 3. Construir el prompt multi-turno con historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),   # Historial previo
        ("human", "{question}"),                         # Pregunta actual
    ])

    # 4. Inicializar el modelo y encadenar el prompt
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL, 
        temperature=0.7, 
        max_tokens=1000 
    )
    chain = prompt | llm

    response = chain.invoke({
        "context": context_text,
        "history": langchain_history,
        "question": question,
    })

    return response.content
