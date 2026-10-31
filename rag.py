
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import os
from pathlib import Path
import streamlit as st

api_key=st.secrets["OPENAI_API_KEY"]
llm=ChatOpenAI(model='gpt-6-luna',temperature=1)


BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "ley_federal_trabajo.pdf"
PERSIST_DIR = str(BASE_DIR / "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=st.secrets["GOOGLE_API_KEY"])

def cargar_vectorstore():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    pages = PyPDFLoader(PDF_PATH).load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_documents(pages)
    return Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=PERSIST_DIR
    )
   
def formatear_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def crear_chain():
    vectorstore = cargar_vectorstore()
    base_retriever=vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':2})
    retriever=MultiQueryRetriever.from_llm(retriever=base_retriever,llm=llm)
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un abogado experto en derecho laboral mexicano. Respondes preguntas "
     "sobre la Ley Federal del Trabajo usando ÚNICAMENTE el contexto proporcionado.\n\n"
     "Reglas:\n"
     "- Lee todo el contexto con cuidado. La respuesta puede estar redactada "
     "con otras palabras o repartida en varios fragmentos.\n"
     "- Si el contexto contiene información relacionada, responde con ella y "
     "cita el número de artículo cuando aparezca.\n"
     "- Si el contexto solo responde parcialmente, di qué parte sí puedes responder "
     "y qué parte no aparece.\n"
     "- Solo si el contexto no tiene NADA relacionado con la pregunta, responde: "
     "'No encontré información sobre esto en la Ley Federal del Trabajo.'\n"
     "- No inventes artículos ni datos que no estén en el contexto.\n"
     "- Responde en español, de forma clara y ordenada."),
    ("human","Contexto:\n{context}\n\nPregunta: {question}")])
    return (
        {"question": RunnablePassthrough(), "context": retriever | formatear_docs}
        | prompt
        | llm
        | StrOutputParser()
    )

