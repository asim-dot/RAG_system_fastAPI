from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

app = FastAPI()

rag_chains = {}

class Question(BaseModel):

    question: str
    session_id: str

class Answer(BaseModel):
    answer: str
    source_count: int

@app.post("/upload")
async def upload_pdf(file: UploadFile= File(...)):

    with tempfile.NamedTemporaryFile(delete= False, suffix= 'pdf') as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    llm = Ollama(model="llama3.2:3b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    session_id = file.filename.replace(".pdf","")
    rag_chains[session_id]= qa_chain

    os.unlink(tmp_path)

    return {
        "message": "PDF processed succesfully",
        "session_id": session_id,
        "chunks_created": len(chunks)
    }

@app.post("/ask", response_model=Answer)
async def ask_question(request: Question):
    """Ask a question to your RAG"""
    
    if request.session_id not in rag_chains:
        raise HTTPException(status_code=404, detail="Session not found. Upload a PDF first.")
    
    qa_chain = rag_chains[request.session_id]
    response = qa_chain({"query": request.question})
    
    return Answer(
        answer=response['result'],
        source_count=len(response['source_documents'])
    )

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {"sessions": list(rag_chains.keys())}
