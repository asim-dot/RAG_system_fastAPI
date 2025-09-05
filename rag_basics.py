from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA


def load_file(pdf_file):

    loader = PyPDFLoader(pdf_file)
    document = loader.load()

    print(f"Total number of documents: {len(document)}")
    print(f"Document preview: {document[0].page_content[0:200]}...")

    return document

def document_chunk(document):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 400,
        length_function = len
    )

    chunks = text_splitter.split_documents(document) 
    print(f"Total number of chunks: {len(chunks)}.")
    return chunks

def vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding= embeddings,
        persist_directory= "./new_new_chroma_db"
    )

    print(f"Total embeddings stored: {len(chunks)}")
    return vector_store

def search_documents(vector_store, query, k=3):
    """Search for relevant chunks"""
    
    # Search for similar chunks
    results = vector_store.similarity_search(query, k=k)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:200])
    
    return results


def create_rag_chain(vector_store):
    """Create RAG chain with Ollama"""

    llm = OllamaLLM(
        model="llama3.2:3b",
        temperature=0.2
    )

    # Create QA chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain


def ask_question(qa_chain, question):
    """Ask question with RAG"""

    response = qa_chain({"query": question})

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['result']}")
    print(f"\nSource chunks used: {len(response['source_documents'])}")
    
    return response

# Full pipeline test
if __name__ == "__main__":
    # Load and process
    docs = load_file("multigen.pdf")
    chunks = document_chunk(docs)
    vector_store = vector_store(chunks)
    
    # Create RAG chain
    qa_chain = create_rag_chain(vector_store)
    
    # Ask questions
    ask_question(qa_chain, "What is this document about?")
    ask_question(qa_chain, "What are the main points?")