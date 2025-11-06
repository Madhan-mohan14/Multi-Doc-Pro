# vector_store_handler.py 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_vector_store_from_documents(documents: list[Document]):
    """
    Creates a ChromaDB vector store from a list of documents, handling empty inputs.
    """
    # --- ADDED VALIDATION ---
    # This is the most important check to prevent the error you saw.
    if not documents:
        print("Error: No documents provided to create the vector store. Aborting.")
        return None
        
    print("Starting the vector store creation process...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(documents)

    # --- ADDED VALIDATION ---
    # Also check if chunking resulted in any actual chunks
    if not chunked_docs:
        print("Error: Document chunking resulted in zero chunks. Aborting.")
        return None

    print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory="./chroma_db_prod"
    )
    print(f"Vector store created successfully in './chroma_db_prod'.")
    
    return vector_db.as_retriever()