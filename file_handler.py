# file_handler.py 

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os
import tempfile

def load_documents_from_files(uploaded_files: list) -> list[Document]:
    """
    Loads content from a list of uploaded files, ensuring no empty documents are returned.
    """
    documents = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            print(f"Processing file: {uploaded_file.name}")
            
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(temp_path, encoding='utf-8')
            else:
                print(f"Skipping unsupported file type: {uploaded_file.name}")
                continue
                
            try:
                loaded_docs = loader.load()
                
                # --- ADDED VALIDATION ---
                # Filter out any documents that might be empty after loading
                non_empty_docs = [doc for doc in loaded_docs if doc.page_content.strip()]

                if not non_empty_docs:
                    print(f"Warning: No text content found in {uploaded_file.name}.")
                    continue

                for doc in non_empty_docs:
                    doc.metadata["source"] = uploaded_file.name
                    
                documents.extend(non_empty_docs)
            except Exception as e:
                # Handle cases where a file might be corrupted
                print(f"Error loading file {uploaded_file.name}: {e}")
                continue
            
    if not documents:
        print("Warning: No valid text content was extracted from any of the uploaded files.")

    print(f"Successfully loaded and processed content from uploaded files.")
    return documents