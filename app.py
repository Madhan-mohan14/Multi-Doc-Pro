# app.py 

import streamlit as st
from file_handler import load_documents_from_files
from vector_store_handler import create_vector_store_from_documents
from chain_handler import create_advanced_rag_chain

def main():
    st.set_page_config(page_title="Multi-Doc Pro", page_icon="📚", layout="wide")
    st.title("Multi-Doc Pro: Chat with Your Documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "txt"], accept_multiple_files=True)
        
        if st.button("Process Documents"):
            if uploaded_files:
                # --- THIS IS THE SPINNER ---
                with st.spinner("Processing documents... This can take a moment."):
                    loaded_docs = load_documents_from_files(uploaded_files)
                    if loaded_docs:
                        retriever = create_vector_store_from_documents(loaded_docs)
                        st.session_state.rag_chain = create_advanced_rag_chain(retriever)
                        st.success("Documents processed successfully! Ready to chat.")
                    else:
                        st.error("Could not load any content from the documents.")
            else:
                st.warning("Please upload at least one document.")

    # --- Main Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        if st.session_state.rag_chain is None:
            st.error("Please process your documents first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Invoke the chain and show a spinner while waiting
            with st.spinner("AI is thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_question,
                    "chat_history": [msg for msg in st.session_state.messages if msg["role"] != "user"]
                })
                ai_response = response["answer"]
                
                # --- THIS IS THE SOURCE CITING FEATURE ---
                # Extract the source documents from the context
                source_documents = response["context"]
                source_filenames = set([doc.metadata['source'] for doc in source_documents])
                
                # Format the sources into a readable string
                sources_text = "Sources:\n" + "\n".join(f"- {filename}" for filename in source_filenames)

            # Display the AI's response and the sources
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                # Use an expander to neatly tuck away the sources
                with st.expander("View Sources"):
                    st.info(sources_text)
            
            # Add the full response (with sources) to the message history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response + "\n\n" + sources_text
            })

if __name__ == "__main__":
    main()