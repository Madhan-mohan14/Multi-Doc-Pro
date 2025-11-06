# chain_handler.py (Updated for Day 25)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever,
)
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)
from langchain_groq import ChatGroq

def create_advanced_rag_chain(retriever):
    """
    Creates an advanced RAG chain that is aware of chat history.

    Args:
        retriever: A LangChain Retriever object.

    Returns:
        A runnable LCEL chain ready for conversation.
    """
    print("Creating the advanced, history-aware RAG chain...")
    
    # 1. The LLM - Using Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # 2. Contextualization Prompt (for rephrasing the question)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 3. History-Aware Retriever
    # This chain takes history and a new question, and creates a search query.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # 4. Main Answering Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise."
            "\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # 5. Document Combining Chain
    # This chain takes the context documents and formats them into the prompt.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 6. The Final Retrieval Chain
    # This master chain connects everything together.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    print("Advanced RAG chain created successfully.")
    return rag_chain

# Daily Deliverable: This updated 'chain_handler.py' file.