# 📚 Multi-Doc Pro: Secure, Conversational RAG for Private Knowledge Bases

**The Production-Grade AI Agent for Private Data.**

**Live Demo:** [https://multi-doc-pro.streamlit.app/](https://multi-doc-pro.streamlit.app/)

| Feature | Status | Frameworks Used |
| :--- | :--- | :--- |
| **Conversational RAG** | ✅ Complete | LangChain Expression Language (LCEL) |
| **LLM Integration** | ✅ High-Speed | Groq (`llama-3.1-8b-instant`) |
| **Vector Store** | ✅ Local Persistence | ChromaDB, Google Embeddings |
| **Deployment** | ✅ Cloud-Deployed | Streamlit Community Cloud |

---

## 🛑 Important License Notice: All Rights Reserved

**This project repository is publicly available for the sole purpose of technical portfolio review, demonstration, and potential employment review by interested parties.**

* **No Copying:** You are **not** permitted to copy, download, reproduce, distribute, or modify the source code of this project.
* **No Reuse:** This work is not open-source, and all code is copyrighted and **All Rights Are Reserved** by the author, Madhan Mohan Naidu Mareneni.
* **Full License Details:** See the complete license terms in the [LICENSE.md](LICENSE.md) file.

---

## 💡 Project Overview: The Need for Private, Grounded AI Agents

`Multi-Doc Pro` is an advanced **Retrieval-Augmented Generation (RAG) system** engineered to deliver accurate, conversational answers based **only** on your private documents. It functions as a specialized **AI Agent** capable of chatting over multiple PDF and text files while maintaining conversation history and accurately citing its sources.

Built with an LCEL-based architecture, this project serves as a robust, production-ready blueprint for **secure enterprise knowledge management**.

## ⚙️ Why Multi-Doc Pro is Fundamentally Different (The RAG Advantage)

This solution addresses critical enterprise needs that general-purpose tools like ChatGPT cannot meet.

| Enterprise Value | Multi-Doc Pro (RAG Agent) | General LLM (e.g., ChatGPT) |
| :--- | :--- | :--- |
| **Data Privacy & Security** | **Private & Local.** Data (embeddings) never leaves the local machine or private server (ChromaDB). **Air-gapped solution.** | **Public/Cloud.** Data is transmitted to a third-party server, creating significant privacy and regulatory risks. |
| **Factual Accuracy** | **Focused Expert.** Answers are strictly **grounded** in the provided documents, dramatically reducing the risk of *hallucination*. | **Generalist.** Prone to **hallucination** and pulling from general training data when asked domain-specific questions. |
| **Customization & Control** | **100% Controllable.** Full control over chunking, prompt engineering (AI personality), LLM choice (e.g., Groq for speed), and retrieval strategy. | **Black Box.** Users have zero control over the RAG architecture or internal logic. |
| **Integration** | **A Building Block (API-Ready).** The RAG core can be exposed as an API for easy integration into Slack bots, internal software, or automated workflows. | **A Destination.** Primarily a web interface, requiring manual interaction. |

## 🚀 Technical Highlights

* **Multi-Turn Conversation:** Implements a `create_history_aware_retriever` chain to intelligently rephrase follow-up questions based on the chat history.
* **High-Speed Generation:** Leverages the **Groq API** for extremely fast, low-latency LLM inference.
* **Source Provenance:** Every response includes a **citation of the exact source document** (filename) used to construct the answer, ensuring verifiability.
* **Robust File Handling:** Includes custom validation to ensure only non-empty, supported file types (`.pdf`, `.txt`) are processed, enhancing application stability.

---

## 🛠️ Technical Architecture

The application is built on a modular architecture using the core principles of an AI Agent pipeline:

1.  **File Loading (`file_handler.py`):** Takes multiple PDFs and TXT files, loads them into LangChain `Document` objects, and cleans/validates them.
2.  **Indexing (`vector_store_handler.py`):** Uses **RecursiveCharacterTextSplitter** and **GoogleGenerativeAIEmbeddings** to index data into a persistent **ChromaDB**.
3.  **The Conversational Chain (`chain_handler.py`):** An advanced LangChain Expression Language (LCEL) chain manages contextualization, retrieval, and final generation using Groq's fast LLM.
4.  **Interface (`app.py`):** Streamlit manages the user experience, file uploads, conversation history, and displays the final output with sources.

## 💻 Getting Started (Local Development)

### Prerequisites

* Python 3.10+
* A `GROQ_API_KEY` (for fast inference)
* A `GOOGLE_API_KEY` (for embeddings)

### Setup

1.  **Clone the repository & install dependencies:**
    ```bash
    git clone [https://github.com/Madhan-mohan14/Multi-Doc-Pro.git](https://github.com/Madhan-mohan14/Multi-Doc-Pro.git)
    cd Multi-Doc-Pro
    pip install -r requirements.txt 
    ```
2.  **Set Environment Variables:** Create a `.env` file in the root directory:
    ```
    GROQ_API_KEY="YOUR_GROQ_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_KEY"
    ```
3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---
## 📄 `LICENSE.md`
Copyright (c) 2025 Madhan Mohan Naidu Mareneni All Rights Reserved.

This project, its source code, datasets, architecture, and design are the exclusive intellectual property of the author.

Unauthorized use, copying, modification, reproduction, or distribution of this material, in whole or in part, is strictly prohibited.

Permission to view this repository or its contents does not grant any rights to reuse or reproduce any part of this project.

For permissions or collaborations, contact the author directly at: https://github.com/Madhan-mohan14
