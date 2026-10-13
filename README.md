# RAG PDF Chatbot

An AI-powered PDF Question-Answering Chatbot built using Python, Streamlit, ChromaDB, and Ollama. It allows users to upload a PDF document and ask questions about its content using Retrieval-Augmented Generation (RAG).

## Features

- Upload and process PDF documents
- Extract text and split it into overlapping chunks
- Generate text embeddings using `nomic-embed-text`
- Retrieve relevant passages using ChromaDB
- Generate answers using `llama3.2:3b`
- Maintain conversation history for follow-up questions
- Display source page numbers and retrieved PDF passages
- Show processing progress and basic error messages

## Technologies Used

- Python
- Streamlit
- PyPDF
- ChromaDB
- Ollama
- Retrieval-Augmented Generation (RAG)
- Text Embeddings and Semantic Search

## How It Works

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the text into smaller chunks.
4. Generate embeddings for each chunk.
5. Store the embeddings and text in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve relevant passages from the PDF.
8. Send the retrieved context and question to the language model.
9. Display the answer with source page numbers.

## Prerequisites

- Python installed
- Ollama installed and running
- The required Ollama models downloaded

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-pdf-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Project Structure

```text
rag-pdf-chatbot/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Current Limitations

- The application requires Ollama and the specified models to be available.
- Scanned PDFs without extractable text may not work.
- The application retrieves the top three relevant text chunks for each question.

## Future Improvements

- Support multiple PDF documents
- Improve follow-up question retrieval
- Add a more polished user interface
- Evaluate retrieval quality and answer accuracy
- Deploy the application for public access

## Learning Outcomes

This project demonstrates practical experience with Python, embeddings, vector databases, semantic search, language models, and Retrieval-Augmented Generation.

## Author

Raavi Naveen Kumar