# Multi-doc Chatbot

A Streamlit-powered conversational assistant for asking questions over multiple PDF documents. This project uses LangChain, Groq LLM, Hugging Face embedding models, and Chroma vector store to build a retriever-based chatbot.

## Features

- Load PDF documents from the `data/` folder
- Split documents into embedding-friendly chunks
- Vectorize document chunks and store them in `vector_db_dir/`
- Use Groq `ChatGroq` as the chat LLM
- Support conversational retrieval with source citations shown in the UI
- Simple Streamlit interface for question input and chat history

## Repository Structure

- `main.py` - Streamlit application and conversational retrieval chain
- `vectorize_documents.py` - Script to load PDFs, split text, and build the vector database
- `config.json` - Local config file holding the Groq API key (ignored by git)
- `data/` - Folder for PDF documents to index
- `vector_db_dir/` - Persistent Chroma vector store directory
- `requirements.txt` - Python dependencies
- `.gitignore` - Ignored files including `config.json`, `.venv/`, and local data directories

## Requirements

- Python 3.10+ recommended
- `pip` available
- A valid Groq API key

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create `config.json` at the project root with your Groq API key:

```json
{
  "GROQ_API_KEY": "your_groq_api_key_here"
}
```

4. Add your PDF documents to the `data/` directory.

## Build the Vector Store

Run the vectorization script once after adding or updating documents:

```bash
python vectorize_documents.py
```

This will:

- load PDFs from `data/`
- split each document into text chunks
- create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- persist the vector store to `vector_db_dir/`

## Run the App

Start the Streamlit app:

```bash
streamlit run main.py
```

Open the browser URL shown by Streamlit and ask questions about the loaded documents.

## Notes

- `config.json` is intentionally excluded from Git to protect your API key.
- If you update or add documents, rerun `python vectorize_documents.py` to refresh the vector database.
- The current retrieval chain uses `k=5` to return the top 5 source documents.

## Troubleshooting

- If the app fails to start, verify your Groq API key and that `config.json` exists.
- If no documents are found, ensure PDFs are present in the `data/` directory and the `vectorize_documents.py` script completes successfully.

## License

This repository does not include a licensed file. Add one if you want to define usage terms.
