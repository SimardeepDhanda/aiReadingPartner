# AI Reading Partner — Phase 1, 2, 3 & 4
A Streamlit scaffold (file upload + progress UI) for a no-spoiler book discussion app.

## Setup

```bash
git clone git@github.com:SimardeepDhanda/aiReadingPartner.git
cd aiReadingPartner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Purpose
This sprint focuses on building the basic UI components for the AI Reading Partner application. The current implementation allows users to:
- Upload PDF or EPUB files
- Enter book titles
- Set and track reading progress (chapter, page, and percentage)
- Parse uploaded books into ~300-word chunks with metadata
- Generate vector embeddings and store them in a local ChromaDB database
- Build a FAISS index for fast similarity search
- Chat about the book with spoiler protection

Note: This phase includes basic file parsing, chunking, embedding, and chat functionality, but does not yet include advanced RAG (Retrieval-Augmented Generation) features. Those will be implemented in future sprints.

## Sprint 2
The application now includes a book parsing module that:
1. Takes a PDF or EPUB file and splits it into ~300-word chunks
2. Annotates each chunk with page, chapter, and percent-through metadata
3. Saves all chunks as a JSON array to disk

You can use the parser in two ways:

1. Command Line:
```bash
python parser.py path/to/example.pdf --output_dir .
# Should print: "Chunks saved to: example_chunks.json"
```

2. Streamlit UI:
```bash
streamlit run app.py
# Upload a PDF/EPUB → Click "Parse Book into Chunks" → confirm success message
```

After parsing, you'll find a `*_chunks.json` file in the project directory. This JSON array contains chunk objects, each with:
- chunk_id
- text
- page/chapter number
- percent through the book

## Sprint 3: Embedding + Vector Store
The application now includes vector embedding functionality that:
1. Takes the chunks.json file and converts each chunk's text into a vector embedding
2. Stores these embeddings in a local ChromaDB collection
3. Preserves all metadata (chunk_id, page, chapter, percent) with each embedding

### Embedding Options
- If OPENAI_API_KEY is set in your environment, the app will use OpenAI's text-embedding-3-small model
- Otherwise, it falls back to the local SentenceTransformers model (all-MiniLM-L6-v2)

### Usage
You can generate embeddings in two ways:

1. Command Line:
```bash
python embedder.py path/to/book_chunks.json --collection_name ai_reading_partner
# Creates a ChromaDB store at ./chroma_db/ai_reading_partner
```

2. Streamlit UI:
```bash
streamlit run app.py
# Upload book → Parse into chunks → Click "Generate Embeddings (RAG Base)"
```

The vector store will be created at `./chroma_db/ai_reading_partner` and can be used as a foundation for future RAG implementations.

## Sprint 4: Basic Chatbot with Spoiler Filtering
The application now includes a FAISS-based search and chat interface that:
1. Builds a FAISS index from the book chunks for fast similarity search
2. Provides a chat interface that filters out spoilers based on reading progress
3. Uses GPT-4 (if OPENAI_API_KEY is set) or GPT-3.5-turbo for generating responses

### Building the FAISS Index
You can build the FAISS index in two ways:

1. Command Line:
```bash
python faiss_indexer.py path/to/book_chunks.json
# Creates faiss.index and faiss_metadata.json
```

2. Streamlit UI:
```bash
streamlit run app.py
# Upload book → Parse chunks → Generate embeddings → Click "Build FAISS Index"
```

### Using the Chat Interface
1. Set your current reading progress (chapter and percentage)
2. Type your question in the chat input
3. The system will:
   - Find the most relevant passages using FAISS
   - Filter out any passages beyond your current reading progress
   - Use up to 5 relevant passages to generate a spoiler-safe response
   - Display the response in the chat history

### Model Selection
- If OPENAI_API_KEY is set, the app will use GPT-4 for more accurate and detailed responses
- Otherwise, it falls back to GPT-3.5-turbo for basic responses

The chat interface ensures you can discuss the book without encountering spoilers, making it safe to ask questions about what you've read so far.