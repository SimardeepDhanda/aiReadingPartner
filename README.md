# 📚 AI Reading Partner

AI Reading Partner is a no-spoiler, AI-powered book discussion web app built with Streamlit, FAISS, and ChromaDB. It allows users to upload books, track reading progress, and have intelligent, spoiler-free conversations about their current read using GPT models and semantic search.

---

## 🚀 Features

- 📂 **Book Upload & Parsing**
  - Supports PDF and EPUB formats
  - Parses books into ~300-word text chunks
  - Each chunk is annotated with:
    - `chunk_id`
    - `page`, `chapter`, and `percent_through` metadata
  - Outputs structured chunks as a `.json` file

- 🧠 **Vector Embedding & Storage**
  - Generates vector embeddings for each chunk
  - Stores embeddings and metadata in ChromaDB
  - Uses:
    - OpenAI's `text-embedding-3-small` if `OPENAI_API_KEY` is set
    - SentenceTransformers (`all-MiniLM-L6-v2`) as fallback

- 🔍 **Semantic Search & FAISS Index**
  - Builds a FAISS index for fast similarity search over book chunks
  - Stores index and metadata locally
  - Enables fast retrieval of relevant, in-progress-safe content

- 💬 **Spoiler-Free Chat Interface**
  - Users set current reading progress (chapter/page/percentage)
  - Queries retrieve relevant chunks up to that progress
  - Response is generated using GPT-4 or GPT-3.5-turbo
  - Up to 5 relevant, spoiler-free chunks are included in the prompt

---

## 💻 Installation

```bash
git clone git@github.com:SimardeepDhanda/aiReadingPartner.git
cd aiReadingPartner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py


### Model Selection
- If OPENAI_API_KEY is set, the app will use GPT-4 for more accurate and detailed responses
- Otherwise, it falls back to GPT-3.5-turbo for basic responses

The chat interface ensures you can discuss the book without encountering spoilers, making it safe to ask questions about what you've read so far.
