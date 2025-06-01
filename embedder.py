import os
import json
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import openai
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        embeddings = self.model.encode(input)
        return embeddings.tolist()

def get_embedding_function():
    """
    Returns a SentenceTransformer wrapper for local embeddings.
    """
    return SentenceTransformerEmbeddingFunction()

def embed_chunks_to_chroma(chunks_json_path, collection_name="ai_reading_partner"):
    """
    Load chunks from JSON, create a ChromaDB collection, and add the chunks with embeddings.
    """
    # Load chunks from JSON
    with open(chunks_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db/ai_reading_partner")

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection didn't exist or other error, continue anyway

    # Create new collection with embedding function
    try:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=get_embedding_function()
        )
    except Exception as e:
        # If collection already exists, get it
        if "already exists" in str(e):
            collection = client.get_collection(
                name=collection_name,
                embedding_function=get_embedding_function()
            )
        else:
            raise e

    # Prepare data for batch addition
    ids = []
    documents = []
    metadatas = []

    # Process each chunk
    for chunk in tqdm(chunks, desc="Adding chunks to ChromaDB"):
        ids.append(str(chunk["chunk_id"]))
        documents.append(chunk["text"])
        # Convert None values to empty strings for metadata
        metadatas.append({
            "chunk_id": str(chunk["chunk_id"]),
            "page": str(chunk["page"]) if chunk["page"] is not None else "",
            "chapter": str(chunk["chapter"]) if chunk["chapter"] is not None else "",
            "percent": str(chunk["percent"]) if chunk["percent"] is not None else ""
        })

    # Add all chunks to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    storage_dir = os.path.abspath("./chroma_db/ai_reading_partner")
    print(f"Success! Vector store created at: {storage_dir}")
    return storage_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert book chunks into vector embeddings and store in ChromaDB."
    )
    parser.add_argument("chunks_path", help="Path to chunks.json file")
    parser.add_argument(
        "--collection_name",
        default="ai_reading_partner",
        help="Name for the ChromaDB collection"
    )
    args = parser.parse_args()

    result_path = embed_chunks_to_chroma(args.chunks_path, args.collection_name)
    print(f"ChromaDB store created at: {result_path}") 