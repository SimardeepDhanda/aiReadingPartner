import json
import numpy as np
import faiss
from tqdm import tqdm
from embedder import get_embedding_function

def build_faiss_index(chunks_json_path):
    """
    Build a FAISS index from chunks.json and save it to disk.
    Also saves a parallel metadata file with the same order as embeddings.
    """
    # Load chunks from JSON
    with open(chunks_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Get embedding function
    embedding_function = get_embedding_function()

    # Prepare lists for embeddings and metadata
    embeddings = []
    metadata = []

    # Process each chunk
    print("Computing embeddings...")
    for chunk in tqdm(chunks):
        # Get embedding for the chunk's text
        embedding = embedding_function([chunk["text"]])[0]
        embeddings.append(embedding)
        
        # Store metadata
        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "chapter": chunk["chapter"],
            "percent": chunk["percent"],
            "text": chunk["text"]
        })

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')

    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings_array)

    # Save index and metadata
    faiss.write_index(index, "faiss.index")
    with open("faiss_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Success! FAISS index and metadata saved to disk.")
    return "faiss.index"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from book chunks."
    )
    parser.add_argument("chunks_path", help="Path to chunks.json file")
    args = parser.parse_args()

    result_path = build_faiss_index(args.chunks_path)
    print(f"FAISS index created at: {result_path}") 