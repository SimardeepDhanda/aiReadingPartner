import os
import json
import fitz                      # PyMuPDF
from ebooklib import epub, ITEM_DOCUMENT
import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk

# Ensure the NLTK Punkt tokenizer is available
nltk.download("punkt")

def parse_pdf(filepath):
    """
    Extract text page by page from a PDF using PyMuPDF.
    Returns: List of tuples [(page_number, text), ...]
    """
    doc = fitz.open(filepath)
    page_texts = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text")
        page_texts.append((page_idx + 1, text))
    doc.close()
    return page_texts

def parse_epub(filepath):
    """
    Extract text from each chapter in an EPUB using ebooklib.
    Returns: List of tuples [(chapter_index, text), ...]
    """
    book = epub.read_epub(filepath)
    chapter_texts = []
    chapter_counter = 0
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            chapter_counter += 1
            html_content = item.get_content().decode("utf-8")
            # Strip HTML tags with a simple regex
            plain = re.sub(r"<[^>]+>", "", html_content)
            chapter_texts.append((chapter_counter, plain))
    return chapter_texts

def split_into_chunks(page_or_chapter_list, chunk_size=300, is_pdf=True):
    """
    Given a list of (idx, text) pairs (page or chapter),
    break into ~300-word chunks. Record metadata: chunk_id, text, page/chapter, percent.
    Returns: List of chunk dicts.
    """
    # 1) Count total words for percent calculation
    total_words = sum(len(text.split()) for _, text in page_or_chapter_list)

    all_chunks = []
    chunk_id_counter = 0
    words_so_far = 0

    # 2) Build chunks per page/chapter
    for location_idx, text in tqdm(page_or_chapter_list, desc="Building chunks"):
        sentences = sent_tokenize(text)
        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count <= chunk_size:
                current_sentences.append(sentence)
                current_word_count += word_count
            else:
                # finalize this chunk
                chunk_id_counter += 1
                words_so_far += current_word_count
                percent_through = round((words_so_far / total_words) * 100, 2)
                chunk_text = " ".join(current_sentences)
                all_chunks.append({
                    "chunk_id": chunk_id_counter,
                    "text": chunk_text,
                    # we'll rename location_idx below
                    "location_idx": location_idx,
                    "percent": percent_through
                })
                # reset for next chunk
                current_sentences = [sentence]
                current_word_count = word_count

        # leftover sentences at end of this page/chapter
        if current_sentences:
            chunk_id_counter += 1
            words_so_far += current_word_count
            percent_through = round((words_so_far / total_words) * 100, 2)
            chunk_text = " ".join(current_sentences)
            all_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": chunk_text,
                "location_idx": location_idx,
                "percent": percent_through
            })

    # 3) Rename "location_idx" → "page" or "chapter" per is_pdf
    for chunk in all_chunks:
        if is_pdf:
            chunk["page"] = chunk.pop("location_idx")
            chunk["chapter"] = None
        else:
            chunk["chapter"] = chunk.pop("location_idx")
            chunk["page"] = None

    return all_chunks

def save_chunks_to_json(chunks, output_path):
    """
    Write the list of chunk dicts to a JSON file at output_path.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return output_path

def parse_and_chunk(input_filepath, output_dir=".", chunk_size=300):
    """
    Orchestrates: Detect filetype, parse PDF/EPUB, split into chunks,
    save to JSON. Returns path to JSON file.
    """
    filename = os.path.basename(input_filepath)
    name, ext = os.path.splitext(filename.lower())
    if ext == ".pdf":
        page_chapter_list = parse_pdf(input_filepath)
        is_pdf = True
    elif ext == ".epub":
        page_chapter_list = parse_epub(input_filepath)
        is_pdf = False
    else:
        raise ValueError("Unsupported file format: only PDF or EPUB allowed.")

    chunks = split_into_chunks(page_chapter_list, chunk_size=chunk_size, is_pdf=is_pdf)

    output_filename = os.path.join(output_dir, f"{name}_chunks.json")
    save_chunks_to_json(chunks, output_filename)
    return output_filename

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse a PDF or EPUB into ~300-word chunks with metadata."
    )
    parser.add_argument("input_path", help="Path to PDF or EPUB file")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory where chunks JSON will be saved",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=300,
        help="Approximate number of words per chunk",
    )
    args = parser.parse_args()

    result_path = parse_and_chunk(args.input_path, args.output_dir, args.chunk_size)
    print(f"Chunks saved to: {result_path}") 