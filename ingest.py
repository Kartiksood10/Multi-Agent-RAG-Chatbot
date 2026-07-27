# This file handles document ingestion — reading, chunking, embedding, and
# storing document content into PostgreSQL with pgvector for our RAG Agent

# Storing env variables
import os
import re
from dotenv import load_dotenv

# LlamaIndex is used for reading PDF/DOCX files and splitting them into chunks
from llama_index.core import SimpleDirectoryReader        # Reads files from disk
from llama_index.core.node_parser import SentenceSplitter # Splits documents into chunks


# We use the openai package to call NVIDIA's embedding endpoint
from openai import OpenAI


# psycopg2 is the PostgreSQL driver for Python
import psycopg2

# To allow access to Nvidia embedding API
import httpx


# register_vector teaches psycopg2 how to read/write the pgvector "vector" type
from pgvector.psycopg2 import register_vector


# Load .env file so os.getenv() can access NVIDIA_API_KEY, POSTGRES_HOST, etc.
load_dotenv()


# We point the OpenAI client to NVIDIA's API base URL instead of OpenAI's
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA NIM API endpoint
    api_key=os.getenv("NVIDIA_API_KEY"),             # Your free NVIDIA API key from .env
    http_client=httpx.Client(verify=False, timeout=60.0),
)


# Database connection helper
def get_db_connection():
    """
    Creates and returns a connection to PostgreSQL.
    register_vector() is called so psycopg2 knows how to handle the vector column type.
    """
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    register_vector(conn)  # Enables reading/writing vector(1024) columns via psycopg2
    return conn


# Text cleaning helper
def clean_text(text: str) -> str:
    """
    Cleans extracted text before chunking / embedding:
      - removes null characters
      - collapses repeated whitespace/newlines into single spaces
      - strips leading/trailing whitespace

    Why this is needed:
      PDFs often extract with broken spacing/newlines.
      If we embed messy text, similarity search quality becomes poor.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Embedding function
def get_embedding(text: str) -> list[float]:
    """
    Converts a piece of text into a 1024-dimensional embedding vector
    using NVIDIA's nv-embedqa-e5-v5 model.

    input_type="passage" tells the model this is a document chunk being stored.
    """
    response = nvidia_client.embeddings.create(
        input=[text],                             # The text to embed (must be a list)
        model="nvidia/nv-embedqa-e5-v5",         # NVIDIA embedding model
        encoding_format="float",                 # Return as floating point numbers
        extra_body={
            "input_type": "passage",             # "passage" = document chunk being stored
            "truncate": "END"                    # If text exceeds token limit, cut from the end
        },
    )
    return response.data[0].embedding            # Returns a list of 1024 float values


# Main ingestion function
def ingest_document(file_path: str, original_filename: str, progress_callback=None) -> int:
    """
    Full ingestion pipeline for one document:
      1. Read the PDF or DOCX file
      2. Split into ~500-token chunks with overlap
      3. Embed each chunk using NVIDIA NIM
      4. Store chunk text + embedding into PostgreSQL

    Args:
        file_path         : Path to the temp file saved by Streamlit
        original_filename : The real file name shown in UI (e.g. "hr_policy.pdf")
        progress_callback : Optional function(current, total) to update the UI progress bar

    Returns:
        Total number of chunks stored in the database
    """

    # Step 1: Load the document using LlamaIndex
    # SimpleDirectoryReader auto-detects PDF vs DOCX based on file extension
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    if not documents:
        raise ValueError("No content could be extracted from the uploaded document.")

    # Debug: inspect raw extracted text before chunking
    # This helps verify whether the parser is reading the PDF correctly
    raw_text_found = False
    for idx, doc in enumerate(documents):
        raw_doc_text = getattr(doc, "text", "") or ""
        cleaned_preview = clean_text(raw_doc_text)

        print(f"\n--- RAW DOC {idx} LENGTH: {len(cleaned_preview)} ---")
        print(cleaned_preview[:1000])

        if cleaned_preview:
            raw_text_found = True

    if not raw_text_found:
        raise ValueError("Document was loaded but no readable text was extracted.")

    # Step 2: Split document into chunks
    # chunk_size=500  → each chunk is at most 500 tokens
    # chunk_overlap=80 → last 80 tokens of chunk N are repeated at start of chunk N+1
    # Overlap ensures sentences at chunk boundaries are not cut off and lose context
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=80)
    nodes = splitter.get_nodes_from_documents(documents)  # List of chunk objects

    # Step 2.1: Clean and validate chunks before storing
    # We skip tiny/noisy chunks because they reduce retrieval quality
    prepared_chunks = []
    for i, node in enumerate(nodes):
        chunk_text = clean_text(node.get_content())  # Extract and clean raw text from chunk

        if not chunk_text:
            continue  # Skip empty chunks

        if len(chunk_text) < 80:
            continue  # Skip tiny chunks — usually noise, page numbers, broken text, etc.

        prepared_chunks.append((i, chunk_text))

    total = len(prepared_chunks)  # Total number of valid chunks after cleaning/filtering

    if total == 0:
        raise ValueError("No valid chunks were produced from the document.")

    # Debug: inspect first few chunks before embedding
    print(f"\nTotal valid chunks prepared: {total}")
    for idx, (_, chunk_text) in enumerate(prepared_chunks[:5]):
        print(f"\n--- CHUNK PREVIEW {idx} ---")
        print(chunk_text[:1000])

    # Step 3: Connect to PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # If this document was previously ingested, delete its old chunks first
        # This allows re-uploading and re-ingesting the same file cleanly
        cur.execute("DELETE FROM document_chunks WHERE doc_name = %s", (original_filename,))

        # Step 4: Loop through every chunk, embed it, and store it
        for current_position, (chunk_index, chunk_text) in enumerate(prepared_chunks, start=1):

            # Call NVIDIA API to convert this chunk's text into a vector
            embedding = get_embedding(chunk_text)

            # Insert the chunk into PostgreSQL
            # chunk_index  = position of this chunk in the original document (0-based)
            # chunk_length = number of characters in chunk_text (useful for debugging/filtering)
            cur.execute(
                """
                INSERT INTO document_chunks (doc_name, chunk_text, embedding, chunk_index, chunk_length)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (original_filename, chunk_text, embedding, chunk_index, len(chunk_text)),
            )

            # If a progress callback was provided (from Streamlit), call it to update the bar
            if progress_callback:
                progress_callback(current_position, total)

        # Commit all inserts to PostgreSQL as a single transaction
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()

    return total  # Return chunk count so the UI can display "X chunks stored"


# Helper: list all ingested documents
def list_ingested_documents() -> list[str]:
    """
    Queries PostgreSQL for all unique document names that have been ingested.
    Called by app.py on every page load to populate the sidebar document list.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT doc_name FROM document_chunks ORDER BY doc_name")
    docs = [row[0] for row in cur.fetchall()]  # Extract doc names from result rows
    cur.close()
    conn.close()
    return docs  # e.g. ["hr_policy.pdf", "tech_manual.docx"]


# Helper: delete a document
def delete_document(doc_name: str):
    """
    Deletes all chunks belonging to a specific document from PostgreSQL.
    Called when the user clicks the trash icon next to a document in the sidebar.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM document_chunks WHERE doc_name = %s", (doc_name,))
    conn.commit()
    cur.close()
    conn.close()