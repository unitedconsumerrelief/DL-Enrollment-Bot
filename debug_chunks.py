import os
import openai
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the same chunks and index as the main bot
from slack_doc_bot import load_documents, embed_chunks, create_vector_index, get_top_chunks, is_valid_primary_chunk

print("🔍 Loading documents and creating index...")
chunks, chunk_sources = load_documents()
print(f"📚 Loaded {len(chunks)} total chunks")

# Create embeddings and index
vectors = embed_chunks(chunks)
index = create_vector_index(vectors)

# Test the mortgage loans question
question = "Does elevate or clarity accept mortgage loans?"
print(f"\n🔍 Testing question: {question}")

# Get top chunks
top_chunks = get_top_chunks(question, k=10)
print(f"\n📋 Found {len(top_chunks)} top chunks:")

for i, (chunk, source) in enumerate(top_chunks):
    print(f"\n--- Chunk {i+1} (Source: {source}) ---")
    print(f"Valid: {is_valid_primary_chunk(chunk, source)}")
    print(f"Length: {len(chunk.split())} words")
    print(f"Content: {chunk[:200]}...")

# Show all chunks that contain "mortgage"
print(f"\n🔍 Searching for chunks containing 'mortgage':")
mortgage_chunks = []
for i, (chunk, source) in enumerate(zip(chunks, chunk_sources)):
    if "mortgage" in chunk.lower():
        mortgage_chunks.append((chunk, source, i))

print(f"Found {len(mortgage_chunks)} chunks with 'mortgage':")
for chunk, source, idx in mortgage_chunks:
    print(f"\n--- Chunk {idx} (Source: {source}) ---")
    print(f"Valid: {is_valid_primary_chunk(chunk, source)}")
    print(f"Content: {chunk}") 