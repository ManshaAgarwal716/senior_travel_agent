import os
import chromadb
from sentence_transformers import SentenceTransformer
KNOWLEDGE_DIR = "knowledge"
CHROMA_DIR = "chroma_db"
print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="travel_knowledge"
)
def chunk_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


documents = []
metadatas = []
ids = []

for filename in os.listdir(KNOWLEDGE_DIR):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(KNOWLEDGE_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        documents.append(chunk)

        metadatas.append({
            "source": filename
        })

        ids.append(
            f"{filename}-{i}"
        )

print(f"Found {len(documents)} chunks.")

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Ingestion complete!")

print(
    "Total documents:",
    collection.count()
)