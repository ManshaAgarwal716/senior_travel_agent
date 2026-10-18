import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "travel_knowledge"
print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded.")
client = chromadb.PersistentClient(
    path=CHROMA_DIR
)
collection = client.get_collection(
    name=COLLECTION_NAME
)
print(
    f"Connected to ChromaDB. "
    f"Documents: {collection.count()}"
)
app = FastAPI(
    title="Senior Travel RAG API",
    version="1.0.0")
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
@app.get("/health")
def health():

    return {
        "status": "ok",
        "documents": collection.count()
    }
@app.post("/search")
def search(request: SearchRequest):
    query_embedding = model.encode(
        request.query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        matches.append({
            "text": document,
            "source": metadata.get("source"),
            "distance": distance
        })

    return {
        "query": request.query,
        "results": matches
    }