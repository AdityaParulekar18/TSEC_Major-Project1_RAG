import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# ---- Config ----
JSON_FILE = "kbase.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_STORE_PATH = "faiss_index"

# ---- Load JSON ----
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)["knowledge_base"]

# ---- Convert to LangChain Documents ----
documents = []
for item in data:
    text = (
        f"Crop: {item['crop']}\n"
        f"Disease: {item['disease']}\n"
        f"Symptoms: {item['symptoms']}\n"
        f"Causes: {item['causes']}\n"
        f"Treatment: {item['treatment']}\n"
        f"Prevention: {item['prevention']}"
    )
    metadata = {"crop": item["crop"], "disease": item["disease"]}
    documents.append(Document(page_content=text, metadata=metadata))

# ---- Create Embeddings ----
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---- Create FAISS Store ----
vector_store = FAISS.from_documents(documents, embeddings)

# ---- Save FAISS Index ----
vector_store.save_local(FAISS_STORE_PATH)

print(f"✅ Embeddings created and saved to {FAISS_STORE_PATH}")
