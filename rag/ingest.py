import os
import pickle
import numpy as np
import faiss
import certifi
from tqdm import tqdm
os.environ['SSL_CERT_FILE'] = certifi.where()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

DOCUMENT_PATH = "data/documents"
VECTOR_STORE_PATH = "rag/vector_store"
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
def load_documets():
    documents = []
    for filename in os.listdir(DOCUMENT_PATH):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCUMENT_PATH, filename))
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators = ["\n\n","\n","."," ",""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def generate_embeddings(texts):
    clean_texts = []

    for text in texts:
        # Non string types
        if text is None:
            continue
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                continue

        
        text = text.strip().replace("\x00", "")

      
        if len(text) < 20:
            continue
        if text.lower() in ("none", "null", "nan"):
            continue

     
        try:
            text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

        clean_texts.append(text)

    print(f"Valid chunks after cleaning: {len(clean_texts)}")

    embeddings = embedding_model.encode(
        clean_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings, clean_texts

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def save_vector_store(index, texts):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    faiss.write_index(index,f"{VECTOR_STORE_PATH}/faiss_index")

    with open(f"{VECTOR_STORE_PATH}/texts.pkl","wb") as f:
        pickle.dump(texts,f)

def main():

    print("Loading documents...")
    docs = load_documets()

    print("Chunking the documents...")

    chunks = chunk_documents(docs)
    texts = []
    for chunk in chunks:
        if hasattr(chunk, "page_content") and isinstance(chunk.page_content, str):
            content = chunk.page_content.strip()
            if content:
                texts.append(content)
    
    print(f"Total chujnks created: {len(texts)}")
    print("Generating embeddings...")

    embeddings, texts = generate_embeddings(texts)
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("Saving vector store...")
    save_vector_store(index, texts)
    print("Ingestion Complete !!")
    
if __name__ == "__main__":
    main()
    