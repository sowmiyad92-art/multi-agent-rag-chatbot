import os
os.environ['USE_TF'] = '0'

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client

load_dotenv()

PDF_PATH = "data/LIONSGATE-REPORTS-RESULTS-FOR-FOURTH-QUARTER-FISCAL-2026-2026.pdf"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# 3. Embed all chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Push to Supabase
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

for i, chunk in enumerate(chunks):
    clean_content = chunk.page_content.replace("\x00", "")
    vector = embeddings.embed_query(clean_content)
    supabase.table("pdf_rag_documents").insert({
        "content": clean_content,
        "metadata": {"source": PDF_PATH, "page": chunk.metadata.get("page", None)},
        "embedding": vector,
    }).execute()
    print(f"Inserted chunk {i+1}/{len(chunks)}")

print("Done.")