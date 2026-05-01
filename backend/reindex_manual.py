from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.research_agent.config import get_settings
from src.research_agent.retrieval.dense import DenseRetriever

settings = get_settings()
dense = DenseRetriever(settings)

paper_id = "09aa04bf6edf4481b696d0d9179ede2b"
text_path = Path(f"storage/papers/{paper_id}.txt")
full_text = text_path.read_text(encoding="utf-8")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(full_text)

# Delete old
dense.delete_paper(paper_id)

documents = [
    Document(
        page_content=chunk,
        metadata={
            "paper_id": paper_id,
            "chunk_index": i,
            "filename": "Frame-level Video-Based Temporal Analysisof FPS Gameplay.pdf",
        },
    )
    for i, chunk in enumerate(chunks)
]

dense.upsert_documents(documents)
print(f"Re-indexed {len(chunks)} chunks for paper {paper_id}")