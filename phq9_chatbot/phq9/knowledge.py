"""The RAG part: counselling examples (CSV) -> LangChain Documents -> FAISS retriever."""
from __future__ import annotations
import csv
from pathlib import Path
from langchain_core.documents import Document


class CounselingKnowledge:
    """Loads `Context, Response` rows and answers "which examples resemble what the client just said?"."""

    def __init__(self, csv_path: str | Path, max_docs: int = 200, k: int = 3,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.csv_path = Path(csv_path)
        self.max_docs, self.k, self.embedding_model = max_docs, k, embedding_model
        self.documents: list[Document] = []
        self.retriever = None

    # -- loading -----------------------------------------------------------------------------------------------
    def load_documents(self) -> list[Document]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"counselling CSV not found: {self.csv_path}")
        docs = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                content = f"Context: {row['Context']}\nSuggested Response: {row['Response']}"
                docs.append(Document(page_content=content, metadata={"source": self.csv_path.name}))
        self.documents = docs[: self.max_docs]
        return self.documents

    def build(self) -> "CounselingKnowledge":
        """Embed the documents and build the FAISS index. Imports are inside so tests can run without them."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        if not self.documents:
            self.load_documents()
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={"device": "cpu"})
        store = FAISS.from_documents(self.documents, embeddings)
        self.retriever = store.as_retriever(search_kwargs={"k": self.k})
        return self

    # -- querying ----------------------------------------------------------------------------------------------
    def retrieve(self, query: str) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError("call build() first")
        return self.retriever.invoke(query)

    def context_for(self, query: str) -> str:
        """The text that goes into the {context} slot of the counsellor prompt."""
        return "\n\n".join(doc.page_content for doc in self.retrieve(query))

    def __len__(self) -> int:
        return len(self.documents)
