"""All tunable settings in one dataclass. Change them here, or pass a Settings(...) into main()."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    csv_path: Path = Path("data/train.csv")           # counselling examples with columns Context, Response
    max_docs: int = 200                                # how many CSV rows go into the vector store
    retriever_k: int = 3                               # documents retrieved per user turn
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3:instruct"
    counselor_temperature: float = 0.3                 # a little warmth
    scorer_temperature: float = 0.0                    # scoring must be deterministic
    session_id: str = "Patient-001"
    min_turns_before_exit: int = 5                     # ask to continue if the client leaves too early
