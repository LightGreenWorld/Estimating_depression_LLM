"""Entry point: wire the parts together and run one counselling session.

  python main.py                       # needs Ollama running with llama3:instruct, and data/train.csv
  python main.py --csv path/to.csv --model llama3:instruct
"""
import argparse
from pathlib import Path
from phq9.config import Settings
from phq9.knowledge import CounselingKnowledge
from phq9.counselor import Counselor
from phq9.scorer import PHQ9Scorer
from phq9.session import CounselingSession


def build_session(settings: Settings) -> CounselingSession:
    from langchain_ollama import ChatOllama
    print("Loading counseling data and building the vector store (first run downloads the embedding model)...")
    knowledge = CounselingKnowledge(settings.csv_path, settings.max_docs, settings.retriever_k, settings.embedding_model).build()
    print(f"Vector store ready with {len(knowledge)} documents.\n")
    counselor = Counselor(knowledge, ChatOllama(model=settings.llm_model, temperature=settings.counselor_temperature))
    scorer = PHQ9Scorer(ChatOllama(model=settings.llm_model, temperature=settings.scorer_temperature))
    return CounselingSession(counselor, scorer, settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Settings.csv_path)
    ap.add_argument("--model", default=Settings.llm_model)
    ap.add_argument("--session", default=Settings.session_id)
    args = ap.parse_args()
    settings = Settings(csv_path=args.csv, llm_model=args.model, session_id=args.session)
    build_session(settings).run()


if __name__ == "__main__":
    main()
