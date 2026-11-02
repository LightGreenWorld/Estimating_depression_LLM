"""PHQ-9 counselling chatbot, split into one class per responsibility.

    config.py      Settings            everything you might want to change, in one place
    scale.py       PHQ9Scale           the 9 items and the severity bands (the clinical instrument)
    prompts.py     (constants)         the two system prompts
    knowledge.py   CounselingKnowledge CSV -> documents -> FAISS retriever (the RAG part)
    counselor.py   Counselor           the conversational chain with per-session memory
    scorer.py      PHQ9Scorer          transcript -> Assessment (scores, reasoning, summary)
    report.py      ReportBuilder       Assessment -> printable report
    session.py     CounselingSession   the conversation loop that wires it all together
"""
