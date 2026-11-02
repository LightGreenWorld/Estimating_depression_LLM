# PHQ-9 counselling chatbot (OOP version of `phq9_chatbot.ipynb`)

Same behaviour as the notebook, split into one class per responsibility. Not a diagnostic tool.

```
phq9_chatbot/
├── main.py              entry point: builds the objects and runs a session
├── phq9/
│   ├── config.py        Settings          one dataclass with every knob (model, CSV path, k, session id)
│   ├── scale.py         PHQ9Scale         the 9 items + severity bands + score validation (pure, no LLM)
│   ├── prompts.py       (constants)       the two system prompts
│   ├── knowledge.py     CounselingKnowledge   CSV -> Documents -> FAISS retriever -> context string
│   ├── counselor.py     Counselor         RAG chain with per-session memory; reply(), transcript()
│   ├── scorer.py        PHQ9Scorer, Assessment   transcript -> validated scores/reasoning/summary
│   ├── report.py        ReportBuilder     Assessment -> report text
│   └── session.py       CounselingSession the conversation loop (input/print are injectable)
├── tests/test_phq9.py   6 offline tests with a fake LLM and fake retriever
└── data/train.csv       (put your counselling CSV here: columns Context, Response)
```

## Run

```bash
uv add langchain-core langchain-community langchain-ollama langchain-huggingface faiss-cpu sentence-transformers pytest
ollama serve && ollama pull llama3:instruct
uv run python main.py --csv data/train.csv
uv run pytest -q          # no Ollama needed
```

## How the notebook maps onto the classes

| Notebook cell | Now | Why a class |
| --- | --- | --- |
| `PHQ9`, `SEVERITY_LEVELS`, `get_severity` | `PHQ9Scale` | Clinical constants + the one rule that uses them live together; `validate_scores` added |
| `csv_to_documents`, `setup_rag`, `format_docs` | `CounselingKnowledge` | Loading, indexing and formatting all touch the same documents; `build()` is separate from `load_documents()` so tests can skip the embedding model |
| `setup_chains` (counsellor half), `store`, `get_session_history`, `build_transcript` | `Counselor` | The chain and its memory are one thing; the global `store` dict becomes `self._histories` |
| `setup_chains` (scorer half), `score_conversation` | `PHQ9Scorer` + `Assessment` | The parsing/validation logic is a classmethod so it can be tested with strings, no model |
| `print_report` | `ReportBuilder.render()` | Returns a string instead of printing, so it can be tested, logged or saved |
| `run_conversation`, `main` | `CounselingSession`, `main.py` | `input`/`print` are constructor arguments, so the loop is testable with a scripted client |
| `SESSION_ID`, model names, `k`, `max_docs` | `Settings` | No more magic values scattered across cells |

## Two things changed on purpose

1. **A bug.** The notebook called `scorer_chain.invoke({"Transcript": transcript})` but the prompt's slot is `{transcript}` (lower case). Fixed; the test `test_scorer_parses_good_bad_and_noisy_output` covers the path.
2. **Item 9 (self-harm).** In the notebook's sample run the scorer gave item 9 = 2 although the client never mentioned death or self-harm; it inferred it from hopelessness. The scorer prompt now says to score item 9 above 0 only when the client mentioned it, `Assessment.self_harm_flag` exposes it, and the report prints a safety notice when it is set. Scores from an LLM are validated (`PHQ9Scale.validate_scores`) before anything is summed.

## Reading order

`scale.py` (no LangChain, 50 lines) → `scorer.py` → `report.py` → `knowledge.py` → `counselor.py` → `session.py` → `main.py`. The tests are worked examples of each class.

## Notes

- `RunnableWithMessageHistory` and `ChatMessageHistory` print deprecation warnings in LangChain 1.x; they still work. The next step would be LangGraph's built-in memory, which only touches `Counselor`.
- `max_docs=200` keeps the first run short; raise it in `Settings` once the embedding model is cached.
