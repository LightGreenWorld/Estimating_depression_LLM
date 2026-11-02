"""pytest -q    (offline: fake LLM and fake retriever; no Ollama, no HuggingFace download)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document
from phq9.config import Settings
from phq9.scale import PHQ9Scale
from phq9.knowledge import CounselingKnowledge
from phq9.counselor import Counselor
from phq9.scorer import PHQ9Scorer, Assessment
from phq9.report import ReportBuilder
from phq9.session import CounselingSession


class FakeKnowledge(CounselingKnowledge):
    """Skips FAISS/HuggingFace: always 'retrieves' the first k documents."""
    def build(self):
        self.load_documents()
        return self

    def retrieve(self, query):
        return self.documents[: self.k]


def make_csv(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("Context,Response\n\"I feel sad\",\"It is okay to feel sad; tell me more.\"\n"
                 "\"I can't sleep\",\"Sleep troubles are exhausting. What is on your mind at night?\"\n", encoding="utf-8")
    return p


def test_scale_severity_bands_and_validation():
    assert [PHQ9Scale.severity(t) for t in (0, 4, 5, 9, 10, 14, 15, 19, 20, 27)] == \
        ["None to Minimal"] * 2 + ["Mild"] * 2 + ["Moderate"] * 2 + ["Moderately severe"] * 2 + ["Severe"] * 2
    assert PHQ9Scale.severity(28) == "Unknown"
    assert PHQ9Scale.validate_scores(["2", 1, 0, 3, 0, 0, 0, 0, 1]) == [2, 1, 0, 3, 0, 0, 0, 0, 1]
    for bad in ([1] * 8, [4] + [0] * 8, [-1] + [0] * 8):
        try:
            PHQ9Scale.validate_scores(bad); assert False, bad
        except ValueError:
            pass


def test_knowledge_loads_csv_and_formats_context(tmp_path):
    k = FakeKnowledge(make_csv(tmp_path), max_docs=1, k=3).build()
    assert len(k) == 1 and k.documents[0].page_content.startswith("Context: I feel sad")
    assert "Suggested Response:" in k.context_for("anything")


def test_scorer_parses_good_bad_and_noisy_output():
    good = json.dumps({"scores": [2, 3, 1, 2, 2, 3, 0, 0, 0], "reasoning": ["r"] * 9, "summary": "low mood"})
    a = PHQ9Scorer.parse("Sure! Here is the JSON:\n" + good + "\nHope this helps.")       # chatter around JSON
    assert a.parsed and a.total == 13 and a.severity == "Moderate" and a.summary == "low mood" and not a.self_harm_flag
    b = PHQ9Scorer.parse("I cannot score this.")
    assert not b.parsed and b.scores == [0] * 9 and b.severity == "None to Minimal"
    c = PHQ9Scorer.parse(json.dumps({"scores": [0] * 8}))                                 # wrong length
    assert not c.parsed
    d = PHQ9Scorer.parse(json.dumps({"scores": [0] * 8 + [1]}))
    assert d.self_harm_flag


def test_report_renders_total_severity_and_safety_notice():
    text = ReportBuilder().render(Assessment([3, 3, 2, 2, 2, 3, 1, 0, 1], ["why"] * 9, "summary here"))
    assert "TOTAL SCORE:" in text and "17/27" in text and "Moderately severe" in text
    assert "SAFETY" in text and "CLINICAL SUMMARY" in text and "NOT a clinical diagnosis" in text
    assert "WARNING" in ReportBuilder().render(PHQ9Scorer.parse("garbage"))


def test_counselor_keeps_history_and_builds_transcript(tmp_path):
    knowledge = FakeKnowledge(make_csv(tmp_path)).build()
    c = Counselor(knowledge, FakeListChatModel(responses=["Tell me more.", "How is your sleep?"]))
    assert c.reply("Hi", "s1") == "Tell me more."
    assert c.reply("Not great", "s1") == "How is your sleep?"
    t = c.transcript("s1")
    assert t.splitlines() == ["Client: Hi", "Counselor: Tell me more.", "Client: Not great", "Counselor: How is your sleep?"]
    assert c.transcript("other") == ""                      # sessions are separate


def test_session_loop_early_exit_then_report(tmp_path):
    knowledge = FakeKnowledge(make_csv(tmp_path)).build()
    counselor = Counselor(knowledge, FakeListChatModel(responses=["Welcome.", "Go on.", "I see."]))
    scorer = PHQ9Scorer(FakeListChatModel(responses=[json.dumps({"scores": [1] * 9, "reasoning": [""] * 9, "summary": "s"})]))
    answers = iter(["I feel low", "exit", "a bit more", "exit", "exit"])
    said = []
    s = CounselingSession(counselor, scorer, Settings(min_turns_before_exit=5), ask=lambda _: next(answers), say=said.append)
    a = s.run()
    assert s.user_inputs == ["I feel low", "a bit more"]      # 'exit' is never sent to the counsellor
    assert any("only just started" in x for x in said)       # early-exit nudge shown
    assert a.total == 9 and any("PHQ-9 DEPRESSION ASSESSMENT REPORT" in x for x in said)
