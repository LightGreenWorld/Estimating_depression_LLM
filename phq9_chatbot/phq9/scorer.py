"""The scorer: transcript -> Assessment. The LLM proposes scores; this class validates them."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .prompts import SCORER_SYSTEM_PROMPT
from .scale import PHQ9Scale


@dataclass
class Assessment:
    scores: list[int]                              # 9 ints, 0..3
    reasoning: list[str] = field(default_factory=lambda: [""] * PHQ9Scale.N_ITEMS)
    summary: str = ""
    parsed: bool = True                            # False when the model's output could not be read
    raw: str = ""

    @property
    def total(self) -> int:
        return sum(self.scores)

    @property
    def severity(self) -> str:
        return PHQ9Scale.severity(self.total)

    @property
    def self_harm_flag(self) -> bool:
        return self.scores[PHQ9Scale.SELF_HARM_ITEM - 1] > 0


class PHQ9Scorer:
    JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SCORER_SYSTEM_PROMPT),
            ("human", "Here is the full conversation transcript:\n\n{transcript}"),
        ])
        self.chain = prompt | llm | StrOutputParser()

    def score(self, transcript: str) -> Assessment:
        raw = self.chain.invoke({"transcript": transcript})
        return self.parse(raw)

    @classmethod
    def parse(cls, raw: str) -> Assessment:
        """Pull the JSON out of whatever the model wrote and validate it. Never raises: a bad answer becomes
        an Assessment with parsed=False and all zeros, so the report can say so instead of crashing."""
        match = cls.JSON_BLOCK.search(raw or "")
        if match:
            try:
                data = json.loads(match.group())
                scores = PHQ9Scale.validate_scores(data["scores"])
                reasoning = list(data.get("reasoning", []))[: PHQ9Scale.N_ITEMS]
                reasoning += [""] * (PHQ9Scale.N_ITEMS - len(reasoning))
                return Assessment(scores, reasoning, str(data.get("summary", "")), parsed=True, raw=raw)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
        return Assessment([0] * PHQ9Scale.N_ITEMS, ["Unable to parse"] * PHQ9Scale.N_ITEMS, raw or "", parsed=False, raw=raw or "")
