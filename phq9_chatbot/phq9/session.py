"""The conversation loop. `ask` and `say` are injectable so the loop can be tested without a keyboard."""
from __future__ import annotations
from typing import Callable
from .config import Settings
from .counselor import Counselor
from .scorer import PHQ9Scorer, Assessment
from .report import ReportBuilder

OPENING_LINE = "Hello, I've been wanting to talk to someone about how I've been feeling lately."
BANNER = ("=" * 60 + "\n"
          "  I'm here to support you. Please talk to me if you need someone to listen.\n"
          "  Type 'exit' when you'd like to take a break or end the conversation.\n"
          + "=" * 60 + "\n")
TOO_EARLY = ("Counselor: We've only just started talking. Could you share a little more so I can better "
             "understand you and your situation? (Type 'exit' again to end anyway.)\n")
GOODBYE = "Counselor: Thank you for sharing with me today. Let me put together a summary for you.\n"


class CounselingSession:
    def __init__(self, counselor: Counselor, scorer: PHQ9Scorer, settings: Settings,
                 ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
                 report: ReportBuilder | None = None):
        self.counselor, self.scorer, self.settings = counselor, scorer, settings
        self.ask, self.say = ask, say
        self.report = report or ReportBuilder()
        self.user_inputs: list[str] = []

    # -- one turn ----------------------------------------------------------------------------------------------
    def _counsel(self, text: str) -> None:
        self.user_inputs.append(text)
        answer = self.counselor.reply(text, self.settings.session_id)
        self.say(f"\nCounselor: {answer}\n")

    # -- the loop ----------------------------------------------------------------------------------------------
    def converse(self) -> list[str]:
        self.say(BANNER)
        opening = self.counselor.reply(OPENING_LINE, self.settings.session_id)
        self.say(f"Counselor: {opening}\n")
        turn = 0
        while True:
            text = self.ask("User: ").strip()
            if not text:
                continue
            turn += 1
            if text.lower() != "exit":
                self._counsel(text)
                continue
            if turn <= self.settings.min_turns_before_exit:          # too early: ask once more
                self.say("\n" + TOO_EARLY)
                confirm = self.ask("User: ").strip()
                if confirm.lower() != "exit":
                    self._counsel(confirm)
                    continue
            self.say("\n" + GOODBYE)
            return self.user_inputs

    # -- end of session ----------------------------------------------------------------------------------------
    def assess(self) -> Assessment | None:
        transcript = self.counselor.transcript(self.settings.session_id)
        if not transcript:
            self.say("No conversation recorded.")
            return None
        self.say("Analyzing conversation for PHQ-9 scoring...\n")
        return self.scorer.score(transcript)

    def run(self) -> Assessment | None:
        self.converse()
        assessment = self.assess()
        if assessment:
            self.say(self.report.render(assessment))
        return assessment
