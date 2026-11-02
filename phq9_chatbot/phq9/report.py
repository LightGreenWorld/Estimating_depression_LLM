"""Turns an Assessment into the printed report. Returns a string, so it can be tested and saved, not only printed."""
from __future__ import annotations
from .scale import PHQ9Scale
from .scorer import Assessment


class ReportBuilder:
    WIDTH = 65
    DISCLAIMER = ("NOTE: This is an AI-assisted estimation, NOT a clinical diagnosis.\n"
                  "Please consult a licensed psychiatrist for proper evaluation.")
    SELF_HARM_NOTICE = ("SAFETY: the client may have expressed thoughts of self-harm (item 9 > 0).\n"
                        "This needs a human clinician's attention today, regardless of the total score.")

    def render(self, a: Assessment) -> str:
        bar, rule = "=" * 60, "-" * self.WIDTH
        out = [bar, "        PHQ-9 DEPRESSION ASSESSMENT REPORT", bar, ""]
        if not a.parsed:
            out += ["WARNING: the scorer's answer could not be parsed; all items show 0. Raw answer:", a.raw, ""]
        out.append(f"{'#':<4} {'PHQ-9 Item':<55} {'Score':<6}")
        out.append(rule)
        for i, (item, score) in enumerate(zip(PHQ9Scale.ITEMS, a.scores), 1):
            out.append(f"{i:<4} {item[:55]:<55} {score}/{PHQ9Scale.MAX_ITEM_SCORE}")
            reason = a.reasoning[i - 1]
            if reason and reason != "Unable to parse":
                out.append(f"     Reason: {reason}")
        out.append(rule)
        out.append(f"{'TOTAL SCORE:':<60} {a.total}/{PHQ9Scale.MAX_TOTAL}")
        out.append(f"{'SEVERITY:':<60} {a.severity}")
        out.append("")
        if a.self_harm_flag:
            out += [self.SELF_HARM_NOTICE, ""]
        if a.summary and a.parsed:
            out += ["CLINICAL SUMMARY:", a.summary, ""]
        out += [self.DISCLAIMER, bar]
        return "\n".join(out)
