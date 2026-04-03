"""Lexicon + regex baseline with light negation damping."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SUBSTANCE = [
    r"\b(pills?|percs?|xanax|benzos?|fent|fentanyl|heroin|meth|ice|shard|coke|cocaine|weed|edible|drink|drinking|alcohol|vodka|dope|opioid|naloxone|narcan|suboxone|buprenorphine|methadone|kratom)\b",
    r"\b(using again|relapsed|stash|withdrawal|overdose|OD|high|dosed)\b",
]

_DISTRESS = [
    r"\b(better off without me|hopeless|hollow|drowning|panic attack|can't cope|no point)\b",
    r"\b(mind racing|can't sleep|shaking|terrified|worthless|i hate this)\b",
]

_RECOVERY = [
    r"\b(sober|recovery|sponsor|meeting|na meeting|smart recovery|harm reduction)\b",
    r"\b(clean for|one day at a time|grateful|counselor|reduced my cravings)\b",
]

_NEG = re.compile(r"\b(no|not|never|without|n't)\b", re.I)


@dataclass
class RuleSignalResult:
    substance_score: float
    distress_score: float
    recovery_score: float
    matches: dict[str, list[str]]

    def prediction(self) -> str:
        s, d, r = self.substance_score, self.distress_score, self.recovery_score
        if r >= max(s, d, 0.6) and r > 0:
            return "recovery"
        if s >= d and s > 0.35:
            return "substance_mention"
        if d > s and d > 0.35:
            return "distress"
        return "neutral"


def _apply(text_low: str, patterns: list[str], key: str, matches: dict[str, list[str]]) -> float:
    hits: list[str] = []
    score = 0.0
    for pat in patterns:
        for m in re.finditer(pat, text_low):
            w = 0.35 if _NEG.search(text_low[max(0, m.start() - 48) : m.end()]) else 1.0
            score += w
            hits.append(m.group(0))
    matches[key] = list(dict.fromkeys(hits))[:12]
    return float(min(score, 4.0))


def score_rules(text: str) -> RuleSignalResult:
    t = text.strip().lower()
    matches: dict[str, list[str]] = {}
    sub = _apply(t, _SUBSTANCE, "substance", matches)
    dis = _apply(t, _DISTRESS, "distress", matches)
    rec = _apply(t, _RECOVERY, "recovery", matches)
    return RuleSignalResult(sub, dis, rec, matches)
