"""One-paragraph template summary for analysts."""


def weekly_narrative(counts_row: dict, distress_spike: bool, substance_spike: bool) -> str:
    parts = [
        "Weekly snapshot (demo corpus; synthetic time): ",
        f"distress ≈ {counts_row.get('distress', 0)}, "
        f"substance_mention ≈ {counts_row.get('substance_mention', 0)}, "
        f"recovery ≈ {counts_row.get('recovery', 0)}, neutral ≈ {counts_row.get('neutral', 0)}. ",
    ]
    if distress_spike:
        parts.append("Distress count exceeded a z-score threshold (coarse flag). ")
    if substance_spike:
        parts.append("Substance-mention count also flagged. ")
    if not distress_spike and not substance_spike:
        parts.append("No coarse spike flags on these aggregates. ")
    parts.append("Validate on governed real data before use.")
    return "".join(parts)
