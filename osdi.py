# osdi.py
from typing import List, Tuple

QUESTIONS = [
    "Do your eyes feel dry or gritty after screen use?",
    "Do your eyes become red while using your laptop?",
    "Do you experience blurred vision during or after laptop work?",
    "Do you get headaches after extended laptop use?",
    "Do your eyes feel tired or heavy after screen time?",
    "Do you feel the need to rub your eyes while using your laptop?",
    "Do you feel burning or stinging sensations in your eyes during laptop use?",
]

OSDI_OPTIONS: List[Tuple[str, int]] = [
    ("Never", 0), ("Rarely", 1), ("Sometimes", 2), ("Often", 3), ("Always", 4)
]

def compute_osdi(vals: List[int]) -> float:
    ans = [v for v in vals if v is not None]
    return round((sum(ans) * 25.0) / len(ans), 1) if ans else 0.0

def osdi_severity(score: float) -> Tuple[str, str]:
    if score <= 12: return "Normal", "ok"
    if score <= 22: return "Mild", "ok"
    if score <= 32: return "Moderate", "warn"
    return "Severe", "danger"
