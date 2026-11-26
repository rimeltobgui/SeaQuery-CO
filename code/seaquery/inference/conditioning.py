"""CoralVQA metric serialization and adaptive question conditioning."""

import re
import math
from typing import Any, Mapping, Sequence

METRIC_KEYS = ("live", "bleached", "dead", "algae", "cots")


ADAPTIVE_KEYWORDS = {
    "bleached": ("bleach", "white", "suscept"),
    "algae": ("algae", "algal"),
    "live": ("live coral", "live cover", "coral cover", "coral coverage", "growth"),
    "dead": ("dead", "mort", "mortality", "skeleton", "bare"),
    "cots": ("cots", "starfish", "crown", "acanthaster"),
}


ALL_METRIC_KEYWORDS = ("health", "stress", "condition", "habitat", "environment")


def validate_metrics(metrics):
    result = {}
    for key in METRIC_KEYS:
        if key not in metrics or isinstance(metrics[key], bool):
            raise ValueError(f"Missing or invalid metric: {key}")
        value = float(metrics[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Metric must be finite and nonnegative: {key}")
        if key == "cots":
            if not value.is_integer():
                raise ValueError("COTS must be an integer count")
            result[key] = int(value)
        else:
            if value > 100:
                raise ValueError(f"Coverage must be a percentage from 0 to 100: {key}")
            result[key] = value
    return result


def normalize_question(question: Any) -> str:
    text = re.sub(r"<image>", " ", str(question), flags=re.I)
    text = re.sub(r"\[vqa\]", " ", text, flags=re.I)
    return " ".join(text.split()).strip()


def metric_string(metrics: Mapping[str, Any], keys: Sequence[str]) -> str:
    labels = {
        "live": "Live coral", "bleached": "Bleached coral", "dead": "Dead coral",
        "algae": "Algae", "cots": "COTS",
    }
    parts = []
    for key in METRIC_KEYS:
        if key not in keys:
            continue
        value = metrics[key]
        rendered = f"{int(value)}" if key == "cots" and float(value).is_integer() else f"{float(value):.1f}%"
        parts.append(f"{labels[key]}: {rendered}")
    return "" if not parts else "[Reef metrics - " + ", ".join(parts) + "] "


def build_adaptive_prefix(metrics: Mapping[str, Any], question: str) -> str:
    q = normalize_question(question).lower()
    if any(keyword in q for keyword in ALL_METRIC_KEYWORDS):
        keys = list(METRIC_KEYS)
    else:
        selected = set()
        for key, keywords in ADAPTIVE_KEYWORDS.items():
            if any(keyword in q for keyword in keywords):
                selected.add(key)
        keys = [key for key in METRIC_KEYS if key in selected]
    return metric_string(metrics, keys)


def build_bracket_prefix(metrics: Mapping[str, Any], question: str | None = None) -> str:
    return metric_string(metrics, METRIC_KEYS)

