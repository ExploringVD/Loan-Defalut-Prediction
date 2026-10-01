"""Scoring = the SHAP explainer from src/explain.py. It holds the saved calibrated model, so one call gives the
probability, the decision band (cut-offs from model_meta.json) and the top 5 reasons - no second copy of the
preprocessing or of the model in the API."""
import time

from explain import explain_one, get_explainer, reason_text


def load_model() -> float:
    """Load the model + explainer once (cached by get_explainer). Returns the seconds it took."""
    start = time.perf_counter()
    get_explainer()
    return time.perf_counter() - start


def model_meta() -> dict:
    return get_explainer().meta


def score(applicant: dict) -> dict:
    """Score and explain one applicant (dict with the 11 raw fields)."""
    start = time.perf_counter()
    result = explain_one(applicant)
    reasons = [{**r, "text": text} for r, text in zip(result["reasons"], reason_text(result["reasons"]))]
    return {
        "probability": result["probability"],
        "decision": result["decision"],
        "top_reasons": reasons,
        "model_version": result["model_version"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
