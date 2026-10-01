"""Measure API latency: send real test-split applicants to a RUNNING API and report median / p95 times.

    uvicorn app.main:app            (in another terminal)
    python scripts/benchmark_api.py --n 100

Logs in as the officer from .env. The applications are saved like any other (name "Benchmark <n>").
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
from clean_data import APPLICANT_FIELDS  # noqa: E402
from load_data import get_splits, load_clean  # noqa: E402


def applicants(n: int) -> list[dict]:
    """n real test-split loans in the raw API format (grade letter, Y/N, missing -> None)."""
    rows = load_clean().loc[get_splits()["X_test"].sample(n, random_state=42).index]
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        a = {f: row[f] for f in APPLICANT_FIELDS}
        a["cb_person_default_on_file"] = "Y" if row["cb_person_default_on_file"] == 1 else "N"
        a = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in a.items()}
        out.append({"applicant_name": f"Benchmark {i + 1}", **{k: (v.item() if hasattr(v, "item") else v) for k, v in a.items()}})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    env = dotenv_values(PROJECT_DIR / ".env")
    client = httpx.Client(base_url=args.url, timeout=30)
    token = client.post("/auth/login", data={"username": env["OFFICER_USERNAME"],
                                             "password": env["OFFICER_PASSWORD"]}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    model_ms, server_ms, client_ms, decisions = [], [], [], []
    for body in applicants(args.n):
        start = time.perf_counter()
        r = client.post("/applications", json=body, headers=headers)
        client_ms.append((time.perf_counter() - start) * 1000)
        r.raise_for_status()
        model_ms.append(r.json()["prediction"]["latency_ms"])
        server_ms.append(float(r.headers["x-process-time-ms"]))
        decisions.append(r.json()["prediction"]["decision"])

    print(f"{args.n} POST /applications requests (real test-split applicants)")
    for name, values in [("model: score + SHAP explanation", model_ms), ("server: whole request incl. database", server_ms),
                         ("client: round trip", client_ms)]:
        print(f"  {name:38s} median {statistics.median(values):6.1f} ms | p95 {np.percentile(values, 95):6.1f} ms | "
              f"max {max(values):6.1f} ms")
    print("  decisions:", {d: decisions.count(d) for d in ["APPROVE", "REVIEW", "REJECT"]})


if __name__ == "__main__":
    main()
