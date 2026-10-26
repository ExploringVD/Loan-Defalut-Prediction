"""Batch scoring (src/predict_batch.py) and POST /applications/batch."""
import io
import json
import sys

import pandas as pd
import pytest

from conftest import PROJECT_DIR, needs_model

SAMPLE = PROJECT_DIR / "frontend" / "assets" / "sample_applicants.csv"   # 10 real test-split loans


def sample_df() -> pd.DataFrame:
    from predict_batch import read_applicants_csv
    return read_applicants_csv(SAMPLE)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# The scoring function
# ---------------------------------------------------------------------------

@needs_model
def test_valid_rows_are_scored():
    from predict_batch import score_batch
    result = score_batch(sample_df())
    scored, summary = result["scored"], result["summary"]
    assert len(scored) == 10 and result["rejected"].empty
    assert list(scored.columns) == ["row", "loan_id", "applicant_name", "probability", "decision", "top_reason",
                                    "loan_status"]
    assert scored["probability"].between(0, 1).all()
    assert scored["top_reason"].str.endswith(("increases risk", "decreases risk")).all()
    assert summary["rows_scored"] == 10 and sum(summary["decisions"].values()) == 10
    assert set(scored["decision"]) == {"APPROVE", "REVIEW", "REJECT"}       # the sample has all three bands


@needs_model
def test_batch_gives_the_same_answer_as_single_scoring():
    """Same model, conversion, cut-offs and reasons as POST /applications (explain_one)."""
    from clean_data import APPLICANT_FIELDS
    from explain import explain_one, reason_text
    from predict_batch import score_batch
    df = sample_df()
    scored = score_batch(df)["scored"]
    for record, (_, row) in zip(df.to_dict("records"), scored.iterrows()):
        applicant = {f: (None if pd.isna(record[f]) else record[f]) for f in APPLICANT_FIELDS}
        single = explain_one(applicant)
        assert row["probability"] == pytest.approx(single["probability"], abs=1e-4)
        assert row["decision"] == single["decision"]
        assert row["top_reason"] == reason_text(single["reasons"])[0]


@needs_model
def test_cutoffs_come_from_model_meta(monkeypatch):
    from decision import assign_bands
    from explain import get_explainer
    from predict_batch import score_batch
    meta = json.loads((PROJECT_DIR / "models" / "model_meta.json").read_text())["decision_cutoffs"]
    result = score_batch(sample_df())
    scored = result["scored"]
    assert result["summary"]["cutoffs"] == {"review": meta["review"], "reject": meta["reject"]}
    assert list(scored["decision"]) == list(assign_bands(scored["probability"], meta["review"], meta["reject"]))
    # other cut-offs -> other bands: nothing is hard-coded in the batch code
    monkeypatch.setattr(get_explainer(), "cutoffs", {"review": 0.0, "reject": 1.01})
    assert set(score_batch(sample_df())["scored"]["decision"]) == {"REVIEW"}


@needs_model
def test_invalid_rows_are_rejected_with_a_reason():
    from predict_batch import score_batch
    good = sample_df().iloc[:3]
    bad = pd.concat([good.iloc[[0]]] * 5, ignore_index=True)
    bad.loc[0, "person_age"] = 15                                   # too young
    bad.loc[1, "loan_grade"] = "Z"                                  # unknown grade
    bad.loc[2, ["person_age", "person_emp_length"]] = [25, 30]      # employed longer than possible
    bad.loc[3, "loan_percent_income"] = 0.9                         # does not match amount / income
    bad.loc[4, "person_income"] = None                              # required value missing
    result = score_batch(pd.concat([good, bad], ignore_index=True))
    rejected = result["rejected"]
    assert len(result["scored"]) == 3 and len(rejected) == 5
    assert list(rejected["row"]) == [4, 5, 6, 7, 8]                 # row numbers in the file
    reasons = list(rejected["reason"])
    assert "person_age: Input should be greater than or equal to 18" in reasons[0]
    assert reasons[1].startswith("loan_grade: Input should be")
    assert "person_emp_length cannot be more than person_age - 14" in reasons[2]
    assert "does not match loan_amnt / person_income" in reasons[3]
    assert reasons[4].startswith("person_income:")
    assert result["summary"]["rows_rejected"] == 5 and result["summary"]["rows_total"] == 8


@needs_model
def test_optional_fields_and_columns():
    from predict_batch import score_batch
    df = sample_df().drop(columns=["applicant_name", "loan_status"])
    df.loc[0, ["loan_int_rate", "person_emp_length"]] = None        # optional -> filled in by the model
    result = score_batch(df)
    assert len(result["scored"]) == 10
    assert list(result["scored"].columns) == ["row", "loan_id", "probability", "decision", "top_reason"]
    assert result["summary"]["evaluation"] is None                  # no loan_status -> no ROC-AUC


@needs_model
def test_evaluation_when_loan_status_is_given():
    from predict_batch import score_batch
    ev = score_batch(sample_df())["summary"]["evaluation"]
    assert ev["rows_with_loan_status"] == 10
    assert 0.5 <= ev["roc_auc"] <= 1
    assert sum(ev["rows_by_band"].values()) == 10
    assert all(r is None or 0 <= r <= 1 for r in ev["default_rate_by_band"].values())


def test_file_level_problems_raise_a_clear_error():
    from predict_batch import BatchError, read_applicants_csv
    with pytest.raises(BatchError, match="missing these columns: loan_grade"):
        read_applicants_csv(io.BytesIO(csv_bytes(pd.read_csv(SAMPLE).drop(columns="loan_grade"))))
    with pytest.raises(BatchError, match="no rows"):
        read_applicants_csv(io.BytesIO(SAMPLE.read_text().splitlines()[0].encode()))
    with pytest.raises(BatchError, match="Could not read"):
        read_applicants_csv(io.BytesIO(b""))


@needs_model
def test_command_line_writes_both_files(tmp_path, monkeypatch, capsys):
    import predict_batch
    data = pd.read_csv(SAMPLE)
    data.loc[0, "person_age"] = 15
    source, output = tmp_path / "in.csv", tmp_path / "out.csv"
    data.to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", ["predict_batch.py", "--input", str(source), "--output", str(output)])
    predict_batch.main()
    assert len(pd.read_csv(output)) == 9
    assert len(pd.read_csv(tmp_path / "out_rejected.csv")) == 1
    printed = capsys.readouterr().out
    assert "scored: 9 | rejected: 1" in printed and "ROC-AUC" in printed


# ---------------------------------------------------------------------------
# API: POST /applications/batch
# ---------------------------------------------------------------------------

def upload(content: bytes, name: str = "applicants.csv") -> dict:
    return {"file": (name, content, "text/csv")}


@needs_model
def test_batch_endpoint_needs_login(client):
    r = client.post("/applications/batch", files=upload(SAMPLE.read_bytes()))
    assert r.status_code == 401


@needs_model
def test_batch_endpoint_scores_without_storing(client, officer):
    from sqlalchemy import func, select

    from app.db import SessionLocal
    from app.models import AuditLog, LoanApplication

    def counts():
        with SessionLocal() as db:
            return (db.scalar(select(func.count(LoanApplication.id))),
                    db.scalar(select(func.count(AuditLog.id)).where(AuditLog.action == "score_batch")))

    data = pd.read_csv(SAMPLE)
    data.loc[0, "loan_grade"] = "Z"
    apps_before, audits_before = counts()
    r = client.post("/applications/batch", files=upload(csv_bytes(data)), headers=officer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["scored"]) == 9 and len(body["rejected"]) == 1
    assert body["rejected"][0]["row"] == 1 and body["rejected"][0]["reason"].startswith("loan_grade")
    assert body["scored"][0]["decision"] in {"APPROVE", "REVIEW", "REJECT"}
    assert body["summary"]["rows_scored"] == 9 and body["summary"]["evaluation"]["rows_with_loan_status"] == 9
    apps_after, audits_after = counts()
    assert apps_after == apps_before                 # batch rows are not stored
    assert audits_after == audits_before + 1         # one audit entry per batch


@needs_model
def test_batch_endpoint_rejects_bad_files(client, officer):
    r = client.post("/applications/batch", headers=officer,
                    files=upload(csv_bytes(pd.read_csv(SAMPLE).drop(columns="person_age"))))
    assert r.status_code == 422 and "missing these columns: person_age" in r.json()["detail"]

    too_many = pd.concat([pd.read_csv(SAMPLE)] * 501, ignore_index=True)        # 5,010 rows
    r = client.post("/applications/batch", headers=officer, files=upload(csv_bytes(too_many)))
    assert r.status_code == 413 and "limit is 5,000" in r.json()["detail"]
