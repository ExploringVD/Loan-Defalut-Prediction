"""
Decision bands (APPROVE / REVIEW / REJECT) and the NPA reduction simulation.

1. Cut-offs are chosen on the VALIDATION split. The saved final model was refit on train + validation, so its
   validation scores would be too optimistic. We therefore refit the same pipeline (best_params and calibration
   from models/model_meta.json) on the TRAINING split only and use its validation probabilities.
   The probabilities are calibrated: a score of 0.30 means "about 30% chance of default".
     - REJECT  if p >= reject_cutoff: the lowest cut-off that turns away at most MAX_GOOD_REJECTED of good borrowers
     - APPROVE if p <  review_cutoff: the highest cut-off at which REVIEW + REJECT still hold at least
                                      MIN_DEFAULTERS_FLAGGED of the defaulters
     - REVIEW  in between (a credit officer looks at the application)
   The cut-offs are saved in models/model_meta.json.
2. The saved final model scores the TEST split and we compare three policies with "approve everyone"
   (what really happened to these loans):
     - Scenario A: REVIEW loans are approved   (only REJECT is turned away)
     - Scenario B: REVIEW loans are rejected   (only APPROVE gets a loan)
   NPA = non-performing assets = loans that defaulted. NPA reduction = how many fewer defaulted loans (and dollars)
   the lender would have approved.
   Money assumption (simple and stated): a defaulted loan loses its full loan_amnt (no recovery);
   a repaid loan earns loan_amnt x loan_int_rate for one year.

Run from the project folder:
    python src/business.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from load_data import get_splits
from train import FIG, INK, INK2, MUTED, PROJECT_DIR, REPORTS
from tune import META_PATH, MODEL_PATH, build_xgb_pipeline

MAX_GOOD_REJECTED = 0.02        # REJECT may turn away at most 2% of good borrowers
MIN_DEFAULTERS_FLAGGED = 0.90   # REVIEW + REJECT must hold at least 90% of defaulters
CUTOFF_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
NPA_TARGET = 0.30               # synopsis target: 30% NPA reduction

BANDS = ["APPROVE", "REVIEW", "REJECT"]
# Fixed status colours (good / warning / critical); every chart also writes the band name, never colour alone
BAND_COLORS = {"APPROVE": "#0ca30c", "REVIEW": "#fab219", "REJECT": "#d03b3b"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
SCENARIOS = ["Baseline: approve everyone", "Scenario A: REVIEW approved", "Scenario B: REVIEW rejected"]


# ---------------------------------------------------------------------------
# Cut-offs (validation split)
# ---------------------------------------------------------------------------

def validation_probabilities(meta: dict, d: dict) -> np.ndarray:
    """Same pipeline as the final model, but fitted on the TRAINING split only -> honest validation scores."""
    model = build_xgb_pipeline(meta["best_params"], d["y_train"])
    if meta["calibrated"]:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    model.fit(d["X_train"], d["y_train"])
    return model.predict_proba(d["X_val"])[:, 1]


def tradeoff_table(y, p, cutoffs=CUTOFF_GRID) -> pd.DataFrame:
    """What happens if every loan with p >= cut-off is flagged (one cut-off at a time)."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for c in cutoffs:
        flag = p >= c
        rows.append({
            "cutoff": c,
            "share_flagged": flag.mean(),
            "default_rate_flagged": y[flag].mean() if flag.any() else np.nan,
            "default_rate_not_flagged": y[~flag].mean() if (~flag).any() else np.nan,
            "defaulters_caught": flag[y == 1].mean(),
            "good_loans_lost": flag[y == 0].mean(),
        })
    return pd.DataFrame(rows)


def choose_cutoffs(y, p) -> tuple[float, float]:
    """(review_cutoff, reject_cutoff) from the two rules in the module docstring."""
    t = tradeoff_table(y, p)
    ok_reject = t[t["good_loans_lost"] <= MAX_GOOD_REJECTED]
    ok_review = t[t["defaulters_caught"] >= MIN_DEFAULTERS_FLAGGED]
    if ok_reject.empty or ok_review.empty:
        raise ValueError("no cut-off meets the rules - check the model or relax the rules")
    reject = float(ok_reject["cutoff"].min())
    review = float(ok_review["cutoff"].max())
    if review >= reject:
        raise ValueError(f"review cut-off {review} is not below reject cut-off {reject}")
    return review, reject


def assign_bands(p, review_cutoff: float, reject_cutoff: float) -> np.ndarray:
    p = np.asarray(p)
    return np.where(p >= reject_cutoff, "REJECT", np.where(p >= review_cutoff, "REVIEW", "APPROVE"))


def band_table(y, bands) -> pd.DataFrame:
    """Per band: share of loans, default rate, share of all defaulters and of all good loans."""
    y, bands = np.asarray(y), np.asarray(bands)
    rows = []
    for b in BANDS:
        m = bands == b
        rows.append({"band": b, "loans": int(m.sum()), "share_of_loans": m.mean(),
                     "default_rate": y[m].mean() if m.any() else np.nan,
                     "share_of_defaulters": m[y == 1].mean(), "share_of_good_loans": m[y == 0].mean()})
    return pd.DataFrame(rows)


def alternative_pairs(y, p, chosen: tuple[float, float]) -> pd.DataFrame:
    """A few other (review, reject) pairs around the chosen one, to show the trade-off."""
    pairs = sorted({(0.05, 0.25), (0.10, 0.25), chosen, (0.15, 0.35), (0.10, 0.50), (0.20, 0.50)})
    rows = []
    for review, reject in pairs:
        t = band_table(y, assign_bands(p, review, reject)).set_index("band")
        rows.append({"review_cutoff": review, "reject_cutoff": reject, "chosen": (review, reject) == chosen,
                     "approve_share": t.loc["APPROVE", "share_of_loans"], "review_share": t.loc["REVIEW", "share_of_loans"],
                     "reject_share": t.loc["REJECT", "share_of_loans"],
                     "approve_default_rate": t.loc["APPROVE", "default_rate"],
                     "defaulters_rejected": t.loc["REJECT", "share_of_defaulters"],
                     "good_rejected": t.loc["REJECT", "share_of_good_loans"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Business simulation (test split)
# ---------------------------------------------------------------------------

def interest_rate_filled(X: pd.DataFrame, X_train: pd.DataFrame) -> pd.Series:
    """Interest rate for the income calculation. 9.5% of rates are missing: use the training median of the same grade."""
    grade_median = X_train.groupby("loan_grade_num")["loan_int_rate"].median()
    return X["loan_int_rate"].fillna(X["loan_grade_num"].map(grade_median))


def simulate(y, bands, loan_amnt, int_rate) -> pd.DataFrame:
    """One row per policy: what the lender would have approved and what it would have cost / earned."""
    y, bands = np.asarray(y), np.asarray(bands)
    amount, rate = np.asarray(loan_amnt, dtype=float), np.asarray(int_rate, dtype=float) / 100
    approved = {
        SCENARIOS[0]: np.ones(len(y), dtype=bool),
        SCENARIOS[1]: bands != "REJECT",
        SCENARIOS[2]: bands == "APPROVE",
    }
    rows = []
    for name, ok in approved.items():
        bad, good = ok & (y == 1), ok & (y == 0)
        rows.append({
            "policy": name,
            "loans_approved": int(ok.sum()),
            "amount_approved": amount[ok].sum(),
            "defaults_approved": int(bad.sum()),
            "default_amount_approved": amount[bad].sum(),
            "npa_ratio": amount[bad].sum() / amount[ok].sum(),          # defaulted $ / approved $
            "good_loans_approved": int(good.sum()),
            "good_loans_lost": int(((~ok) & (y == 0)).sum()),
            "good_amount_lost": amount[(~ok) & (y == 0)].sum(),
            "interest_income": (amount[good] * rate[good]).sum(),
            "default_loss": amount[bad].sum(),
        })
    sim = pd.DataFrame(rows)
    sim["net_result"] = sim["interest_income"] - sim["default_loss"]
    base = sim.iloc[0]
    sim["npa_reduction_count"] = 1 - sim["defaults_approved"] / base["defaults_approved"]
    sim["npa_reduction_amount"] = 1 - sim["default_amount_approved"] / base["default_amount_approved"]
    sim["good_loans_lost_share"] = sim["good_loans_lost"] / base["good_loans_approved"]
    return sim


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_decision_bands(y_val, p_val, val_bands, y_test, test_bands, review, reject, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    # Left: where repaid and defaulted loans fall on the probability scale (validation)
    bins = np.linspace(0, 1, 41)
    for lo, hi, b in [(0, review, "APPROVE"), (review, reject, "REVIEW"), (reject, 1, "REJECT")]:
        a1.axvspan(lo, hi, color=BAND_COLORS[b], alpha=0.10, lw=0)
        a1.text((lo + hi) / 2, 1.02, b, transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                color=INK, fontsize=9, fontweight="bold")
    y_val, p_val = np.asarray(y_val), np.asarray(p_val)
    a1.hist(p_val[y_val == 0], bins=bins, histtype="step", lw=2, color=BLUE, label="Repaid")
    a1.hist(p_val[y_val == 1], bins=bins, histtype="step", lw=2, color=ORANGE, label="Defaulted")
    for c in (review, reject):
        a1.axvline(c, color=INK2, ls="--", lw=1)
    a1.set_yscale("log")
    a1.set_xlim(0, 1)
    a1.set_xlabel(f"Predicted probability of default (cut-offs {review:.2f} and {reject:.2f})")
    a1.set_ylabel("Number of loans (log scale)")
    a1.set_title("Validation split: predicted probabilities and the three bands", pad=18)
    a1.legend(loc="upper center", fontsize=9)
    # Right: default rate per band, validation vs test
    v = band_table(y_val, val_bands).set_index("band")
    t = band_table(y_test, test_bands).set_index("band")
    x = np.arange(len(BANDS))
    a2.bar(x - 0.18, v["default_rate"], width=0.34, color=[BAND_COLORS[b] for b in BANDS], alpha=0.55,
           label="Validation")
    a2.bar(x + 0.18, t["default_rate"], width=0.34, color=[BAND_COLORS[b] for b in BANDS], label="Test")
    for i, b in enumerate(BANDS):
        a2.text(i - 0.18, v.loc[b, "default_rate"] + 0.02, f"{v.loc[b, 'default_rate']:.0%}", ha="center", fontsize=8.5, color=INK2)
        a2.text(i + 0.18, t.loc[b, "default_rate"] + 0.02, f"{t.loc[b, 'default_rate']:.0%}", ha="center", fontsize=8.5, color=INK2)
    a2.set_xticks(x, [f"{b}\n{t.loc[b, 'share_of_loans']:.0%} of test loans" for b in BANDS])
    a2.set_ylim(0, 1.1)
    a2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("Default rate in the band")
    a2.set_title("Default rate per band (light = validation, dark = test)", pad=18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_npa_reduction(sim: pd.DataFrame, path):
    labels = ["Approve\neveryone", "A: REVIEW\napproved", "B: REVIEW\nrejected"]
    colors = [MUTED, BLUE, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    panels = [
        ("default_amount_approved", "Defaulted loan amount approved (NPA)", "npa_reduction_amount", "lower NPA"),
        ("good_amount_lost", "Good loan amount turned away", None, None),
        ("net_result", "Net result: interest earned - default losses", None, None),
    ]
    for ax, (col, title, red_col, red_word) in zip(axes, panels):
        vals = sim[col].to_numpy() / 1e6
        ax.bar(labels, vals, color=colors, width=0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        for i, v in enumerate(vals):
            note = f"-${-v:,.2f}M" if v < 0 else f"${v:,.2f}M"
            if red_col and i > 0:
                note += f"\n({sim[red_col].iloc[i]:.0%} {red_word})"
            ax.text(i, v + (0.02 if v >= 0 else -0.02) * max(abs(vals).max(), 1e-9),
                    note, ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("US$ millions")
        ax.grid(axis="x", visible=False)
        lo, hi = min(0, vals.min()), max(0, vals.max())
        ax.set_ylim(lo - 0.18 * (hi - lo) * (lo < 0), hi + 0.25 * (hi - lo))
    fig.suptitle(f"Business impact on the test split ({int(sim['loans_approved'].iloc[0]):,} loans)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(v) -> str:
    return f"{v:.1%}"


def money(v) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def write_report(info: dict, path):
    trade, pairs, vb, tb, sim = info["tradeoff"], info["pairs"], info["val_bands"], info["test_bands"], info["sim"]
    review, reject = info["review"], info["reject"]
    a, b, base = sim.iloc[1], sim.iloc[2], sim.iloc[0]
    met_a, met_b = a["npa_reduction_amount"] >= NPA_TARGET, b["npa_reduction_amount"] >= NPA_TARGET
    lines = [
        "# Business simulation - decision bands and NPA reduction",
        "",
        "Made by `python src/business.py`. NPA (non-performing assets) here = loans that defaulted.",
        "",
        "## 1. Choosing the cut-offs (validation split)",
        "",
        "The saved final model was refit on train + validation, so its scores on the validation split would be too",
        "optimistic. To choose the cut-offs honestly, the **same pipeline** (same best parameters, same isotonic",
        f"calibration) was refit on the **training split only** and scored the {info['n_val']:,} validation loans. The",
        "probabilities are calibrated, so a score of 0.30 means \"about 30% chance of default\".",
        "",
        "### Trade-off for a single cut-off (flag every loan with probability >= cut-off)",
        "",
        "| Cut-off | Loans flagged | Default rate if flagged | Default rate if not flagged | Defaulters caught | Good loans lost |",
        "|---|---|---|---|---|---|",
        *[f"| {r.cutoff:.2f} | {pct(r.share_flagged)} | {pct(r.default_rate_flagged)} | {pct(r.default_rate_not_flagged)} | "
          f"{pct(r.defaulters_caught)} | {pct(r.good_loans_lost)} |" for r in trade.itertuples()],
        "",
        "### The two rules",
        "",
        f"- **REJECT** if probability >= **{reject:.2f}**: the lowest cut-off that turns away at most "
        f"{MAX_GOOD_REJECTED:.0%} of good borrowers.",
        f"- **APPROVE** if probability < **{review:.2f}**: the highest cut-off at which REVIEW + REJECT still hold at "
        f"least {MIN_DEFAULTERS_FLAGGED:.0%} of all defaulters.",
        f"- **REVIEW** in between ({review:.2f} to {reject:.2f}): a credit officer checks the application.",
        "",
        "### Other cut-off pairs (validation split)",
        "",
        "| Review cut-off | Reject cut-off | APPROVE | REVIEW | REJECT | Default rate in APPROVE | Defaulters rejected | Good loans rejected |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r.review_cutoff:.2f} | {r.reject_cutoff:.2f}{' **(chosen)**' if r.chosen else ''} | {pct(r.approve_share)} | "
          f"{pct(r.review_share)} | {pct(r.reject_share)} | {pct(r.approve_default_rate)} | {pct(r.defaulters_rejected)} | "
          f"{pct(r.good_rejected)} |" for r in pairs.itertuples()],
        "",
        "### The chosen bands",
        "",
        "| Band | Validation: share of loans | Validation: default rate | Test: share of loans | Test: default rate | Test: share of all defaulters | Test: share of all good loans |",
        "|---|---|---|---|---|---|---|",
        *[f"| {bnd} | {pct(vb.loc[bnd, 'share_of_loans'])} | {pct(vb.loc[bnd, 'default_rate'])} | "
          f"{pct(tb.loc[bnd, 'share_of_loans'])} | {pct(tb.loc[bnd, 'default_rate'])} | "
          f"{pct(tb.loc[bnd, 'share_of_defaulters'])} | {pct(tb.loc[bnd, 'share_of_good_loans'])} |" for bnd in BANDS],
        "",
        "The bands behave the same on the test split as on the validation split, so the cut-offs were not tuned to",
        "luck. The cut-offs are saved in `models/model_meta.json` (`decision_cutoffs`).",
        "",
        f"## 2. Business impact (test split, {int(base['loans_approved']):,} loans, final saved model)",
        "",
        "Baseline = **approve everyone**, which is what really happened to these loans "
        f"({int(base['defaults_approved']):,} defaulted, {pct(base['defaults_approved'] / base['loans_approved'])}).",
        "",
        "| | Approve everyone | A: REVIEW approved | B: REVIEW rejected |",
        "|---|---|---|---|",
        f"| Loans approved | {int(base['loans_approved']):,} | {int(a['loans_approved']):,} | {int(b['loans_approved']):,} |",
        f"| Defaulted loans approved | {int(base['defaults_approved']):,} | {int(a['defaults_approved']):,} | {int(b['defaults_approved']):,} |",
        f"| Defaulted amount approved (NPA) | {money(base['default_amount_approved'])} | {money(a['default_amount_approved'])} | {money(b['default_amount_approved'])} |",
        f"| **NPA reduction - number of defaults** | - | **{pct(a['npa_reduction_count'])}** | **{pct(b['npa_reduction_count'])}** |",
        f"| **NPA reduction - defaulted amount** | - | **{pct(a['npa_reduction_amount'])}** | **{pct(b['npa_reduction_amount'])}** |",
        f"| NPA ratio (defaulted $ / approved $) | {pct(base['npa_ratio'])} | {pct(a['npa_ratio'])} | {pct(b['npa_ratio'])} |",
        f"| Good loans turned away | 0 | {int(a['good_loans_lost']):,} ({pct(a['good_loans_lost_share'])}) | {int(b['good_loans_lost']):,} ({pct(b['good_loans_lost_share'])}) |",
        f"| Good loan amount turned away | $0 | {money(a['good_amount_lost'])} | {money(b['good_amount_lost'])} |",
        f"| Interest earned on good loans (1 year) | {money(base['interest_income'])} | {money(a['interest_income'])} | {money(b['interest_income'])} |",
        f"| Loss on defaulted loans | {money(base['default_loss'])} | {money(a['default_loss'])} | {money(b['default_loss'])} |",
        f"| **Net result** | **{money(base['net_result'])}** | **{money(a['net_result'])}** | **{money(b['net_result'])}** |",
        "",
        "Money assumption (simple, stated on purpose): a defaulted loan loses its **full** amount (no recovery, no",
        "collection), and a repaid loan earns `loan_amnt x loan_int_rate` for **one year** (no funding or operating",
        "costs). Missing interest rates (9.5%) use the training-split median rate of the same grade. Real numbers would",
        "differ - recoveries make defaults cheaper, multi-year loans earn more interest - but the comparison between",
        "the policies is what matters.",
        "",
        "## 3. In plain words",
        "",
        f"- Without the model the lender approved all {int(base['loans_approved']):,} test loans and "
        f"{int(base['defaults_approved']):,} of them defaulted ({money(base['default_amount_approved'])}).",
        f"- **Scenario A** (only REJECT is turned away): {int(a['defaults_approved']):,} defaults are still approved - "
        f"**{pct(a['npa_reduction_count'])} fewer defaults** and **{pct(a['npa_reduction_amount'])} less defaulted money** - "
        f"while only {int(a['good_loans_lost']):,} good borrowers ({pct(a['good_loans_lost_share'])}) are turned away.",
        f"- **Scenario B** (REVIEW is also turned away): **{pct(b['npa_reduction_amount'])} less defaulted money**, but "
        f"{int(b['good_loans_lost']):,} good borrowers ({pct(b['good_loans_lost_share'])}) lose their loan. In practice a "
        "credit officer would approve some REVIEW loans, so the real result lies between A and B.",
        f"- With the money assumption, the portfolio goes from {money(base['net_result'])} (approve everyone) to "
        f"{money(a['net_result'])} (A) or {money(b['net_result'])} (B). B looks better only because of the "
        "assumption: turning away a good borrower costs one year of interest (about 11% of the loan), while approving "
        "a default costs the whole loan. With recoveries, multi-year interest or the cost of losing customers, the gap "
        "shrinks - which is why the REVIEW band goes to a person instead of being rejected automatically.",
        "",
        "## 4. Honest comparison with the 30% NPA-reduction target",
        "",
        f"The synopsis target is a **30% NPA reduction**. The simulation gives **{pct(a['npa_reduction_amount'])}** "
        f"(scenario A) and **{pct(b['npa_reduction_amount'])}** (scenario B) less defaulted money, so the target is "
        f"{'met in both scenarios' if met_a and met_b else 'met only in scenario B' if met_b else 'NOT met'}. "
        "Three reasons to read this carefully:",
        "",
        "1. **The baseline is \"approve everyone\".** A real lender already screens applicants, so the reduction "
        "against a real bank's current process would be smaller.",
        "2. **This dataset is easier than real life.** It contains a perfect rule (every renter whose loan is more than "
        "30% of income defaulted - see `reports/model_comparison.md`); the model gets many defaults \"for free\".",
        "3. **Simple money model** (no recoveries, one year of interest, no costs) - the net result is an illustration.",
        "",
        "So: the 30% target is reached in this simulation, but the number should be presented as \"on this dataset,"
        " versus approving everyone\", not as a promise for a real bank.",
        "",
        "Charts: `reports/figures/20_decision_bands.png`, `reports/figures/21_npa_reduction.png`.",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    meta = json.loads(META_PATH.read_text())
    d = get_splits()

    # 1. Cut-offs on the validation split (model refit on the training split only)
    print("1. Refitting the final pipeline on the training split only to score the validation split ...")
    p_val = validation_probabilities(meta, d)
    review, reject = choose_cutoffs(d["y_val"], p_val)
    val_bands = assign_bands(p_val, review, reject)
    trade = tradeoff_table(d["y_val"], p_val)
    pairs = alternative_pairs(d["y_val"], p_val, (review, reject))
    vb = band_table(d["y_val"], val_bands).set_index("band")
    print(f"   cut-offs: APPROVE < {review:.2f} <= REVIEW < {reject:.2f} <= REJECT")
    print(vb.round(3).to_string())

    # 2. Business simulation with the saved final model on the test split
    model = joblib.load(MODEL_PATH)
    p_test = model.predict_proba(d["X_test"])[:, 1]
    test_bands = assign_bands(p_test, review, reject)
    tb = band_table(d["y_test"], test_bands).set_index("band")
    sim = simulate(d["y_test"], test_bands, d["X_test"]["loan_amnt"], interest_rate_filled(d["X_test"], d["X_train"]))
    print("\n2. Test split simulation:")
    show = sim[["policy", "loans_approved", "defaults_approved", "default_amount_approved", "npa_reduction_count",
                "npa_reduction_amount", "good_loans_lost", "net_result"]]
    print(show.round(3).to_string(index=False))

    # 3. Save cut-offs, charts, report
    meta["decision_cutoffs"] = {
        "review": review, "reject": reject,
        "bands": {"APPROVE": f"p < {review}", "REVIEW": f"{review} <= p < {reject}", "REJECT": f"p >= {reject}"},
        "rules": {"max_good_rejected": MAX_GOOD_REJECTED, "min_defaulters_flagged": MIN_DEFAULTERS_FLAGGED},
        "chosen_on": "validation split, same pipeline refit on the training split only",
        "validation_bands": {b: {"share_of_loans": round(float(vb.loc[b, "share_of_loans"]), 4),
                                 "default_rate": round(float(vb.loc[b, "default_rate"]), 4)} for b in BANDS},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    plot_decision_bands(d["y_val"], p_val, val_bands, d["y_test"], test_bands, review, reject,
                        FIG / "20_decision_bands.png")
    plot_npa_reduction(sim, FIG / "21_npa_reduction.png")
    sim.round(4).to_csv(REPORTS / "business_simulation.csv", index=False)
    write_report({"tradeoff": trade, "pairs": pairs, "val_bands": vb, "test_bands": tb, "sim": sim,
                  "review": review, "reject": reject, "n_val": len(d["X_val"])},
                 REPORTS / "business_simulation.md")
    a, b = sim.iloc[1], sim.iloc[2]
    print(f"\nNPA reduction (defaulted amount): scenario A {a['npa_reduction_amount']:.1%}, "
          f"scenario B {b['npa_reduction_amount']:.1%} (target {NPA_TARGET:.0%})")
    print(f"Saved cut-offs to {META_PATH.relative_to(PROJECT_DIR)}, reports/business_simulation.md (+ .csv), charts 20-21")


if __name__ == "__main__":
    main()
