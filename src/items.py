"""Categorical items for association-rule mining on the OULAD day-28 table.

Shared by notebook 06 (mining on the 2013 training years) and notebook 07 (replication on 2014), so that both
use identical definitions. Thresholds that depend on data are fitted on the training years only and stored in a spec.
"""
import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
MP = ["code_module", "code_presentation"]

PROFILE_NAMES = {0: "NoActivity", 1: "Low", 2: "Intermittent", 3: "Sustained"}
EDU_FINE = {0: "NoFormal", 1: "BelowALevel", 2: "ALevel", 3: "HE", 4: "PostGrad"}

# Variables that become items, and the levels kept as items (None = every level).
# Binary variables keep only the "marked" level, because "Disability=No" or "PriorAttempts=0" are just the background.
RULE_VARS = ["Profile", "Education", "Deprivation", "Age", "Disability", "Gender", "PriorAttempts", "Module",
             "Load", "EarlyAssessment", "EarlyScore", "Timing", "Forum"]
MARKED_ONLY = {"Disability": "Yes", "PriorAttempts": "1plus", "Forum": "High"}
SENSITIVE_ITEMS = ["Deprivation=Deprived", "Deprivation=Unknown", "Disability=Yes", "Education=BelowALevel", "Age=35plus", "Gender=F", "Gender=M"]
OUTCOME_ITEMS = ["Outcome=AtRisk", "Outcome=NotAtRisk", "Result=Distinction", "Result=Pass", "Result=Fail", "Result=Withdrawn"]


def fit_item_spec(train):
    """Data-dependent thresholds, fitted on the training years only."""
    s = train["score_pct_mean"].dropna()
    return {"score_tertile_cuts": [float(s.quantile(1 / 3)), float(s.quantile(2 / 3))],
            "forum_rank_cut": 0.667, "early_days_cut": 7, "light_credits": 60, "medium_credits": 120}


def categories(df, clusters, spec):
    """One categorical value per student and variable (NaN when the variable does not apply)."""
    d = df.merge(clusters[KEY + ["cluster"]], on=KEY, how="left", validate="one_to_one")
    assert d["cluster"].notna().all(), "students without a profile"
    c = pd.DataFrame(index=d.index)
    c["Profile"] = d.cluster.map(PROFILE_NAMES)
    c["EducationFine"] = d.edu_level.map(EDU_FINE)
    c["Education"] = np.where(d.edu_level <= 1, "BelowALevel", np.where(d.edu_level == 2, "ALevel", "HEplus"))
    c["IMDBand"] = d.imd_order.map(lambda v: f"B{int(v) + 1:02d}" if pd.notna(v) else np.nan)
    c["Deprivation"] = np.where(d.imd_missing == 1, "Unknown", np.where(d.imd_order <= 2, "Deprived", np.where(d.imd_order <= 6, "Middle", "Affluent")))
    c["AgeFine"] = d.age_order.map({0: "0-35", 1: "35-55", 2: "55plus"})
    c["Age"] = np.where(d.age_order == 0, "Under35", "35plus")
    c["Disability"] = np.where(d.disab == 1, "Yes", "No")
    c["Gender"] = np.where(d.male == 1, "M", "F")
    c["PriorAttempts"] = np.where(d.num_prev_capped >= 1, "1plus", "0")
    c["Module"] = d.code_module.values
    c["Region"] = d.region.values
    c["Load"] = np.where(d.credits_capped <= spec["light_credits"], "Light", np.where(d.credits_capped <= spec["medium_credits"], "Medium", "Heavy"))
    c["EarlyAssessment"] = np.where(d.n_assess_due == 0, "NoneDue", np.where(d.submit_rate_early >= 1, "SubmittedAll", "MissedSome"))
    q1, q2 = spec["score_tertile_cuts"]
    sc = d.score_pct_mean
    c["EarlyScore"] = pd.Series(np.where(sc <= q1, "Low", np.where(sc <= q2, "Mid", "High")), index=d.index).where(sc.notna())
    t = d.days_before_due_mean
    c["Timing"] = pd.Series(np.where(t >= spec["early_days_cut"], "Early", "Deadline"), index=d.index).where(t.notna())
    active = d.no_vle_activity == 0
    rank = d[active].groupby(MP).share_forumng.rank(pct=True)
    forum = pd.Series(np.nan, index=d.index, dtype=object)
    forum[active] = np.where(rank >= spec["forum_rank_cut"], "High", "Normal")
    c["Forum"] = forum
    c.insert(0, "id_student", d.id_student.values)
    return c


def to_items(cats):
    """Boolean basket matrix with one column per 'Variable=Level' item."""
    cols = {}
    for v in RULE_VARS:
        levels = [MARKED_ONLY[v]] if v in MARKED_ONLY else sorted(cats[v].dropna().unique())
        for lv in levels:
            cols[f"{v}={lv}"] = (cats[v] == lv).values
    return pd.DataFrame(cols, index=cats.index)


def outcome_items(df):
    out = pd.DataFrame(index=df.index)
    out["Outcome=AtRisk"] = df.target_at_risk.values == 1
    out["Outcome=NotAtRisk"] = df.target_at_risk.values == 0
    for r in ("Distinction", "Pass", "Fail", "Withdrawn"):
        out[f"Result={r}"] = (df.target_final_result.values == r)
    return out
