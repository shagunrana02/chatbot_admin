"""
ChatAudit - inter-rater agreement for the RQ3 failure coding.

A single person's codes are open to the "you just coded what you expected"
objection. The usual answer (as in Hill, 2024, who reports Cohen's kappa and
Krippendorff's alpha for his annotations) is to have a second person code a
random sample independently and report agreement.

Step 1 - make a blank sample for the second coder (default 25% of coded rows):
    python coding_agreement.py sample
    -> results/analysis/coding_sample_coder2.csv  (the 'code' column is empty)
       Give it to the second coder with the list of codes and their definitions.
Step 2 - when it comes back filled in:
    python coding_agreement.py compare results/analysis/coding_sample_coder2.csv
    -> Cohen's kappa, % agreement, confusion table, rows you disagree on
       (results/analysis/coding_agreement.md)

Reading kappa (Landis & Koch, 1977): <0.20 slight, 0.21-0.40 fair,
0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.
Discuss the disagreements, refine the code definitions, and report both kappa
and how disagreements were resolved.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).parent
DIR = BASE / "results" / "analysis"
KEY = ["participant_id", "task_id", "n"]


def strength(k):
    return ("slight" if k <= .2 else "fair" if k <= .4 else "moderate" if k <= .6
            else "substantial" if k <= .8 else "almost perfect")


def sample(frac=.25, seed=11):
    sheet = pd.read_csv(DIR / "coding_sheet.csv", dtype=str).fillna("")
    coded = sheet[sheet["code"] != ""]
    if coded.empty:
        raise SystemExit("Code the queries first (coding_sheet.csv or the admin Analysis tab), then run again.")
    s = coded.sample(max(1, round(frac * len(coded))), random_state=seed)
    s = s[KEY + ["query", "doc_id", "relevance_rating"]].assign(code="", note="")
    out = DIR / "coding_sample_coder2.csv"
    s.to_csv(out, index=False)
    print(f"{len(s)} of {len(coded)} coded queries written to {out} with the codes removed.")


def compare(path):
    a = pd.read_csv(DIR / "coding_sheet.csv", dtype=str).fillna("")
    b = pd.read_csv(path, dtype=str).fillna("")
    m = a.merge(b[KEY + ["code"]], on=KEY, suffixes=("_1", "_2"))
    m = m[(m["code_1"] != "") & (m["code_2"] != "")]
    if m.empty:
        raise SystemExit("No rows coded by both coders.")
    k = cohen_kappa_score(m["code_1"], m["code_2"])
    agree = (m["code_1"] == m["code_2"]).mean()
    table = pd.crosstab(m["code_1"], m["code_2"], rownames=["coder 1"], colnames=["coder 2"])
    dis = m[m["code_1"] != m["code_2"]][KEY + ["query", "code_1", "code_2"]]
    lines = [
        "# Failure-coding agreement", "",
        f"{len(m)} queries coded by both coders. Agreement {100 * agree:.1f}%; "
        f"Cohen's kappa = {k:.2f} ({strength(k)}, Landis & Koch, 1977).", "",
        "## Confusion table", "", "```", table.to_string(), "```", "",
        f"## Disagreements ({len(dis)}) - discuss and record the agreed code", "",
        "```", dis.to_string(index=False) if len(dis) else "none", "```",
    ]
    (DIR / "coding_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:3]))
    print(f"\nFull table in {DIR / 'coding_agreement.md'}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "sample":
        sample()
    elif len(sys.argv) == 3 and sys.argv[1] == "compare":
        compare(sys.argv[2])
    else:
        raise SystemExit(__doc__)
