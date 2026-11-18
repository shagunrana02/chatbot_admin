"""
ChatAudit - RQ3: which task fails most, and does query PHRASING lead to lower ratings?

Run after the study (venv active):
    python phrasing_analysis.py
    python phrasing_analysis.py --sessions sessions_synthetic --out results_synthetic/phrasing   # dry run

Part A - failures by task: low-rating rate, rephrasing rate, queries per task, the
         wrong documents most often returned, and task-level failure ranking.
Part B - phrasing features of every logged query, each tested against the relevance
         rating (Mann-Whitney U for yes/no features, Spearman for numeric ones),
         Benjamini-Hochberg corrected:
           n_words            content words after stop-word removal
           question_form      starts with a question word or ends with "?"
           keyword_only       3 words or fewer and not a question
           likely_typo        a word not in the knowledge base that is 1-2 edits from one that is
           oov_share          share of content words the knowledge base never uses (vocabulary mismatch)
           polite_filler      please / thanks / hi / hello
           negation           not / n't / never / no
           is_reformulation   2nd+ query in the task
Part C - reformulations: how participants rephrased after a poor answer (more
         specific, shorter, word swap, spelling fix, new approach) and how often it worked.

Writes to results/analysis/phrasing/:
    phrasing_report.md, task_failures.csv, query_features.csv, feature_tests.csv, reformulations.csv
"""

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import aggregate
from retrieval import Retriever

BASE = Path(__file__).parent
QUESTION_WORDS = ("how", "what", "where", "when", "why", "which", "who", "can", "could", "do", "does",
                  "is", "are", "will", "would", "should", "may", "did")
POLITE = re.compile(r"\b(please|pls|plz|thanks|thank you|thx|hi|hello|hey)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|none|nothing)\b|n't\b", re.I)
TASK_NAMES = {"T1": "Return Policy", "T2": "Shipping", "T3": "Student Discount",
              "T4": "Product Warranty", "T5": "Order Tracking"}


def load_sessions(folder):
    out = []
    for f in sorted(Path(folder).glob("P-*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s["status"] == "complete":
            out.append(s)
    return out


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


class Featuriser:
    def __init__(self):
        bot = Retriever(BASE / "knowledge_base.json")
        self.analyze = bot.vectorizer.build_analyzer()
        self.vocab = {t for t in bot.vectorizer.get_feature_names_out() if " " not in t}

    def content(self, q):
        return [t for t in self.analyze(q) if " " not in t]

    def __call__(self, q):
        words = self.content(q)
        raw = q.lower().strip()
        oov = [w for w in words if w not in self.vocab]
        typo = any(difflib.get_close_matches(w, self.vocab, n=1, cutoff=.8) for w in oov if len(w) >= 4)
        question = raw.endswith("?") or raw.split(" ", 1)[0] in QUESTION_WORDS
        return {
            "n_words": len(words),
            "question_form": question,
            "keyword_only": len(raw.split()) <= 3 and not question,
            "likely_typo": typo,
            "oov_share": len(oov) / len(words) if words else 1.0,
            "polite_filler": bool(POLITE.search(raw)),
            "negation": bool(NEGATION.search(raw)),
        }


def reformulation_type(prev, cur, feat):
    a, b = set(feat.content(prev)), set(feat.content(cur))
    char_sim = difflib.SequenceMatcher(None, prev.lower(), cur.lower()).ratio()
    if char_sim >= .85 and a != b:
        return "spelling/wording fix"
    if a and b and a < b:
        return "more specific (added words)"
    if a and b and b < a:
        return "shorter (removed words)"
    overlap = len(a & b) / max(1, len(a | b))
    if overlap >= .3:
        return "word swap (partial overlap)"
    return "new approach (little overlap)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(BASE / "sessions"))
    ap.add_argument("--out", default=str(BASE / "results" / "analysis" / "phrasing"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.sessions)
    if not sessions:
        raise SystemExit(f"No completed sessions in {args.sessions}.")
    inter, tasks, _ = aggregate.flatten(sessions)
    inter = inter.dropna(subset=["relevance_rating"]).copy()
    inter["relevance_rating"] = inter["relevance_rating"].astype(int)
    inter["success"] = inter["relevance_rating"] >= 3

    # ---------------- Part A: failures by task
    rows = []
    for tid, g in inter.groupby("task_id"):
        t = tasks[tasks.task_id == tid]
        wrong = g[~g.success]["doc_id"].fillna("no answer").value_counts().head(3)
        rows.append({
            "task": tid, "name": TASK_NAMES.get(tid, ""), "queries": len(g),
            "low_rated_%": round(100 * (~g.success).mean(), 1),
            "mean_rating": round(g.relevance_rating.mean(), 2),
            "rephrased_%": round(100 * (t.n_queries > 1).mean(), 1),
            "queries_per_task": round(t.n_queries.mean(), 2),
            "not_completed_%": round(100 * (1 - t.completed.astype(bool).mean()), 1),
            "no_answer_%": round(100 * g.doc_id.isna().mean(), 1),
            "top_wrong_docs": ", ".join(f"{d} ({n})" for d, n in wrong.items()),
        })
    task_fail = pd.DataFrame(rows).sort_values("low_rated_%", ascending=False)
    task_fail.to_csv(out / "task_failures.csv", index=False)

    # ---------------- Part B: phrasing features vs rating
    feat = Featuriser()
    F = pd.DataFrame([feat(q) for q in inter["query"]], index=inter.index)
    F["is_reformulation"] = inter["n"].astype(int) > 1
    q = pd.concat([inter[["participant_id", "task_id", "n", "query", "doc_id", "similarity_score",
                          "relevance_rating", "success"]], F], axis=1)
    q.to_csv(out / "query_features.csv", index=False)

    tests = []
    for c in ["question_form", "keyword_only", "likely_typo", "polite_filler", "negation", "is_reformulation"]:
        yes, no = q[q[c]], q[~q[c]]
        row = {"feature": c, "type": "yes/no", "n_yes": len(yes), "n_no": len(no),
               "rating_yes": yes.relevance_rating.mean(), "rating_no": no.relevance_rating.mean(),
               "success_yes_%": 100 * yes.success.mean() if len(yes) else np.nan,
               "success_no_%": 100 * no.success.mean() if len(no) else np.nan}
        if len(yes) >= 3 and len(no) >= 3:
            u, p = mannwhitneyu(yes.relevance_rating, no.relevance_rating, alternative="two-sided")
            row.update(effect=1 - 2 * u / (len(yes) * len(no)), p=p)  # rank-biserial: >0 = "yes" rated lower
        tests.append(row)
    for c in ["n_words", "oov_share", "similarity_score"]:
        d = q[[c, "relevance_rating"]].dropna()
        row = {"feature": c, "type": "numeric", "n_yes": len(d)}
        if len(d) >= 5 and d[c].nunique() > 1:
            rho, p = spearmanr(d[c], d.relevance_rating)
            row.update(effect=rho, p=p)
        tests.append(row)
    tests = pd.DataFrame(tests)
    m = tests["p"].notna()
    tests.loc[m, "p_bh"] = bh(tests.loc[m, "p"])
    tests["significant"] = tests["p_bh"] < .05
    tests.round(4).to_csv(out / "feature_tests.csv", index=False)

    # ---------------- Part C: reformulations
    ref = []
    for (pid, tid), g in inter.sort_values("n").groupby(["participant_id", "task_id"]):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            ref.append({"participant_id": pid, "task_id": tid, "from": g.loc[i - 1, "query"], "to": g.loc[i, "query"],
                        "type": reformulation_type(g.loc[i - 1, "query"], g.loc[i, "query"], feat),
                        "rating_before": g.loc[i - 1, "relevance_rating"], "rating_after": g.loc[i, "relevance_rating"]})
    ref = pd.DataFrame(ref)
    if len(ref):
        ref["improved"] = ref.rating_after > ref.rating_before
        ref.to_csv(out / "reformulations.csv", index=False)
        ref_sum = ref.groupby("type").agg(n=("improved", "size"), improved_pct=("improved", lambda s: 100 * s.mean()),
                                          mean_gain=("rating_after", "mean")).round(1)
        ref_sum["mean_gain"] = (ref.groupby("type").rating_after.mean() - ref.groupby("type").rating_before.mean()).round(2)
    else:
        ref_sum = pd.DataFrame()

    # ---------------- report
    def md(df, index=False):
        try:
            return df.to_markdown(index=index)
        except ImportError:
            return "```\n" + df.to_string(index=index) + "\n```"

    worst = task_fail.iloc[0]
    sig = tests[tests.significant == True]  # noqa: E712
    show = tests.copy()
    for c in ["rating_yes", "rating_no", "success_yes_%", "success_no_%", "effect", "p", "p_bh"]:
        show[c] = show[c].astype(float).round(3)
    lines = [
        "# RQ3: retrieval failures by task and by query phrasing", "",
        f"{len(inter)} rated queries from {inter.participant_id.nunique()} participants.", "",
        "## A. Which task fails most?", "",
        md(task_fail), "",
        f"**{worst['task']} ({worst['name']})** had the highest share of low-rated answers "
        f"({worst['low_rated_%']}%), most often returning {worst['top_wrong_docs'] or 'n/a'}.", "",
        "## B. Does phrasing predict lower ratings?", "",
        "Yes/no features: Mann-Whitney U on the 1-5 rating; effect = rank-biserial correlation "
        "(positive = queries WITH the feature were rated lower). Numeric features: Spearman's rho with the rating. "
        "p-values Benjamini-Hochberg adjusted.", "",
        md(show[["feature", "type", "n_yes", "n_no", "rating_yes", "rating_no", "success_yes_%", "success_no_%",
                 "effect", "p", "p_bh", "significant"]]), "",
        ("Significant after correction: " + ", ".join(sig.feature) + ".") if len(sig) else
        "No feature is significant after correction. With 15-20 participants only large effects can be detected, "
        "so report effect sizes and treat these as indicative.", "",
        "Caveat: queries from the same participant are not independent. For a robustness check, repeat the "
        "comparison on per-participant means, or fit a mixed model with participant as a random effect.", "",
        "## C. How did participants rephrase after a poor answer?", "",
        md(ref_sum, index=True) if len(ref_sum) else "No reformulations recorded.", "",
        "Offline evidence for the same question (controlled, same questions reworded): see results/audit/summary.txt "
        "(typos cost about 10 points of Hit@1; case, punctuation and text-speak cost nothing).",
    ]
    (out / "phrasing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\nFull RQ3 report in {out / 'phrasing_report.md'}")


if __name__ == "__main__":
    main()
