"""
ChatAudit - offline "v3" recommendation: a typo-tolerant hybrid retriever.

The perturbation audit showed typos cost the study bot about 10 points of Hit@1,
because TF-IDF matches whole words only ("delivry" shares nothing with "delivery").
This tests the simplest fix: add a second TF-IDF over character n-grams (3-5
characters inside word boundaries), which still overlaps when a word is misspelt,
and combine the two scores:

    score = alpha * word_tfidf + (1 - alpha) * char_tfidf

alpha and the "no answer" threshold are tuned on the DEV half only; results are
reported on the held-out TEST half, both for the original wording and with the
audit's typo perturbations. The study chatbot itself is not changed: this is
evidence for the recommendations chapter (Objective 5) and RQ5.

Usage:
    python typo_tolerant.py bitext.csv
Writes results/offline/typo_tolerant.csv and typo_tolerant.txt
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from offline_eval import RELEVANT, clean
from perturbation_audit import add_typos, casual
from retrieval import Retriever

BASE = Path(__file__).parent
OUT = BASE / "results" / "offline"


class HybridRetriever:
    """Word TF-IDF (the study bot) + character n-gram TF-IDF."""

    def __init__(self, kb_path=BASE / "knowledge_base.json", alpha=0.5, min_score=0.1):
        self.word = Retriever(kb_path)
        corpus = [f"{d['title']}. {d['text']} {d.get('keywords', '')}" for d in self.word.docs]
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self.char_matrix = self.char.fit_transform(corpus)
        self.ids = [d["id"] for d in self.word.docs]
        self.alpha, self.min_score = alpha, min_score

    def scores(self, queries):
        w = cosine_similarity(self.word.vectorizer.transform(queries), self.word.doc_matrix)
        c = cosine_similarity(self.char.transform(queries), self.char_matrix)
        return w, c

    def rank(self, w, c, alpha=None):
        a = self.alpha if alpha is None else alpha
        s = a * w + (1 - a) * c
        order = np.argsort(-s, axis=1)[:, :3]
        return order, s[np.arange(len(s)), order[:, 0]]


def evaluate(model, queries, intents, alpha, threshold):
    w, c = model.scores(queries)
    order, top = model.rank(w, c, alpha)
    hit1, rr, refused = [], [], []
    for row, score, intent in zip(order, top, intents):
        ids = [model.ids[j] for j in row]
        ans = score >= threshold
        rank = next((i + 1 for i, d in enumerate(ids) if d in RELEVANT[intent]), None) if ans else None
        hit1.append(rank == 1)
        rr.append(1 / rank if rank else 0.0)
        refused.append(not ans)
    return np.mean(hit1), np.mean(rr), np.mean(refused)


def oos_refusal(model, queries, alpha, threshold):
    w, c = model.scores(queries)
    _, top = model.rank(w, c, alpha)
    return np.mean(top < threshold)


def typo_version(queries, seed=0):
    return [add_typos(casual(q), "mixed", random.Random(f"{i}-{seed}"), 0.15) for i, q in enumerate(queries)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    data = pd.read_csv(args.csv)
    is_dev = data.groupby("intent", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=42)).index
    data["split"] = "test"
    data.loc[is_dev, "split"] = "dev"
    data["query"] = data["instruction"].map(clean)
    data["in_scope"] = data["intent"].isin(RELEVANT)
    dev, test = data[data.split == "dev"], data[data.split == "test"]
    dev_in, test_in = dev[dev.in_scope], test[test.in_scope]

    model = HybridRetriever()

    # --- tune alpha and threshold on DEV (clean + typo versions), under one fairness
    # constraint: the hybrid must refuse out-of-scope questions at least as often as
    # the study bot does on dev - otherwise it "wins" simply by never saying "no answer".
    dq = dev_in["query"].tolist()
    dq_typo = typo_version(dq, seed=99)
    dev_oos = dev[~dev.in_scope]["query"].tolist()
    base_oos = oos_refusal(model, dev_oos, 1.0, 0.1)
    grid = []
    for a in np.arange(0.0, 1.01, 0.1):
        for t in np.arange(0.05, 0.41, 0.025):
            h_clean = evaluate(model, dq, dev_in["intent"], a, t)[0]
            h_typo = evaluate(model, dq_typo, dev_in["intent"], a, t)[0]
            grid.append({"alpha": round(a, 1), "threshold": round(t, 3), "dev_clean": h_clean, "dev_typo": h_typo,
                         "dev_mean": (h_clean + h_typo) / 2, "dev_oos_refused": oos_refusal(model, dev_oos, a, t)})
    grid = pd.DataFrame(grid)
    ok = grid[grid.dev_oos_refused >= base_oos]
    best = ok.sort_values("dev_mean", ascending=False).iloc[0]
    print(f"Study bot refuses {100 * base_oos:.1f}% of out-of-scope dev questions; the hybrid must match that.")
    a, t = best.alpha, best.threshold
    print(f"Tuned on dev: alpha={a} (1 = word only), threshold={t}")

    # --- TEST: baseline (alpha=1, deployed threshold) vs hybrid
    tq = test_in["query"].tolist()
    rows = []
    for name, alpha, thr in [("Study bot (word TF-IDF)", 1.0, 0.1), ("Hybrid word + char n-grams", a, t)]:
        for cond, qs in [("original", tq)] + [(f"typos (seed {s})", typo_version(tq, s)) for s in range(3)]:
            h1, mrr, ref = evaluate(model, qs, test_in["intent"], alpha, thr)
            rows.append({"model": name, "condition": cond, "hit_at_1": 100 * h1, "mrr": mrr, "refused_%": 100 * ref})
        rows.append({"model": name, "condition": "out-of-scope correctly refused",
                     "refused_%": 100 * oos_refusal(model, test[~test.in_scope]["query"].tolist(), alpha, thr)})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.round(3).to_csv(OUT / "typo_tolerant.csv", index=False)
    grid.round(4).to_csv(OUT / "typo_tolerant_dev_grid.csv", index=False)

    summ = res[res.condition.str.startswith("typos")].groupby("model")["hit_at_1"].agg(["mean", "std"])
    orig = res[res.condition == "original"].set_index("model")["hit_at_1"]
    lines = [
        "Typo-tolerant hybrid retriever (offline only; the study bot is unchanged)",
        f"alpha={a}, threshold={t} chosen on the dev half; results on the held-out test half "
        f"({len(test_in):,} answerable questions).", "",
        res.round(2).to_string(index=False), "",
        "Summary (Hit@1 %):",
        *[f"  {m:30s} original {orig[m]:.1f}   with typos {summ.loc[m, 'mean']:.1f} (SD {summ.loc[m, 'std']:.1f})   "
          f"typo cost {orig[m] - summ.loc[m, 'mean']:.1f} pts" for m in orig.index],
    ]
    report = "\n".join(lines)
    (OUT / "typo_tolerant.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
