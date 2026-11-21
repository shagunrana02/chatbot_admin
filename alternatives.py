"""
ChatAudit - offline comparison of alternatives to the study bot's TF-IDF retriever.

The study chatbot is NOT changed. This script answers RQ5 / Objective 5:
"instead of TF-IDF, what else could a customer-support chatbot use, and what would it gain?"

Every model ranks the same 30 documents of the v3 knowledge base. All tuning
(mixing weights, LSA space, classifier training, refusal thresholds) uses the Bitext
DEV half only; every number reported is on the held-out TEST half.

Models
  1. TF-IDF (study bot)        word unigrams+bigrams, the deployed chatbot
  2. BM25                      Okapi BM25 on the same tokens
  3. Char n-gram TF-IDF        3-5 character pieces: tolerant of typos
  4. Word + char hybrid        weighted sum of 1 and 3 (= typo_tolerant.py)
  5. LSA (latent semantic)     TF-IDF reduced to 150 "topics" learned from unlabelled dev questions
  6. Word + LSA hybrid         weighted sum of 1 and 5: exact words + related words
  7. Rank fusion (RRF)         combines the rankings of 1, 2, 3 and 5 with no tuning at all
  8. Supervised classifier     logistic regression trained on labelled dev questions
                               (an upper bound: needs labelled data for every topic)
  9. Sentence embeddings       all-MiniLM-L6-v2, only if sentence-transformers is installed
 10. Embeddings + word hybrid  weighted sum of 1 and 9

Measures (test half)
  hit1 / hit3 / mrr        answerable questions, ranking only (no threshold)
  hit1_typos               same questions with typos added (3 seeds, as in perturbation_audit.py)
  oos_refused              out-of-scope questions refused, with each model's threshold set on dev
                           so that it refuses 5% of answerable dev questions (equal cost for all)
  ms                       retrieval time per query

Usage:
    python alternatives.py bitext.csv              # models 1-8 (and 9-10 if installed)
    pip install sentence-transformers              # optional, about 1 GB, then run again
Writes results/offline/alternatives.csv and alternatives.txt
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

from offline_eval import RELEVANT, clean
from perturbation_audit import add_typos
from retrieval import Retriever

BASE = Path(__file__).parent
OUT = BASE / "results" / "offline"
KB = BASE / "knowledge_base.json"
TYPO_SEEDS, TYPO_RATE = (0, 1, 2), 0.15
DEV_REFUSE = 0.05  # each model refuses 5% of answerable dev questions


# ------------------------------------------------------------------ scorers
# Each scorer maps a list of queries to an (n_queries x n_docs) score matrix.

class Word:
    name = "TF-IDF (study bot)"

    def __init__(self, bot, docs_text, dev_queries):
        self.bot = bot

    def scores(self, q):
        return (self.bot.vectorizer.transform(q) @ self.bot.doc_matrix.T).toarray()


class BM25:
    name = "BM25"

    def __init__(self, bot, docs_text, dev_queries, k1=1.5, b=0.75):
        self.an = bot.vectorizer.build_analyzer()
        tfs = [Counter(self.an(t)) for t in docs_text]
        self.vocab = sorted({t for tf in tfs for t in tf})
        idx = {t: i for i, t in enumerate(self.vocab)}
        n, V = len(tfs), len(self.vocab)
        lens = np.array([sum(tf.values()) for tf in tfs], float)
        df = Counter(t for tf in tfs for t in tf)
        idf = np.array([math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in self.vocab])
        W = np.zeros((V, n))
        for j, tf in enumerate(tfs):
            norm = k1 * (1 - b + b * lens[j] / lens.mean())
            for t, c in tf.items():
                W[idx[t], j] = idf[idx[t]] * c * (k1 + 1) / (c + norm)
        self.W, self.idx = W, idx

    def scores(self, q):
        out = np.zeros((len(q), self.W.shape[1]))
        for i, text in enumerate(q):
            rows = [self.idx[t] for t in self.an(text) if t in self.idx]
            if rows:
                out[i] = self.W[rows].sum(0)
        return out


class Char:
    name = "Char n-gram TF-IDF"

    def __init__(self, bot, docs_text, dev_queries):
        self.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.D = self.v.fit_transform(docs_text)

    def scores(self, q):
        return (self.v.transform(q) @ self.D.T).toarray()


class LSA:
    name = "LSA (latent semantic)"

    def __init__(self, bot, docs_text, dev_queries, dims=150):
        # Same word processing as the bot; the space is learned from KB text plus
        # unlabelled dev questions (no intent labels are used).
        an = bot.vectorizer.build_analyzer()
        self.v = TfidfVectorizer(analyzer=an, sublinear_tf=True, min_df=2)
        X = self.v.fit_transform(list(docs_text) + list(dev_queries))
        self.svd = TruncatedSVD(n_components=dims, random_state=0).fit(X)
        self.D = normalize(self.svd.transform(self.v.transform(docs_text)))

    def scores(self, q):
        return normalize(self.svd.transform(self.v.transform(q))) @ self.D.T


class Embed:
    name = "Sentence embeddings (MiniLM)"

    def __init__(self, bot, docs_text, dev_queries):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.D = self.m.encode(list(docs_text), normalize_embeddings=True)

    def scores(self, q):
        return self.m.encode(list(q), normalize_embeddings=True, batch_size=256) @ self.D.T


class Mix:
    """alpha * A + (1 - alpha) * B, alpha chosen on dev Hit@1."""

    def __init__(self, a, b, name):
        self.a, self.b, self.name, self.alpha = a, b, name, 0.5

    def tune(self, dev_q, dev_rel, ids):
        A, B = self.a.scores(dev_q), self.b.scores(dev_q)
        best = max(np.arange(0.1, 1.0, 0.1), key=lambda al: hit1(al * A + (1 - al) * B, dev_rel, ids))
        self.alpha = round(float(best), 1)
        self.name += f" (alpha={self.alpha})"

    def scores(self, q):
        return self.alpha * self.a.scores(q) + (1 - self.alpha) * self.b.scores(q)


class RRF:
    """Reciprocal rank fusion: score = sum over models of 1 / (60 + rank)."""

    def __init__(self, parts):
        self.parts = parts
        self.name = "Rank fusion: " + " + ".join(p.name.split(" (")[0] for p in parts)

    def scores(self, q):
        total = 0
        for p in self.parts:
            s = p.scores(q)
            ranks = np.argsort(np.argsort(-s, axis=1, kind="stable"), axis=1) + 1
            total = total + 1.0 / (60 + ranks)
        return total


# ------------------------------------------------------------------ measures
def ranks_of(S, rel, ids):
    order = np.argsort(-S, axis=1, kind="stable")
    top = S.max(1)
    out = []
    for row, r, t in zip(order, rel, top):
        # a query that matches nothing (all scores 0) gets no answer, not document 1 by default
        out.append(99 if t <= 0 else next((k + 1 for k, j in enumerate(row) if ids[j] in r), 99))
    return np.array(out)


def hit1(S, rel, ids):
    return float((ranks_of(S, rel, ids) == 1).mean())


def evaluate(model, sets, ids):
    dev_q, dev_rel, test_q, test_rel, typo_sets, oos_q = sets
    t0 = time.perf_counter()
    S = model.scores(test_q)
    ms = (time.perf_counter() - t0) * 1000 / len(test_q)
    r = ranks_of(S, test_rel, ids)
    typo = [hit1(model.scores(tq), test_rel, ids) for tq in typo_sets]
    thr = np.quantile(model.scores(dev_q).max(1), DEV_REFUSE)
    oos = float((model.scores(oos_q).max(1) < thr).mean())
    return {"model": model.name, "hit1": 100 * (r == 1).mean(), "hit3": 100 * (r <= 3).mean(),
            "mrr": float(np.where(r < 99, 1 / r, 0).mean()), "hit1_typos": 100 * np.mean(typo),
            "typo_cost_pts": 100 * ((r == 1).mean() - np.mean(typo)),
            "oos_refused": 100 * oos, "ms": ms}


def supervised(bot, docs, sets, dev_df, test_df, oos_df, typo_sets):
    """Logistic regression on labelled dev questions (all 27 intents, so it also learns 'out of scope').
    Predicted in-scope intent -> best TF-IDF document among that intent's documents."""
    feats_w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    feats_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    Xw, Xc = feats_w.fit_transform(dev_df["query"]), feats_c.fit_transform(dev_df["query"])
    clf = LogisticRegression(max_iter=2000, C=5).fit(hstack([Xw, Xc]).tocsr(), dev_df["intent"])
    ids = [d["id"] for d in docs]
    word = Word(bot, None, None)

    def predict(queries):
        X = hstack([feats_w.transform(queries), feats_c.transform(queries)]).tocsr()
        intents = clf.predict(X)
        S = word.scores(queries)
        out = []
        for i, it in enumerate(intents):
            if it not in RELEVANT:
                out.append(None)  # refuses: predicted out of scope
            else:
                cand = [j for j, d in enumerate(ids) if d in RELEVANT[it]]
                out.append(ids[max(cand, key=lambda j: S[i, j])])
        return out

    t0 = time.perf_counter()
    pred = predict(list(test_df["query"]))
    ms = (time.perf_counter() - t0) * 1000 / len(test_df)
    rel = [RELEVANT[i] for i in test_df["intent"]]
    h1 = np.mean([p in r for p, r in zip(pred, rel)])
    typo = [np.mean([p in r for p, r in zip(predict(tq), rel)]) for tq in typo_sets]
    oos = np.mean([p is None for p in predict(list(oos_df["query"]))])
    return {"model": "Supervised intent classifier (needs labelled data)", "hit1": 100 * h1,
            "hit3": np.nan, "mrr": np.nan, "hit1_typos": 100 * np.mean(typo),
            "typo_cost_pts": 100 * (h1 - np.mean(typo)), "oos_refused": 100 * oos, "ms": ms}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()

    data = pd.read_csv(args.csv)
    is_dev = data.groupby("intent", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=42)).index
    data["split"] = "test"
    data.loc[is_dev, "split"] = "dev"
    data["query"] = data["instruction"].map(clean)
    ins = data["intent"].isin(RELEVANT)
    dev_in, test_in = data[ins & (data.split == "dev")], data[ins & (data.split == "test")]
    test_oos = data[~ins & (data.split == "test")]

    bot = Retriever(KB)
    docs = json.loads(KB.read_text(encoding="utf-8"))
    ids = [d["id"] for d in docs]
    docs_text = [f"{d['title']}. {d['text']} {d.get('keywords', '')}" for d in docs]
    dev_q, test_q = list(dev_in["query"]), list(test_in["query"])
    dev_rel, test_rel = [RELEVANT[i] for i in dev_in["intent"]], [RELEVANT[i] for i in test_in["intent"]]
    typo_sets = [[add_typos(q, "mixed", random.Random(s * 100003 + k), TYPO_RATE) for k, q in enumerate(test_q)]
                 for s in TYPO_SEEDS]
    sets = (dev_q, dev_rel, test_q, test_rel, typo_sets, list(test_oos["query"]))
    all_dev_q = list(data[data.split == "dev"]["query"])  # unlabelled, for LSA only
    print(f"test: {len(test_q):,} answerable, {len(test_oos):,} out-of-scope questions\n")

    word, bm25, char = Word(bot, docs_text, None), BM25(bot, docs_text, None), Char(bot, docs_text, None)
    lsa = LSA(bot, docs_text, all_dev_q)
    models = [word, bm25, char, Mix(word, char, "Word + char hybrid"), lsa, Mix(word, lsa, "Word + LSA hybrid"),
              RRF([word, bm25, char, lsa])]
    try:
        emb = Embed(bot, docs_text, None)
        models += [emb, Mix(word, emb, "Embeddings + word hybrid")]
    except Exception as e:  # not installed or model cannot be downloaded
        print(f"(sentence embeddings skipped: {type(e).__name__}. Run 'pip install sentence-transformers' "
              f"on a computer with internet access to include them.)\n")

    rows = []
    for m in models:
        if isinstance(m, Mix):
            m.tune(dev_q, dev_rel, ids)
        rows.append(evaluate(m, sets, ids))
        print(f"  {rows[-1]['model']:<48} Hit@1 {rows[-1]['hit1']:5.1f}%")
    rows.append(supervised(bot, docs, sets, data[data.split == "dev"], test_in, test_oos, typo_sets))
    print(f"  {rows[-1]['model']:<48} Hit@1 {rows[-1]['hit1']:5.1f}%")

    res = pd.DataFrame(rows).round(2)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "alternatives.csv", index=False)
    text = ("Alternatives to TF-IDF (offline only; the study bot is unchanged)\n"
            f"Knowledge base v3. Tuning on the dev half; results on the test half "
            f"({len(test_q):,} answerable, {len(test_oos):,} out-of-scope questions).\n"
            "hit1/hit3/mrr: ranking only. hit1_typos: mean over 3 typo seeds (rate 0.15).\n"
            "oos_refused: out-of-scope questions refused when each model's threshold refuses 5% of answerable dev questions.\n\n"
            + res.to_string(index=False) + "\n")
    (OUT / "alternatives.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
