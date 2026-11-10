"""
ChatAudit - offline comparison of three retrieval models on the Bitext test half.

Models (all rank the same 30 knowledge-base documents and return the top one):
  tfidf      - the study chatbot (retrieval.py): TF-IDF + cosine similarity
  bm25       - Okapi BM25 (k1=1.5, b=0.75) on the same tokens as the chatbot
  embeddings - sentence embeddings (all-MiniLM-L6-v2) + cosine similarity

Each model is run on both knowledge-base versions:
  v1 = original documents, no hidden keywords
  v2 = documents + hidden synonym keywords (the frozen study version)
so the results show whether meaning-based search removes the need for hand-written keywords.

Usage (inside the project folder, venv active):
    python compare_models.py bitext.csv
    # embeddings need one extra install (about 1 GB with PyTorch), then run again:
    pip install sentence-transformers

Writes results/offline/model_comparison.csv, model_by_intent.csv, model_by_wording.csv
and prints a summary. The study chatbot itself is not changed.
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from offline_eval import FLAGS, RELEVANT, clean
from retrieval import Retriever

BASE = Path(__file__).parent
OUT = BASE / "results" / "offline"


def doc_texts(docs, with_keywords):
    return [f"{d['title']}. {d['text']}" + (f" {d.get('keywords', '')}" if with_keywords else "") for d in docs]


class TfidfModel:
    """The study chatbot exactly as participants use it."""
    name = "TF-IDF"

    def __init__(self, kb_path, with_keywords):
        self.bot = Retriever(kb_path, domain_stop_words=with_keywords)

    def rank(self, query):
        res, _ = self.bot.search(query, top_k=3)
        return [r["id"] for r in res], res[0]["score"]


class BM25Model:
    """Okapi BM25 over the chatbot's own tokens (same stop words and bigrams)."""
    name = "BM25"

    def __init__(self, kb_path, with_keywords, k1=1.5, b=0.75):
        bot = Retriever(kb_path, domain_stop_words=with_keywords)
        self.analyze = bot.vectorizer.build_analyzer()
        self.ids = [d["id"] for d in bot.docs]
        self.tfs = [Counter(self.analyze(t)) for t in doc_texts(bot.docs, with_keywords)]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = self.lens.mean()
        n = len(self.tfs)
        df = Counter(term for tf in self.tfs for term in tf)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query):
        q = [t for t in self.analyze(query) if t in self.idf]
        scores = np.zeros(len(self.ids))
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            scores[i] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        order = np.argsort(-scores, kind="stable")[:3]
        return [self.ids[i] for i in order], float(scores[order[0]])


class EmbeddingModel:
    """Sentence embeddings: matches meaning rather than exact words."""
    name = "Embeddings (MiniLM)"
    _model = None

    def __init__(self, kb_path, with_keywords, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # only needed for this model
        if EmbeddingModel._model is None:
            EmbeddingModel._model = SentenceTransformer(model_name)
        self.m = EmbeddingModel._model
        docs = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        self.ids = [d["id"] for d in docs]
        self.doc_vecs = self.m.encode(doc_texts(docs, with_keywords), normalize_embeddings=True)

    def rank_many(self, queries):
        q = self.m.encode(queries, normalize_embeddings=True, batch_size=256, show_progress_bar=True)
        sims = q @ self.doc_vecs.T
        order = np.argsort(-sims, axis=1)[:, :3]
        return [([self.ids[j] for j in row], float(sims[i, row[0]])) for i, row in enumerate(order)]


def evaluate(model, data):
    t0 = time.perf_counter()
    if hasattr(model, "rank_many"):
        ranked = model.rank_many(data["query"].tolist())
    else:
        ranked = [model.rank(q) for q in data["query"]]
    ms = (time.perf_counter() - t0) * 1000 / len(data)
    hit1, hit3, rr = [], [], []
    for (ids, _), intent in zip(ranked, data["intent"]):
        rel = RELEVANT[intent]
        rank = next((i + 1 for i, d in enumerate(ids) if d in rel), None)
        hit1.append(rank == 1)
        hit3.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
    out = data[["intent", "flags"]].copy()
    out["hit1"], out["hit3"], out["rr"] = hit1, hit3, rr
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = ap.parse_args()

    data = pd.read_csv(args.csv)
    is_dev = data.groupby("intent", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=42)).index
    data["split"] = "test"
    data.loc[is_dev, "split"] = "dev"
    if args.split != "all":
        data = data[data["split"] == args.split]
    data = data[data["intent"].isin(RELEVANT)].reset_index(drop=True)  # questions the KB can answer
    data["query"] = data["instruction"].map(clean)
    print(f"{len(data):,} answerable questions ({args.split} split)\n")

    kbs = {"v1 (no keywords)": (BASE / "knowledge_base_v1.json", False),
           "v2 (with keywords)": (BASE / "knowledge_base.json", True)}
    model_classes = [TfidfModel, BM25Model, EmbeddingModel]

    summary, by_intent, by_wording = [], [], []
    for cls in model_classes:
        for kb_name, (path, kw) in kbs.items():
            try:
                model = cls(path, kw)
            except ImportError:
                print(f"Skipping {cls.name}: run  pip install sentence-transformers  to include it.")
                break
            res, ms = evaluate(model, data)
            label = f"{cls.name} · {kb_name}"
            summary.append({"model": cls.name, "knowledge_base": kb_name, "questions": len(res),
                            "hit_at_1": round(100 * res.hit1.mean(), 1), "hit_at_3": round(100 * res.hit3.mean(), 1),
                            "mrr": round(res.rr.mean(), 3), "ms_per_query": round(ms, 2)})
            for intent, g in res.groupby("intent"):
                by_intent.append({"model": cls.name, "knowledge_base": kb_name, "intent": intent,
                                  "n": len(g), "hit_at_1": round(100 * g.hit1.mean(), 1)})
            for f, name in FLAGS.items():
                g = res[res["flags"].astype(str).str.contains(f)]
                if len(g) >= 30:
                    by_wording.append({"model": cls.name, "knowledge_base": kb_name, "variation": name,
                                       "n": len(g), "hit_at_1": round(100 * g.hit1.mean(), 1)})
            print(f"  {label:42s} Hit@1 {100 * res.hit1.mean():5.1f}%  Hit@3 {100 * res.hit3.mean():5.1f}%  "
                  f"MRR {res.rr.mean():.3f}  {ms:.2f} ms/query")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(by_intent).to_csv(OUT / "model_by_intent.csv", index=False)
    pd.DataFrame(by_wording).to_csv(OUT / "model_by_wording.csv", index=False)
    print(f"\nSaved model_comparison.csv, model_by_intent.csv, model_by_wording.csv in {OUT}")


if __name__ == "__main__":
    main()
