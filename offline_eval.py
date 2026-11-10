"""
ChatAudit - offline retrieval evaluation with the Bitext customer support dataset.

Usage:
    python offline_eval.py Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
    python offline_eval.py bitext.csv --stemming        # needs: pip install nltk

Writes results/offline/: summary.txt, per_query.csv, by_intent.csv, by_flag.csv, failures.csv
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from retrieval import Retriever

# Bitext intent -> knowledge-base documents that correctly answer it.
# Intents not listed here have no answer in the knowledge base (out of scope).
RELEVANT = {
    "check_refund_policy": {"RET01", "RET03", "RET05", "RET06"},
    "get_refund":          {"RET01", "RET02", "RET03", "RET04"},
    "track_refund":        {"RET04"},
    "delivery_options":    {"SHP01", "SHP02", "SHP03", "SHP04", "SHP05", "SHP06"},
    "delivery_period":     {"SHP01", "SHP03", "SHP05"},
    "track_order":         {"ORD01", "ORD02", "ORD03", "ORD04"},
    "cancel_order":        {"ORD05"},
    "contact_customer_service": {"ORD06"},
    "contact_human_agent":      {"ORD06"},
}
TOPIC = {d: t for t, ds in {
    "returns": {"RET01", "RET02", "RET03", "RET04", "RET05", "RET06"},
    "shipping": {"SHP01", "SHP02", "SHP03", "SHP04", "SHP05", "SHP06"},
    "tracking": {"ORD01", "ORD02", "ORD03", "ORD04", "ORD05", "ORD06"},
}.items() for d in ds}

# Bitext language-variation flags (from the dataset documentation)
FLAGS = {
    "Q": "colloquial", "Z": "typos/noise", "K": "keyword-only", "P": "polite",
    "W": "offensive", "M": "morphological variation", "L": "lexical variation (synonyms)",
    "B": "basic syntax", "I": "interrogative", "C": "coordinated", "E": "abbreviations",
    "N": "negation", "R": "regional",
}


def clean(text):
    """Remove Bitext placeholders such as {{Order Number}}."""
    return re.sub(r"\{\{.*?\}\}", " ", str(text)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--kb", default="knowledge_base.json")
    ap.add_argument("--stemming", action="store_true")
    ap.add_argument("--no-domain-stops", action="store_true",
                    help="keep request words like 'help'/'check' (baseline behaviour)")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="dev = half used for tuning; test = held-out half for reporting")
    ap.add_argument("--out", default="results/offline")
    args = ap.parse_args()

    bot = Retriever(args.kb, use_stemming=args.stemming,
                    domain_stop_words=not args.no_domain_stops)
    data = pd.read_csv(args.csv)
    # Fixed 50/50 split within each intent, so dev and test have the same mix
    is_dev = data.groupby("intent", group_keys=False).apply(
        lambda g: g.sample(frac=0.5, random_state=42)).index
    data["split"] = "test"
    data.loc[is_dev, "split"] = "dev"
    if args.split != "all":
        data = data[data["split"] == args.split].reset_index(drop=True)
    data["query"] = data["instruction"].map(clean)
    data["in_scope"] = data["intent"].isin(RELEVANT)

    rows = []
    t0 = time.perf_counter()
    for r in data.itertuples():
        results, ms = bot.search(r.query, top_k=3)
        ids = [x["id"] for x in results]
        top_score = results[0]["score"]
        answered = top_score >= bot.min_score
        rel = RELEVANT.get(r.intent, set())
        rank = next((i + 1 for i, d in enumerate(ids) if d in rel), None)
        rows.append({
            "intent": r.intent, "category": r.category, "flags": r.flags, "query": r.query,
            "in_scope": r.in_scope, "top1": ids[0], "top1_score": top_score,
            "answered": answered, "response_ms": ms,
            "hit1": bool(rel) and answered and rank == 1,
            "hit3": bool(rel) and answered and rank is not None,
            "rr": (1 / rank) if (rel and answered and rank) else 0.0,
            "topic_ok": bool(rel) and answered and TOPIC.get(ids[0]) == TOPIC.get(next(iter(rel))),
        })
    total_s = time.perf_counter() - t0
    res = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_query.csv", index=False)

    ins, oos = res[res.in_scope], res[~res.in_scope]
    by_intent = ins.groupby("intent").agg(
        n=("hit1", "size"), hit_at_1=("hit1", "mean"), hit_at_3=("hit3", "mean"),
        mrr=("rr", "mean"), topic_acc=("topic_ok", "mean"), no_answer=("answered", lambda s: 1 - s.mean()),
    ).sort_values("hit_at_1")
    by_intent.round(3).to_csv(out / "by_intent.csv")

    flag_rows = []
    for f, name in FLAGS.items():
        has = ins[ins["flags"].astype(str).str.contains(f)]
        if len(has) >= 30:
            flag_rows.append({"flag": f, "variation": name, "n": len(has),
                              "hit_at_1": has.hit1.mean(), "mrr": has.rr.mean()})
    by_flag = pd.DataFrame(flag_rows).sort_values("hit_at_1")
    by_flag.round(3).to_csv(out / "by_flag.csv", index=False)

    fails = ins[~ins.hit1].sample(min(200, (~ins.hit1).sum()), random_state=1)
    fails[["intent", "flags", "query", "top1", "top1_score", "answered"]].to_csv(out / "failures.csv", index=False)

    lines = [
        f"ChatAudit offline evaluation ({'with' if args.stemming else 'without'} stemming)",
        f"Dataset: {args.csv}  split={args.split}  kb={args.kb}  domain_stop_words={not args.no_domain_stops}  ({len(res):,} queries, {total_s:.1f}s)",
        "",
        f"IN-SCOPE queries (answer exists in knowledge base): {len(ins):,}",
        f"  Hit@1 (right document first)      {ins.hit1.mean():6.1%}",
        f"  Hit@3 (right document in top 3)   {ins.hit3.mean():6.1%}",
        f"  MRR                                {ins.rr.mean():6.3f}",
        f"  Topic accuracy (right topic first) {ins.topic_ok.mean():6.1%}",
        f"  'Sorry, no answer' given           {1 - ins.answered.mean():6.1%}",
        "",
        f"OUT-OF-SCOPE queries (no answer exists): {len(oos):,}",
        f"  Correctly said 'no answer'         {1 - oos.answered.mean():6.1%}",
        f"  Gave a wrong answer instead        {oos.answered.mean():6.1%}",
        "",
        f"Mean response time {res.response_ms.mean():.2f} ms (95th percentile {res.response_ms.quantile(.95):.2f} ms)",
        "",
        "Hit@1 by intent:",
        by_intent[["n", "hit_at_1", "hit_at_3", "topic_acc"]].round(3).to_string(),
        "",
        "Hit@1 by language variation (in-scope):",
        by_flag.round(3).to_string(index=False),
    ]
    report = "\n".join(lines)
    (out / "summary.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
