"""
ChatAudit - controlled perturbation audit for quality-of-service harms.

Method follows Harvey, Kizilcec and Koenecke (2025), "A Framework for Auditing
Chatbots for Dialect-Based Quality-of-Service Harms": take a sample of prompts,
perturb each one in controlled ways (register, typos, optionally dialect), send
every version to the chatbot, and compare answer quality for the SAME question
across versions with paired tests and Benjamini-Hochberg correction.

The Bitext flags in offline_eval.py only show that, for example, typo queries
do worse on average - but those are different questions. Here each question is
compared with itself, so any drop is caused by the wording change alone.

It also measures answer stability (Uberti-Bona Marin et al., 2026): whether the
same question gives the same answer when asked again, and how consistently
paraphrases of one intent land on the same document.

Usage (venv active, in the project folder):
    python perturbation_audit.py bitext.csv
    python perturbation_audit.py bitext.csv --per-intent 200 --seeds 5 --models tfidf,bm25
    python perturbation_audit.py bitext.csv --models tfidf,bm25,embeddings   # needs sentence-transformers
    python perturbation_audit.py bitext.csv --dialects                        # needs Multi-VALUE, see below

Dialects: Harvey et al. recommend a validated transformation tool rather than
hand-written rules. Install Multi-VALUE with  pip install value-nlp  (plus its
spaCy/stanza models). If it is not installed the dialect conditions are skipped.
Alternatively run  --export-base  to write base_queries.csv, transform it with any
validated tool or with dialect speakers, and pass it back with  --dialect-csv FILE
(columns: qid, condition, query).

The study chatbot is not changed. Writes results/audit/:
    summary.txt            the report
    conditions.csv         Hit@1 / MRR / refusal / answer-change per model x condition
    tests.csv              paired tests vs the original wording, BH-adjusted
    by_intent.csv          Hit@1 drop per intent (TF-IDF)
    consistency.csv        repeat-determinism and paraphrase consistency per intent
    examples.csv           example perturbations for the appendix
    per_query.csv          every query x condition x seed x model
    fig_audit.png          Hit@1 by condition and model
"""

import argparse
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from offline_eval import RELEVANT, clean
from retrieval import Retriever

BASE = Path(__file__).parent
OUT = BASE / "results" / "audit"
FALLBACK_MIN_SCORE = 0.1  # the deployed chatbot's "Sorry, no answer" threshold

# ------------------------------------------------------------------ perturbations
# Typo types from Harvey et al. (2025, section 4): duplicate a character, omit a
# character, transpose two adjacent characters, swap with a QWERTY neighbour.
QWERTY = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBOURS = {}
for r, row in enumerate(QWERTY):
    for c, ch in enumerate(row):
        near = set()
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < len(QWERTY):
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if (dr or dc) and 0 <= cc < len(QWERTY[rr]):
                        near.add(QWERTY[rr][cc])
        NEIGHBOURS[ch] = sorted(near)


def _typo_word(w, kind, rng):
    i = rng.randrange(len(w))
    if kind == "duplicate":
        return w[:i] + w[i] + w[i:]
    if kind == "omit":
        return w[:i] + w[i + 1:]
    if kind == "transpose":
        i = rng.randrange(len(w) - 1)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if kind == "keyboard":
        letters = [k for k, ch in enumerate(w) if ch in NEIGHBOURS]
        if not letters:
            return w
        i = rng.choice(letters)
        return w[:i] + rng.choice(NEIGHBOURS[w[i]]) + w[i + 1:]
    raise ValueError(kind)


def add_typos(text, kind, rng, rate):
    """Give each word of 3+ letters a typo with probability `rate` (at least one per query)."""
    words = text.split()
    eligible = [k for k, w in enumerate(words) if len(w) >= 3 and w.isalpha()]
    if not eligible:
        return text
    chosen = [k for k in eligible if rng.random() < rate] or [rng.choice(eligible)]
    for k in chosen:
        kk = rng.choice(["duplicate", "omit", "transpose", "keyboard"]) if kind == "mixed" else kind
        words[k] = _typo_word(words[k], kk, rng)
    return " ".join(words)


def casual(text):
    """Lowercase, no punctuation: the base Harvey et al. add typos to."""
    return re.sub(r"\s+", " ", text.lower().translate(str.maketrans("", "", string.punctuation))).strip()


# Common text-speak / chat abbreviations (register change, not a dialect).
TEXTSPEAK = {"you": "u", "your": "ur", "you're": "ur", "are": "r", "please": "pls", "thanks": "thx",
             "thank you": "ty", "because": "cuz", "want to": "wanna", "going to": "gonna",
             "how": "hw", "what": "wat", "with": "w/", "for": "4", "to": "2", "information": "info",
             "something": "smth", "possible": "poss", "message": "msg", "about": "abt"}


def textspeak(text):
    t = casual(text)
    for k in sorted(TEXTSPEAK, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(k)}\b", TEXTSPEAK[k], t)
    return t


DETERMINISTIC = {
    "original": lambda q: q,
    "lowercase": lambda q: q.lower(),
    "no_punctuation": lambda q: q.translate(str.maketrans("", "", string.punctuation)),
    "casual": casual,
    "textspeak": textspeak,
}
TYPO_KINDS = ["duplicate", "omit", "transpose", "keyboard", "mixed"]


def dialect_variants(base):
    """Multi-VALUE transformations, if the package is installed. Returns rows or []."""
    try:
        from multivalue import Dialects  # pip install value-nlp
    except Exception as e:  # noqa: BLE001 - any import problem means "skip"
        print(f"  Dialect conditions skipped (Multi-VALUE not available: {type(e).__name__}).")
        return []
    wanted = {  # same five dialects as the Rufus case study
        "dialect_AppE": "AppalachianDialect", "dialect_ChcE": "ChicanoDialect",
        "dialect_AAE": "AfricanAmericanVernacular", "dialect_IndE": "IndianDialect",
        "dialect_SgE": "ColloquialSingaporeDialect",
    }
    rows = []
    for cond, cls_name in wanted.items():
        cls = getattr(Dialects, cls_name, None)
        if cls is None:
            print(f"  {cls_name} not found in this Multi-VALUE version; skipped.")
            continue
        d = cls()
        for r in base.itertuples():
            try:
                rows.append({"qid": r.qid, "condition": cond, "seed": 0, "query": d.transform(r.query)})
            except Exception:  # noqa: BLE001 - keep the original if a sentence fails
                rows.append({"qid": r.qid, "condition": cond, "seed": 0, "query": r.query})
        print(f"  {cond}: transformed {len(base)} queries")
    return rows


def build_conditions(base, seeds, typo_rate, use_dialects, dialect_csv):
    rows = []
    for r in base.itertuples():
        for name, fn in DETERMINISTIC.items():
            rows.append({"qid": r.qid, "condition": name, "seed": 0, "query": fn(r.query)})
        for kind in TYPO_KINDS:
            for s in range(seeds):
                rng = random.Random(f"{r.qid}-{kind}-{s}")
                rows.append({"qid": r.qid, "condition": f"typo_{kind}", "seed": s,
                             "query": add_typos(casual(r.query), kind, rng, typo_rate)})
    if use_dialects:
        rows += dialect_variants(base)
    if dialect_csv:
        ext = pd.read_csv(dialect_csv)
        ext["seed"] = 0
        rows += ext[["qid", "condition", "seed", "query"]].to_dict("records")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ models
class Tfidf:
    """The deployed study chatbot, including its 'Sorry, no answer' threshold."""
    name = "TF-IDF (study bot)"

    def __init__(self):
        self.bot = Retriever(BASE / "knowledge_base.json")

    def rank_many(self, queries):
        out = []
        for q in queries:
            res, _ = self.bot.search(q, top_k=3)
            out.append(([x["id"] for x in res], res[0]["score"], res[0]["score"] >= self.bot.min_score))
        return out


class Bm25:
    name = "BM25"

    def __init__(self):
        from compare_models import BM25Model
        self.m = BM25Model(BASE / "knowledge_base.json", True)

    def rank_many(self, queries):
        return [(ids, s, s > 0) for ids, s in (self.m.rank(q) for q in queries)]


class Embeddings:
    name = "Embeddings (MiniLM)"

    def __init__(self):
        from compare_models import EmbeddingModel
        self.m = EmbeddingModel(BASE / "knowledge_base.json", True)

    def rank_many(self, queries):
        return [(ids, s, True) for ids, s in self.m.rank_many(list(queries))]


MODELS = {"tfidf": Tfidf, "bm25": Bm25, "embeddings": Embeddings}


# ------------------------------------------------------------------ statistics
def mcnemar_exact(a, b):
    """Exact McNemar test for paired binary outcomes (a = original, b = perturbed)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())  # lost / gained
    if n01 + n10 == 0:
        return n01, n10, 1.0
    return n01, n10, binomtest(n01, n01 + n10, 0.5).pvalue


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (as used by Harvey et al.)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def score_rows(df, model_cls):
    model = model_cls()
    t0 = time.perf_counter()
    ranked = model.rank_many(df["query"].tolist())
    ms = (time.perf_counter() - t0) * 1000 / len(df)
    hit1, rr, top1, answered = [], [], [], []
    for (ids, _, ans), intent in zip(ranked, df["intent"]):
        rel = RELEVANT[intent]
        rank = next((i + 1 for i, d in enumerate(ids) if d in rel), None)
        hit1.append(bool(ans and rank == 1))
        rr.append((1 / rank) if (ans and rank) else 0.0)
        top1.append(ids[0] if ans else None)
        answered.append(bool(ans))
    out = df.copy()
    out["model"], out["hit1"], out["rr"], out["top1"], out["answered"] = model.name, hit1, rr, top1, answered
    return out, ms


# ------------------------------------------------------------------ consistency
def consistency(data, repeats=5):
    """Repeat-determinism and paraphrase consistency for the deployed chatbot."""
    bot = Retriever(BASE / "knowledge_base.json")
    rows = []
    for intent, g in data.groupby("intent"):
        same = 0
        tops = []
        for q in g["query"]:
            answers = [(a["doc_id"], a["score"]) for a in (bot.answer(q) for _ in range(repeats))]
            same += len(set(answers)) == 1
            tops.append(answers[0][0] or "NO_ANSWER")
        counts = Counter(tops)
        p = np.array(list(counts.values()), float) / len(tops)
        modal, modal_n = counts.most_common(1)[0]
        rows.append({
            "intent": intent, "n": len(g),
            "repeat_identical_%": round(100 * same / len(g), 1),
            "modal_answer": modal, "modal_share_%": round(100 * modal_n / len(tops), 1),
            "modal_is_correct": modal in RELEVANT[intent],
            "distinct_answers": len(counts),
            "normalised_entropy": round(float(-(p * np.log(p)).sum() / np.log(31)), 3),  # 30 docs + no answer
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--per-intent", type=int, default=200, help="base questions per intent (test split)")
    ap.add_argument("--seeds", type=int, default=5, help="random typo draws per question")
    ap.add_argument("--typo-rate", type=float, default=0.15, help="chance each word gets a typo")
    ap.add_argument("--models", default="tfidf,bm25")
    ap.add_argument("--dialects", action="store_true", help="add Multi-VALUE dialect conditions")
    ap.add_argument("--dialect-csv", help="pre-transformed queries: qid, condition, query")
    ap.add_argument("--export-base", action="store_true", help="write base_queries.csv and stop")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Same 50/50 split as offline_eval.py; only the held-out test half is used.
    data = pd.read_csv(args.csv)
    is_dev = data.groupby("intent", group_keys=False).apply(lambda g: g.sample(frac=0.5, random_state=42)).index
    data = data.drop(index=is_dev)
    data = data[data["intent"].isin(RELEVANT)].copy()
    data["query"] = data["instruction"].map(clean)

    # Base prompts: questions Bitext has NOT already noised (no Z = typos flag),
    # so the perturbation is the only source of noise.
    clean_rows = data[~data["flags"].astype(str).str.contains("Z")]
    base = pd.concat([g.sample(min(args.per_intent, len(g)), random_state=7)
                      for _, g in clean_rows.groupby("intent")], ignore_index=True)
    base["qid"] = [f"Q{i:05d}" for i in range(len(base))]
    if args.export_base:
        base[["qid", "intent", "query"]].to_csv(out / "base_queries.csv", index=False)
        print(f"Wrote {out / 'base_queries.csv'} ({len(base)} queries). Transform it and pass with --dialect-csv.")
        return
    print(f"{len(base)} base questions ({base['intent'].nunique()} intents, test split, no Bitext typos)")

    cond = build_conditions(base, args.seeds, args.typo_rate, args.dialects, args.dialect_csv)
    cond = cond.merge(base[["qid", "intent"]], on="qid")
    print(f"{len(cond):,} prompts across {cond['condition'].nunique()} conditions\n")

    scored = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            res, ms = score_rows(cond, MODELS[key])
        except ImportError:
            print(f"Skipping {key}: run  pip install sentence-transformers  to include it.")
            continue
        print(f"  scored {res['model'].iloc[0]:22s} {ms:.2f} ms/query")
        scored.append(res)
    res = pd.concat(scored, ignore_index=True)
    res.to_csv(out / "per_query.csv", index=False)

    # --- per condition (mean over seeds, SD across seeds) ---
    orig = res[res.condition == "original"][["model", "qid", "hit1", "rr", "top1"]].rename(
        columns={"hit1": "hit1_o", "rr": "rr_o", "top1": "top1_o"})
    res = res.merge(orig, on=["model", "qid"])
    res["changed"] = res["top1"] != res["top1_o"]
    per_seed = res.groupby(["model", "condition", "seed"]).agg(
        hit1=("hit1", "mean"), mrr=("rr", "mean"), refused=("answered", lambda s: 1 - s.mean()),
        changed=("changed", "mean")).reset_index()
    summ = per_seed.groupby(["model", "condition"]).agg(
        hit_at_1=("hit1", "mean"), hit_at_1_sd=("hit1", "std"), mrr=("mrr", "mean"),
        refused=("refused", "mean"), answer_changed=("changed", "mean"), seeds=("seed", "nunique")).reset_index()
    o = summ[summ.condition == "original"].set_index("model")["hit_at_1"]
    summ["drop_pts"] = 100 * (summ["model"].map(o) - summ["hit_at_1"])
    summ["relative_drop_%"] = 100 * (1 - summ["hit_at_1"] / summ["model"].map(o))
    order = list(DETERMINISTIC) + [f"typo_{k}" for k in TYPO_KINDS] + sorted(
        c for c in summ.condition.unique() if c not in DETERMINISTIC and not c.startswith("typo_"))
    summ["condition"] = pd.Categorical(summ["condition"], order, ordered=True)
    summ = summ.sort_values(["model", "condition"])
    for c in ["hit_at_1", "hit_at_1_sd", "refused", "answer_changed"]:
        summ[c] = 100 * summ[c]
    summ.round(3).to_csv(out / "conditions.csv", index=False)

    # --- paired tests vs original: McNemar on Hit@1 (seed 0), Wilcoxon on per-question mean RR over seeds ---
    tests = []
    for (m, c), g in res[res.condition != "original"].groupby(["model", "condition"], observed=True):
        g0 = g[g.seed == 0]
        lost, gained, p_mc = mcnemar_exact(g0["hit1_o"], g0["hit1"])
        per_q = g.groupby("qid").agg(rr=("rr", "mean"), rr_o=("rr_o", "first"))
        diff = per_q["rr"] - per_q["rr_o"]
        p_w = wilcoxon(per_q["rr_o"], per_q["rr"]).pvalue if (diff != 0).any() else 1.0
        tests.append({"model": m, "condition": c, "n": len(g0), "lost_hit1": lost, "gained_hit1": gained,
                      "p_mcnemar": p_mc, "mean_rr_change": diff.mean(), "p_wilcoxon_rr": p_w})
    tests = pd.DataFrame(tests)
    tests["p_mcnemar_bh"] = bh_adjust(tests["p_mcnemar"])
    tests["p_wilcoxon_bh"] = bh_adjust(tests["p_wilcoxon_rr"])
    tests["significant_bh_.05"] = tests["p_mcnemar_bh"] < .05
    tests["condition"] = pd.Categorical(tests["condition"], order, ordered=True)
    tests = tests.sort_values(["model", "condition"])
    tests.round(5).to_csv(out / "tests.csv", index=False)

    # --- per intent for the study bot ---
    bot_name = Tfidf.name
    t = res[res.model == bot_name]
    bi = t.groupby(["intent", "condition"], observed=True)["hit1"].mean().unstack() * 100
    by_intent = pd.DataFrame({"original": bi["original"]})
    for c in bi.columns:
        if c != "original":
            by_intent[f"drop_{c}"] = bi["original"] - bi[c]
    by_intent.round(1).to_csv(out / "by_intent.csv")

    # --- consistency (stability) ---
    cons = consistency(data)
    cons.to_csv(out / "consistency.csv", index=False)

    # --- examples for the appendix ---
    ex_q = base.groupby("intent").head(1)["qid"]
    ex = cond[cond.qid.isin(ex_q) & (cond.seed == 0)].pivot(index="qid", columns="condition", values="query")
    ex.to_csv(out / "examples.csv")

    # --- figure ---
    try:
        figure(summ, out, order)
        fig_note = "Figure: fig_audit.png"
    except ImportError:
        fig_note = "Figure skipped (pip install matplotlib)."

    # --- report ---
    show = summ[["model", "condition", "hit_at_1", "hit_at_1_sd", "drop_pts", "refused", "answer_changed"]]
    tshow = tests[["model", "condition", "lost_hit1", "gained_hit1", "p_mcnemar_bh", "mean_rr_change", "p_wilcoxon_bh"]]
    lines = [
        "ChatAudit perturbation audit (method: Harvey, Kizilcec & Koenecke, 2025)",
        f"{len(base)} base questions, test split, no Bitext typo flag; {args.seeds} typo seeds; "
        f"typo rate {args.typo_rate}; models: {', '.join(res.model.unique())}",
        "",
        "Hit@1 by condition (%, mean over seeds; drop_pts = percentage points lost vs original wording;",
        "refused = 'Sorry, no answer'; answer_changed = top document differs from the original wording's)",
        show.round(1).to_string(index=False),
        "",
        "Paired tests vs original wording (same questions). McNemar exact on Hit@1, Wilcoxon on reciprocal rank;",
        "p-values Benjamini-Hochberg adjusted over all comparisons.",
        tshow.round(4).to_string(index=False),
        "",
        "Study bot: Hit@1 points lost per intent (largest typo effect = typo_mixed)",
        by_intent[["original"] + [c for c in by_intent if c in ("drop_casual", "drop_textspeak", "drop_typo_mixed")]]
        .sort_values("drop_typo_mixed", ascending=False).round(1).to_string(),
        "",
        "Answer stability of the study bot (all in-scope test questions, original wording):",
        f"  identical answer on {5} repeats: {cons['repeat_identical_%'].mean():.1f}% of questions "
        "(retrieval is deterministic, unlike the LLM chatbots in Uberti-Bona Marin et al., 2026)",
        cons[["intent", "n", "modal_answer", "modal_share_%", "modal_is_correct", "distinct_answers",
              "normalised_entropy"]].to_string(index=False),
        "",
        fig_note,
    ]
    report = "\n".join(lines)
    (out / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report + f"\n\nAll files in {out}")


def figure(summ, out, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    colours = ["#1f3a5f", "#9aa5b1", "#b04a3a"]
    models = list(summ["model"].unique())
    conds = [c for c in order if c in set(summ["condition"].astype(str))]
    fig, ax = plt.subplots(figsize=(max(7, .55 * len(conds) * len(models)), 3.8))
    w = .8 / len(models)
    x = np.arange(len(conds))
    for k, m in enumerate(models):
        s = summ[summ.model == m].set_index(summ[summ.model == m]["condition"].astype(str)).reindex(conds)
        ax.bar(x + (k - (len(models) - 1) / 2) * w, s["hit_at_1"], w, yerr=s["hit_at_1_sd"].fillna(0),
               capsize=2, label=m, color=colours[k % len(colours)])
    ax.set_xticks(x, [c.replace("_", "\n", 1) for c in conds], fontsize=8)
    ax.set_ylabel("Hit@1 (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Same questions, different wording: Hit@1 by perturbation (error bars = SD over typo seeds)")
    fig.tight_layout(); fig.savefig(out / "fig_audit.png"); plt.close(fig)


if __name__ == "__main__":
    main()
