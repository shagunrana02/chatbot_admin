"""
ChatAudit - final system check and freeze record.

Run once on the study laptop, inside the project folder with the venv active:
    python freeze_check.py

It checks the chatbot works, then writes SYSTEM_FREEZE.md: the exact versions,
settings and file fingerprints of the system participants will use. Keep that
file; it is the record for your methodology chapter and appendix.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from importlib.metadata import version

from retrieval import DOMAIN_STOP_WORDS, Retriever

BASE = Path(__file__).parent
FILES = ["app.py", "retrieval.py", "knowledge_base.json", "static/index.html", "static/admin.html", "aggregate.py"]

# One natural question per study task, and the documents that answer it
TASK_CHECKS = [
    ("T1 Return Policy", "can I return headphones with bad sound", {"RET01", "RET02", "RET03"}),
    ("T2 Shipping", "how fast can you deliver and how much does it cost", {"SHP01", "SHP03", "SHP04"}),
    ("T3 Student Discount", "I'm at uni, can I get money off", {"DIS01", "DIS02"}),
    ("T4 Product Warranty", "my laptop won't turn on, is it under warranty", {"WAR01", "WAR02", "WAR04"}),
    ("T5 Order Tracking", "ordered 5 days ago and no shipping confirmation", {"ORD01", "ORD02"}),
]


def sha(path):
    return hashlib.sha256((BASE / path).read_bytes()).hexdigest()[:12]


def main():
    problems = []
    bot = Retriever(BASE / "knowledge_base.json")
    docs = bot.docs
    topics = sorted({d["topic"] for d in docs})

    # 1. Knowledge base sanity
    ids = [d["id"] for d in docs]
    if len(ids) != len(set(ids)):
        problems.append("Duplicate document IDs in knowledge_base.json")
    if not 30 <= len(docs) <= 40:
        problems.append(f"Knowledge base has {len(docs)} documents (proposal says 30-40)")

    # 2. Each task finds a correct document
    rows = []
    for name, q, good in TASK_CHECKS:
        r = bot.answer(q)
        ok = r["doc_id"] in good
        if not ok:
            problems.append(f"{name}: '{q}' returned {r['doc_id']}")
        rows.append((name, q, r["doc_id"], r["score"], r["response_time_ms"], ok))

    # 3. Off-topic question gives the fallback
    off = bot.answer("what is the weather in Glasgow")
    if off["doc_id"] is not None:
        problems.append("Off-topic question did not get the 'no answer' fallback")

    # 4. Response time
    times = [bot.answer(q)["response_time_ms"] for _, q, _ in TASK_CHECKS for _ in range(20)]
    mean_ms = sum(times) / len(times)
    if mean_ms >= 500:
        problems.append(f"Mean response time {mean_ms:.1f} ms is over the 500 ms target")

    # 5. Sessions folder should be empty before the study
    n_sessions = len(list((BASE / "sessions").glob("P-*.json"))) if (BASE / "sessions").exists() else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# ChatAudit system freeze record",
        "",
        f"Frozen on: {now}",
        f"Result of final check: {'PASSED' if not problems else 'PROBLEMS FOUND'}",
        "",
        "## Software versions",
        f"- Python {platform.python_version()} on {platform.system()} {platform.release()}",
        f"- Flask {version('flask')}",
        f"- scikit-learn {version('scikit-learn')}",
        f"- pandas {version('pandas')}",
        "",
        "## Retrieval model settings",
        "- Model: TF-IDF vectors (scikit-learn TfidfVectorizer) ranked by cosine similarity; top-1 document returned",
        "- Indexed text per document: title + text + hidden keywords (participants only see the text)",
        "- Lowercasing: yes; n-grams: unigrams and bigrams; sublinear term frequency: yes",
        f"- Stop words: scikit-learn English list (318 words) plus {len(DOMAIN_STOP_WORDS)} domain request words: "
        + ", ".join(sorted(DOMAIN_STOP_WORDS)),
        f"- 'No answer' threshold: cosine similarity below {bot.min_score}",
        "- Stemming: off",
        "",
        "## Knowledge base",
        f"- {len(docs)} documents in {len(topics)} topics: "
        + ", ".join(f"{t} ({sum(d['topic'] == t for d in docs)})" for t in topics),
        f"- Documents with hidden keywords: {sum(1 for d in docs if d.get('keywords'))}",
        "- Version: v2 (synonym keywords added using the Bitext dev split only)",
        "",
        "## Task check (one natural question per task)",
        "",
        "| Task | Question | Returned | Score | Time (ms) | Correct |",
        "| --- | --- | --- | --- | --- | --- |",
        *[f"| {n} | {q} | {d} | {s:.3f} | {t:.2f} | {'yes' if ok else 'NO'} |" for n, q, d, s, t, ok in rows],
        "",
        f"Off-topic question ('what is the weather in Glasgow'): {'fallback given' if off['doc_id'] is None else 'answered ' + off['doc_id']}",
        f"Mean response time over {len(times)} queries: {mean_ms:.2f} ms (target under 500 ms)",
        "",
        "## File fingerprints (SHA-256, first 12 characters)",
        "If any of these change after this date, the system is no longer the frozen version.",
        "",
        *[f"- {f}: {sha(f)}" for f in FILES if (BASE / f).exists()],
        "",
        f"## Data folder\n- Session files present: {n_sessions} "
        + ("(delete test sessions before the pilot)" if n_sessions else "(empty, ready for the study)"),
    ]
    if problems:
        lines += ["", "## Problems", *[f"- {p}" for p in problems]]

    (BASE / "SYSTEM_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    for n, q, d, s, t, ok in rows:
        print(f"  {'OK ' if ok else 'BAD'} {n:22s} -> {d}  ({s:.3f})")
    print(f"  Off-topic -> {'fallback' if off['doc_id'] is None else off['doc_id']}")
    print(f"  Mean response time {mean_ms:.2f} ms")
    print(f"  Session files present: {n_sessions}")
    for p in problems:
        print("  PROBLEM:", p)
    print("\nWrote SYSTEM_FREEZE.md")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
