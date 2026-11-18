"""
ChatAudit - the full chatbot app (free chat + project dashboard).

This is the everyday version of the chatbot: no tasks, no ratings questionnaire.
It runs SEPARATELY from the study app, so app.py / retrieval.py / the knowledge
base stay frozen and nothing here is mixed into the study data.

Run (venv active, in the project folder):
    python chat_app.py
Open:  http://127.0.0.1:5051

Pages
    Chat            talk to the bot; see the source document, confidence, other
                    possible answers, and mark each answer helpful / not helpful
    Dashboard       usage of this app + offline evaluation, model comparison,
                    perturbation audit and study progress, all in one place
    Knowledge base  browse and search all 30 documents
    Study           links to the participant page and admin dashboard (app.py, port 5050)

Chats are logged locally to logs/chat_log.jsonl (never to sessions/).
"""

import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

from retrieval import Retriever

BASE = Path(__file__).parent
LOG = BASE / "logs" / "chat_log.jsonl"
LOG.parent.mkdir(exist_ok=True)
RESULTS = BASE / "results"

app = Flask(__name__, static_folder="static")
bot = Retriever(BASE / "knowledge_base.json")
DOCS = {d["id"]: d for d in bot.docs}
lock = threading.Lock()

SUGGESTIONS = [
    # Each of these is checked to return the right document with the frozen bot.
    "How do I start a return?", "How much is next day delivery?", "Is there a student discount?",
    "My laptop won't turn on", "How do I track my order?", "Can I cancel my order?",
    "Do you ship internationally?", "Can I use a discount code?",
]


def confidence(score):
    if score < bot.min_score:
        return "none"
    return "high" if score >= 0.35 else "medium" if score >= 0.2 else "low"


def read_log():
    if not LOG.exists():
        return []
    rows = {}
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "feedback":
                if r["id"] in rows:
                    rows[r["id"]]["helpful"] = r["helpful"]
            else:
                rows[r["id"]] = r
    return list(rows.values())


def append_log(rec):
    with lock, LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def csv_records(path):
    p = RESULTS / path
    return pd.read_csv(p).fillna("").to_dict("records") if p.exists() else []


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "app.html")


@app.get("/api/suggestions")
def suggestions():
    return jsonify(SUGGESTIONS)


@app.post("/api/chat")
def chat():
    text = (request.get_json(force=True).get("message") or "").strip()
    if not text:
        abort(400)
    results, ms = bot.search(text, top_k=3)
    best = results[0]
    conf = confidence(best["score"])
    rid = uuid.uuid4().hex[:10]
    answered = conf != "none"
    reply = {
        "id": rid,
        "answered": answered,
        "response": best["text"] if answered else
        "Sorry, I couldn't find an answer to that. Could you try rephrasing your question?",
        "doc": {"id": best["id"], "title": best["title"], "topic": best["topic"]} if answered else None,
        "score": best["score"],
        "confidence": conf,
        "matched": [t for t, _ in bot.explain(text, best["id"])] if answered else [],
        "alternatives": [{"id": r["id"], "title": r["title"], "topic": r["topic"], "text": r["text"],
                          "score": r["score"]} for r in results[1:] if r["score"] > 0],
        "response_time_ms": ms,
    }
    append_log({"type": "chat", "id": rid, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "query": text, "doc_id": best["id"] if answered else None, "topic": best["topic"] if answered else None,
                "score": best["score"], "confidence": conf, "ms": ms, "helpful": None})
    return jsonify(reply)


@app.post("/api/feedback")
def feedback():
    d = request.get_json(force=True)
    if not isinstance(d.get("helpful"), bool) or not d.get("id"):
        abort(400)
    append_log({"type": "feedback", "id": d["id"], "helpful": d["helpful"]})
    return jsonify({"ok": True})


@app.get("/api/kb")
def kb():
    return jsonify([{k: d[k] for k in ("id", "topic", "title", "text")} for d in bot.docs])


@app.get("/api/dashboard")
def dashboard():
    log = read_log()
    rated = [r for r in log if r.get("helpful") is not None]
    usage = {
        "queries": len(log),
        "answered_pct": round(100 * sum(r["doc_id"] is not None for r in log) / len(log), 1) if log else None,
        "helpful_pct": round(100 * sum(r["helpful"] for r in rated) / len(rated), 1) if rated else None,
        "rated": len(rated),
        "mean_ms": round(sum(r["ms"] for r in log) / len(log), 2) if log else None,
        "by_topic": Counter(r["topic"] or "no answer" for r in log).most_common(),
        "top_docs": [{"id": k, "title": DOCS[k]["title"], "n": n}
                     for k, n in Counter(r["doc_id"] for r in log if r["doc_id"]).most_common(5)],
        "recent": [{k: r[k] for k in ("time", "query", "doc_id", "confidence", "helpful")} for r in log[-10:][::-1]],
        "unhelpful": [{k: r[k] for k in ("query", "doc_id")} for r in log if r.get("helpful") is False][-10:][::-1],
    }
    study = {"completed": 0, "in_progress": 0}
    sdir = BASE / "sessions"
    if sdir.exists():
        for f in sdir.glob("P-*.json"):
            s = json.loads(f.read_text(encoding="utf-8"))
            study["completed" if s.get("status") == "complete" else "in_progress"] += 1
    audit = [r for r in csv_records("audit/conditions.csv") if r["model"].startswith("TF-IDF")]
    return jsonify({
        "kb": {"docs": len(bot.docs), "topics": Counter(d["topic"] for d in bot.docs)},
        "usage": usage,
        "offline": csv_records("offline/test_comparison.csv"),
        "models": csv_records("offline/model_comparison.csv"),
        "audit": audit,
        "study": study,
    })


if __name__ == "__main__":
    print("\n  ChatAudit app:  http://127.0.0.1:5051")
    print("  (study app is separate: python app.py -> http://127.0.0.1:5050)\n")
    app.run(host="127.0.0.1", port=5051, debug=False)
