"""
ChatAudit - participant web app + admin dashboard.

Run:
    pip install flask scikit-learn pandas
    python app.py
Participant page:  http://127.0.0.1:5050
Admin dashboard:   http://127.0.0.1:5050/admin

Admin password: set it before starting with
    export CHATAUDIT_ADMIN_PASSWORD="choose-a-password"
If you don't, a random one is generated and printed in the terminal at startup.

Every session is saved to sessions/<participant_id>.json and re-saved after
each event, so a crash or refresh never loses more than the last click.
Resume a session with:  http://127.0.0.1:5050/?pid=P-XXXXXX
"""

import hmac
import io
import json
import math
import os
import secrets
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_from_directory, session as cookie

import aggregate
from retrieval import Retriever

BASE = Path(__file__).parent
SESSIONS_DIR = BASE / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
OFFLINE_DIR = BASE / "results" / "offline"

# Study options. ask_trust adds "How much would you trust this answer?" after each
# relevance rating. Keep it false unless your ethics approval covers a trust rating.
_cfg_file = BASE / "study_config.json"
STUDY_CONFIG = {"ask_trust": False, **(json.loads(_cfg_file.read_text()) if _cfg_file.exists() else {})}
CODING_FILE = BASE / "results" / "coding.json"
FAILURE_CODES = ["Vocabulary mismatch", "Missing topic", "Wrong document chosen", "Typo or spelling",
                 "Off-topic question", "Answer incomplete", "Other"]

ADMIN_PASSWORD = os.environ.get("CHATAUDIT_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
PASSWORD_WAS_GENERATED = "CHATAUDIT_ADMIN_PASSWORD" not in os.environ

TASKS = [
    {"id": "T1", "topic": "Return Policy",
     "text": "You purchased a pair of headphones two weeks ago. The audio quality is poor, "
             "and you would like to return them. Find out whether a return is possible and what steps to take."},
    {"id": "T2", "topic": "Shipping",
     "text": "You have an urgent gift that needs to be delivered. "
             "Find out the available delivery speeds and their costs."},
    {"id": "T3", "topic": "Student Discount",
     "text": "You are a university student and have heard the store offers a student discount. "
             "Find out whether this is available and how to access it."},
    {"id": "T4", "topic": "Product Warranty",
     "text": "You purchased a laptop several months ago, and it is no longer powering up. "
             "Find out whether it is covered by warranty and the procedure for making a claim."},
    {"id": "T5", "topic": "Order Tracking",
     "text": "You placed an order five days ago and have not received a shipping confirmation. "
             "Find out what to do next."},
]
TASK_IDS = [t["id"] for t in TASKS]

app = Flask(__name__, static_folder="static")
app.secret_key = secrets.token_bytes(32)  # new key each start: admins log in again after a restart
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
bot = Retriever(BASE / "knowledge_base.json")
lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def path_for(pid):
    # Only allow our own ID format, so the pid can't be used to reach other files
    if not (pid.startswith("P-") and len(pid) == 8 and pid[2:].isalnum()):
        abort(400, "bad participant id")
    return SESSIONS_DIR / f"{pid}.json"


def load(pid):
    p = path_for(pid)
    if not p.exists():
        abort(404, "session not found")
    return json.loads(p.read_text(encoding="utf-8"))


def save(session):
    p = path_for(session["participant_id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic write: never leaves a half-written file


def sus_score(answers):
    """Brooke (1996): odd items score-1, even items 5-score, sum x 2.5 -> 0-100."""
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/tasks")
def tasks():
    return jsonify(TASKS)


@app.get("/api/config")
def config():
    """What the participant page should ask. Only ask_trust is configurable."""
    return jsonify({"ask_trust": STUDY_CONFIG["ask_trust"]})


@app.post("/api/start")
def start():
    with lock:
        while True:
            pid = "P-" + secrets.token_hex(3).upper()
            if not path_for(pid).exists():
                break
        session = {
            "participant_id": pid,
            "started_at": now(),
            "finished_at": None,
            "status": "in_progress",
            "tasks": {t: {"started_at": None, "completed": None,
                          "completed_at": None, "interactions": []} for t in TASK_IDS},
            "sus": None,
            "feedback": None,
        }
        save(session)
    return jsonify({"participant_id": pid})


@app.get("/api/session/<pid>")
def get_session(pid):
    return jsonify(load(pid))


@app.post("/api/task_start")
def task_start():
    """Records when a task was first shown, for task-time measurement."""
    data = request.get_json(force=True)
    if data.get("task_id") not in TASK_IDS:
        abort(400)
    with lock:
        s = load(data["participant_id"])
        t = s["tasks"][data["task_id"]]
        if not t.get("shown_at"):
            t["shown_at"] = now()
            save(s)
    return jsonify({"ok": True})


@app.post("/api/query")
def query():
    data = request.get_json(force=True)
    pid, task, text = data["participant_id"], data["task_id"], data["query"].strip()
    if task not in TASK_IDS or not text:
        abort(400)
    result = bot.answer(text)
    with lock:
        s = load(pid)
        t = s["tasks"][task]
        t["started_at"] = t["started_at"] or now()
        t["interactions"].append({
            "n": len(t["interactions"]) + 1,
            "timestamp": now(),
            "query": text,
            "response": result["response"],
            "doc_id": result["doc_id"],
            "similarity_score": result["score"],
            "server_response_time_ms": result["response_time_ms"],
            "client_response_time_ms": None,  # filled in by /api/rate
            "relevance_rating": None,
        })
        save(s)
        n = len(t["interactions"])
    return jsonify({"n": n, "response": result["response"]})


@app.post("/api/rate")
def rate():
    data = request.get_json(force=True)
    with lock:
        s = load(data["participant_id"])
        inter = s["tasks"][data["task_id"]]["interactions"][int(data["n"]) - 1]
        if "rating" in data:
            rating = int(data["rating"])
            if not 1 <= rating <= 5:
                abort(400)
            inter["relevance_rating"] = rating
            inter["client_response_time_ms"] = data.get("client_response_time_ms")
        if "trust" in data:
            if not STUDY_CONFIG["ask_trust"]:
                abort(400, "trust rating is switched off")
            trust = int(data["trust"])
            if not 1 <= trust <= 5:
                abort(400)
            inter["trust_rating"] = trust
        save(s)
    return jsonify({"ok": True})


@app.post("/api/complete")
def complete():
    data = request.get_json(force=True)
    with lock:
        s = load(data["participant_id"])
        t = s["tasks"][data["task_id"]]
        t["completed"] = bool(data["completed"])
        t["completed_at"] = now()
        save(s)
    return jsonify({"ok": True})


@app.post("/api/finish")
def finish():
    data = request.get_json(force=True)
    answers = [int(a) for a in data["sus"]]
    if len(answers) != 10 or not all(1 <= a <= 5 for a in answers):
        abort(400, "SUS needs 10 answers from 1 to 5")
    with lock:
        s = load(data["participant_id"])
        s["sus"] = {"answers": answers, "score": sus_score(answers)}
        s["feedback"] = {
            "helpful": data.get("helpful", "").strip(),
            "disappointed": data.get("disappointed", "").strip(),
            "would_change": data.get("would_change", "").strip(),
        }
        s["finished_at"] = now()
        s["status"] = "complete"
        save(s)
    return jsonify({"ok": True})


# ============================================================== admin side

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not cookie.get("admin"):
            return jsonify({"error": "login required"}), 401
        return view(*args, **kwargs)
    return wrapped


def clean(obj):
    """Make pandas/numpy values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def frames(include_incomplete=False):
    sessions = aggregate.load_sessions(include_incomplete=include_incomplete)
    return sessions, *aggregate.flatten(sessions)


@app.get("/admin")
def admin_page():
    return send_from_directory(app.static_folder, "admin.html")


@app.post("/api/admin/login")
def admin_login():
    pw = (request.get_json(force=True) or {}).get("password", "")
    if hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode()):
        cookie["admin"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Wrong password"}), 401


@app.post("/api/admin/logout")
def admin_logout():
    cookie.clear()
    return jsonify({"ok": True})


@app.get("/api/admin/summary")
@admin_required
def admin_summary():
    all_sessions = aggregate.load_sessions(include_incomplete=True)
    complete, inter, tasks_df, parts = frames()

    overall, per_task = None, []
    if complete:
        overall = aggregate.metrics(inter, tasks_df)
        overall["SUS"] = parts["sus_score"].mean()
        overall["SUS_sd"] = parts["sus_score"].std()
        for tid in TASK_IDS:
            m = aggregate.metrics(inter[inter.task_id == tid], tasks_df[tasks_df.task_id == tid])
            per_task.append({"task_id": tid, "topic": next(t["topic"] for t in TASKS if t["id"] == tid), **m})

    rows = []
    for s in all_sessions:
        ts = s["tasks"].values()
        rows.append({
            "participant_id": s["participant_id"], "status": s["status"],
            "started_at": s["started_at"], "finished_at": s["finished_at"],
            "tasks_done": sum(t["completed"] is not None for t in ts),
            "tasks_completed": sum(bool(t["completed"]) for t in ts),
            "queries": sum(len(t["interactions"]) for t in ts),
            "sus": (s["sus"] or {}).get("score"),
        })
    rows.sort(key=lambda r: r["started_at"], reverse=True)

    feedback = [{"participant_id": s["participant_id"], **s["feedback"]}
                for s in all_sessions if s.get("feedback")]
    low = []
    if len(inter):
        lr = inter[inter.relevance_rating <= 2]
        low = lr[["participant_id", "task_id", "query", "response", "doc_id",
                  "similarity_score", "relevance_rating"]].to_dict("records")

    return jsonify(clean({
        "counts": {"complete": len(complete), "in_progress": len(all_sessions) - len(complete)},
        "targets": aggregate.TARGETS, "overall": overall, "per_task": per_task,
        "sessions": rows, "feedback": feedback, "low_rated": low,
        "sus_scores": parts["sus_score"].dropna().tolist() if len(parts) else [],
    }))


@app.get("/api/admin/session/<pid>")
@admin_required
def admin_session(pid):
    return jsonify(load(pid))


@app.delete("/api/admin/session/<pid>")
@admin_required
def admin_delete(pid):
    """Participant withdrawal: permanently deletes their session file."""
    with lock:
        p = path_for(pid)
        if not p.exists():
            abort(404)
        p.unlink()
    return jsonify({"ok": True, "deleted": pid})


@app.get("/api/admin/export/<kind>")
@admin_required
def admin_export(kind):
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if kind == "sessions.json":
        body = json.dumps(aggregate.load_sessions(include_incomplete=True), indent=2, ensure_ascii=False)
        return Response(body, mimetype="application/json", headers={
            "Content-Disposition": f"attachment; filename=chataudit_sessions_{stamp}.json"})
    _, inter, tasks_df, parts = frames(include_incomplete=request.args.get("all") == "1")
    table = {"interactions.csv": inter, "tasks.csv": tasks_df, "participants.csv": parts}.get(kind)
    if table is None:
        abort(404)
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=chataudit_{kind.replace('.csv', '')}_{stamp}.csv"})


@app.post("/api/admin/search")
@admin_required
def admin_search():
    """Try the chatbot without logging anything to participant data."""
    q = (request.get_json(force=True) or {}).get("query", "").strip()
    if not q:
        abort(400)
    results, ms = bot.search(q, top_k=3)
    for r in results:
        r["matched_on"] = bot.explain(q, r["id"]) if r["score"] > 0 else []
    return jsonify(clean({"results": results, "response_time_ms": ms,
                          "would_answer": results[0]["score"] >= bot.min_score,
                          "threshold": bot.min_score}))


@app.get("/api/admin/analysis")
@admin_required
def admin_analysis():
    """RQ2 correlations and the RQ3 failure-coding list, from completed sessions."""
    import analysis  # local module; uses scipy (installed with scikit-learn)
    sessions, inter, tasks_df, parts = frames()
    if not sessions:
        return jsonify({"ready": False})
    pp = analysis.per_participant(inter, tasks_df, parts)
    corr = [
        analysis.spearman(inter["server_response_time_ms"], inter["relevance_rating"], "Response time vs relevance rating", "query"),
        analysis.spearman(pp["RSR"], pp["SUS"], "Retrieval success rate vs SUS", "participant"),
        analysis.spearman(pp["MRS"], pp["SUS"], "Mean relevance vs SUS", "participant"),
        analysis.spearman(pp["MRT"], pp["SUS"], "Mean response time vs SUS", "participant"),
        analysis.spearman(pp["TCR"], pp["SUS"], "Task completion vs SUS", "participant"),
        analysis.spearman(pp["queries"], pp["SUS"], "Queries needed vs SUS", "participant"),
    ]
    if "trust" in pp:
        corr.append(analysis.spearman(pp["trust"], pp["SUS"], "Mean trust vs SUS", "participant"))
    sheet = analysis.coding_sheet(inter, tasks_df)
    sheet["key"] = sheet["participant_id"] + "|" + sheet["task_id"] + "|" + sheet["n"].astype(str)
    return jsonify(clean({"ready": True, "participants": len(sessions), "correlations": corr,
                          "codes": FAILURE_CODES, "items": sheet.to_dict("records")}))


@app.post("/api/admin/code")
@admin_required
def admin_code():
    """Save a failure code for one query (RQ3). Stored in results/coding.json."""
    data = request.get_json(force=True)
    key, code = data.get("key", ""), data.get("code", "")
    if key.count("|") != 2 or (code and code not in FAILURE_CODES):
        abort(400)
    with lock:
        CODING_FILE.parent.mkdir(parents=True, exist_ok=True)
        codes = json.loads(CODING_FILE.read_text()) if CODING_FILE.exists() else {}
        if code:
            codes[key] = {"code": code, "note": str(data.get("note", ""))[:500]}
        else:
            codes.pop(key, None)
        CODING_FILE.write_text(json.dumps(codes, indent=2, ensure_ascii=False), encoding="utf-8")
    return jsonify({"ok": True, "coded": len(codes)})


@app.get("/api/admin/kb")
@admin_required
def admin_kb():
    return jsonify(bot.docs)


@app.get("/api/admin/offline")
@admin_required
def admin_offline():
    out = {}
    for name in ["test_comparison", "test_by_intent_comparison"]:
        f = OFFLINE_DIR / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f).to_dict("records")
    return jsonify(clean(out))


if __name__ == "__main__":
    print("\n  Participant page:  http://127.0.0.1:5050")
    print("  Admin dashboard:   http://127.0.0.1:5050/admin")
    if PASSWORD_WAS_GENERATED:
        print(f"  Admin password (new each start): {ADMIN_PASSWORD}")
        print('  To choose your own: export CHATAUDIT_ADMIN_PASSWORD="..." before python app.py\n')
    app.run(host="127.0.0.1", port=5050, debug=False)
