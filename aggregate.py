"""
ChatAudit - aggregate all session files and compute the proposal's metrics.

Run:  python aggregate.py
Reads sessions/*.json and writes to results/:
    interactions.csv   one row per query (for RQ2/RQ3 and thematic analysis)
    tasks.csv          one row per participant x task
    participants.csv   one row per participant (SUS + open feedback)
    metrics.txt        RSR, MRS, MRT, TCR, SUS overall and per task
"""

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

TARGETS = {"RSR": 80, "MRS": 3.5, "MRT": 500, "TCR": 75, "SUS": 68}


def load_sessions(include_incomplete=False):
    sessions = []
    for f in sorted((BASE / "sessions").glob("P-*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s["status"] == "complete" or include_incomplete:
            sessions.append(s)
    return sessions


def flatten(sessions):
    inter_rows, task_rows, part_rows = [], [], []
    for s in sessions:
        pid = s["participant_id"]
        for tid, t in s["tasks"].items():
            ratings = [i["relevance_rating"] for i in t["interactions"] if i["relevance_rating"] is not None]
            for i in t["interactions"]:
                inter_rows.append({"participant_id": pid, "task_id": tid, **i})
            # Task time: from when the task was shown (or first query, for older
            # sessions) to when the participant clicked "completed".
            begin = t.get("shown_at") or t.get("started_at")
            task_time = None
            if begin and t.get("completed_at"):
                task_time = (pd.Timestamp(t["completed_at"]) - pd.Timestamp(begin)).total_seconds()
            task_rows.append({
                "participant_id": pid,
                "task_id": tid,
                "n_queries": len(t["interactions"]),
                "first_rating": ratings[0] if ratings else None,
                "best_rating": max(ratings) if ratings else None,
                "mean_rating": sum(ratings) / len(ratings) if ratings else None,
                # Succeeded without rephrasing: first answer rated 3 or higher
                "first_try_success": (ratings[0] >= 3) if ratings else None,
                "task_time_s": task_time,
                "completed": t["completed"],
            })
        part_rows.append({
            "participant_id": pid,
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
            "sus_score": (s["sus"] or {}).get("score"),
            **{f"sus_q{k + 1}": a for k, a in enumerate((s["sus"] or {}).get("answers", []))},
            **{f"fb_{k}": v for k, v in (s["feedback"] or {}).items()},
        })
    return pd.DataFrame(inter_rows), pd.DataFrame(task_rows), pd.DataFrame(part_rows)


def metrics(inter, tasks):
    """The four per-task metrics from section 3.5.1 of the proposal."""
    rated = inter.dropna(subset=["relevance_rating"])
    return {
        # % of participant-task attempts whose best answer was rated 3 or higher
        "RSR": 100 * (tasks["best_rating"] >= 3).mean(),
        "MRS": rated["relevance_rating"].mean(),
        "MRT": inter["server_response_time_ms"].mean(),
        "MRT_client": inter["client_response_time_ms"].mean(),
        "TCR": 100 * tasks["completed"].astype(bool).mean(),
        "queries_per_task": tasks["n_queries"].mean(),
        "first_try_pct": 100 * tasks["first_try_success"].dropna().astype(bool).mean(),
        "task_time_s": tasks["task_time_s"].mean(),
        # Mean trust rating (only when the optional trust question was switched on)
        "MTS": inter["trust_rating"].mean() if "trust_rating" in inter and inter["trust_rating"].notna().any() else None,
    }


def fmt(name, value):
    if value is None or pd.isna(value):
        return f"  {name:<18} n/a"
    unit = {"RSR": "%", "TCR": "%", "MRT": " ms", "MRT_client": " ms"}.get(name, "")
    line = f"  {name:<18} {value:8.2f}{unit}"
    if name in TARGETS:
        ok = value < TARGETS[name] if name == "MRT" else value >= TARGETS[name]
        cmp = "<" if name == "MRT" else ">="
        line += f"   target {cmp} {TARGETS[name]}  {'MET' if ok else 'NOT MET'}"
    return line


if __name__ == "__main__":
    sessions = load_sessions()
    if not sessions:
        raise SystemExit("No completed sessions in sessions/ yet.")

    inter, tasks, parts = flatten(sessions)
    inter.to_csv(OUT / "interactions.csv", index=False)
    tasks.to_csv(OUT / "tasks.csv", index=False)
    parts.to_csv(OUT / "participants.csv", index=False)

    lines = [f"ChatAudit metrics - {len(sessions)} completed sessions, "
             f"{len(inter)} queries\n", "OVERALL"]
    overall = metrics(inter, tasks)
    overall["SUS"] = parts["sus_score"].mean()
    lines += [fmt(k, v) for k, v in overall.items()]
    lines.append(f"  {'SUS (sd)':<18} {parts['sus_score'].std():8.2f}")

    for tid in sorted(tasks["task_id"].unique()):
        lines.append(f"\n{tid}")
        m = metrics(inter[inter.task_id == tid], tasks[tasks.task_id == tid])
        lines += [fmt(k, v) for k, v in m.items()]

    # RQ3: answers the participant rated low (1-2), for vocabulary-mismatch analysis
    low = inter[inter.relevance_rating <= 2][["participant_id", "task_id", "query", "doc_id", "similarity_score"]]
    low.to_csv(OUT / "low_rated_queries.csv", index=False)
    lines.append(f"\n{len(low)} low-rated answers (1-2) saved to results/low_rated_queries.csv")

    report = "\n".join(lines)
    (OUT / "metrics.txt").write_text(report, encoding="utf-8")
    print(report)
