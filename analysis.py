"""
ChatAudit - study analysis for the dissertation (RQ1-RQ4).

Run after the study (venv active, one extra install the first time):
    pip install matplotlib
    python analysis.py

Reads every completed session in sessions/ and writes to results/analysis/:
    report.md            all numbers and tables, ready to copy into the results chapter
    per_task.csv         RQ1/RQ3: metrics for each task
    correlations.csv     RQ2: Spearman correlations with p-values
    coding_sheet.csv     RQ3: failed and reformulated queries to code (fill the 'code' column)
    feedback.csv         RQ4: open answers for thematic analysis
    fig1_tasks.png ... fig5_rsr_vs_sus.png   figures
If you fill in coding_sheet.csv (or code queries in the admin dashboard's Analysis tab),
running the script again adds the code counts per task to report.md.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import aggregate

BASE = Path(__file__).parent
OUT = BASE / "results" / "analysis"
CODES = ["Vocabulary mismatch", "Missing topic", "Wrong document chosen", "Typo or spelling",
         "Off-topic question", "Answer incomplete", "Other"]
TASK_NAMES = {"T1": "Return Policy", "T2": "Shipping", "T3": "Student Discount",
              "T4": "Product Warranty", "T5": "Order Tracking"}


def load():
    sessions = aggregate.load_sessions()
    if not sessions:
        raise SystemExit("No completed sessions in sessions/ yet.")
    inter, tasks, parts = aggregate.flatten(sessions)
    return sessions, inter, tasks, parts


def per_participant(inter, tasks, parts):
    g = tasks.groupby("participant_id")
    pp = pd.DataFrame({
        "RSR": 100 * g["best_rating"].apply(lambda s: (s >= 3).mean()),
        "MRS": inter.groupby("participant_id")["relevance_rating"].mean(),
        "MRT": inter.groupby("participant_id")["server_response_time_ms"].mean(),
        "TCR": 100 * g["completed"].apply(lambda s: s.astype(bool).mean()),
        "queries": g["n_queries"].sum(),
    })
    if "trust_rating" in inter and inter["trust_rating"].notna().any():
        pp["trust"] = inter.groupby("participant_id")["trust_rating"].mean()
    return pp.join(parts.set_index("participant_id")["sus_score"].rename("SUS"))


def spearman(x, y, label, level):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 5 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return {"pair": label, "level": level, "n": len(d), "rho": None, "p": None, "reading": "too few values"}
    rho, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    strength = "negligible" if abs(rho) < .1 else "weak" if abs(rho) < .3 else "moderate" if abs(rho) < .5 else "strong"
    sig = "significant (p < .05)" if p < .05 else "not significant"
    return {"pair": label, "level": level, "n": len(d), "rho": round(rho, 3), "p": round(p, 4),
            "reading": f"{strength} {'positive' if rho > 0 else 'negative'}, {sig}"}


def describe(s):
    s = s.dropna()
    return f"mean {s.mean():.1f}, median {s.median():.1f}, range {s.min():.1f}–{s.max():.1f}"


def coding_sheet(inter, tasks):
    """Low-rated answers plus every query in a task where the participant had to rephrase."""
    rephrased = tasks[tasks["n_queries"] > 1][["participant_id", "task_id"]]
    rows = inter.merge(rephrased.assign(rephrased=True), how="left", on=["participant_id", "task_id"])
    rows = rows[(rows["relevance_rating"] <= 2) | rows["rephrased"].fillna(False)]
    sheet = rows[["participant_id", "task_id", "n", "query", "doc_id", "similarity_score", "relevance_rating"]].copy()
    sheet["code"], sheet["note"] = "", ""
    # merge codes from an edited sheet and from the admin dashboard
    key = lambda df: df["participant_id"] + "|" + df["task_id"] + "|" + df["n"].astype(str)
    sheet["key"] = key(sheet)
    codes = {}
    old = OUT / "coding_sheet.csv"
    if old.exists():
        prev = pd.read_csv(old, dtype=str).fillna("")
        prev["key"] = key(prev.assign(n=prev["n"]))
        codes.update({k: (c, nt) for k, c, nt in zip(prev["key"], prev["code"], prev["note"]) if c})
    admin = BASE / "results" / "coding.json"
    if admin.exists():
        codes.update({k: (v.get("code", ""), v.get("note", "")) for k, v in json.loads(admin.read_text()).items()})
    sheet["code"] = sheet["key"].map(lambda k: codes.get(k, ("", ""))[0])
    sheet["note"] = sheet["key"].map(lambda k: codes.get(k, ("", ""))[1])
    return sheet.drop(columns="key")


def figures(tasks, inter, pp, per_task, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    blue, grey, red = "#1f3a5f", "#9aa5b1", "#b04a3a"

    # 1. RSR and TCR per task with targets
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(per_task))
    ax.bar([i - .2 for i in x], per_task["RSR"], .4, label="Retrieval success rate", color=blue)
    ax.bar([i + .2 for i in x], per_task["TCR"], .4, label="Task completion rate", color=grey)
    ax.axhline(targets["RSR"], color=blue, ls="--", lw=1, label=f"RSR target ({targets['RSR']}%)")
    ax.axhline(targets["TCR"], color=grey, ls="--", lw=1, label=f"TCR target ({targets['TCR']}%)")
    ax.set_xticks(list(x), [f"{t}\n{TASK_NAMES.get(t, '').replace(' ', chr(10))}" for t in per_task["task"]])
    ax.set_ylim(0, 105); ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Retrieval success and task completion by task")
    fig.tight_layout(); fig.savefig(OUT / "fig1_tasks.png"); plt.close(fig)

    # 2. Relevance rating distribution
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = inter["relevance_rating"].dropna().astype(int).value_counts().reindex(range(1, 6), fill_value=0)
    ax.bar(counts.index, counts.values, color=[red, red, grey, blue, blue])
    ax.set_xlabel("Relevance rating (1 = not at all, 5 = very)"); ax.set_ylabel("Answers")
    ax.set_title(f"Relevance ratings (n = {counts.sum()})")
    fig.tight_layout(); fig.savefig(OUT / "fig2_relevance.png"); plt.close(fig)

    # 3. SUS per participant
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sus = pp["SUS"].dropna().sort_values()
    ax.bar(range(len(sus)), sus.values, color=[blue if v >= 68 else grey for v in sus.values])
    ax.axhline(68, color=red, ls="--", lw=1, label="68 = average usability")
    ax.axhline(sus.mean(), color=blue, lw=1, label=f"Mean = {sus.mean():.1f}")
    ax.set_xticks([]); ax.set_xlabel("Participants (sorted)"); ax.set_ylabel("SUS score"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("SUS score per participant")
    fig.tight_layout(); fig.savefig(OUT / "fig3_sus.png"); plt.close(fig)

    # 4. Response time distribution
    fig, ax = plt.subplots(figsize=(5, 3.2))
    rt = inter["server_response_time_ms"].dropna()
    ax.hist(rt, bins=20, color=blue)
    ax.set_xlabel("Server response time (ms)"); ax.set_ylabel("Queries")
    ax.set_title(f"Response time (mean {rt.mean():.2f} ms; target < 500 ms)")
    fig.tight_layout(); fig.savefig(OUT / "fig4_response_time.png"); plt.close(fig)

    # 5. Participant RSR vs SUS
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    d = pp[["RSR", "SUS"]].dropna()
    ax.scatter(d["RSR"], d["SUS"], color=blue)
    if len(d) >= 5 and d["RSR"].nunique() > 1:
        rho, p = spearmanr(d["RSR"], d["SUS"]); ax.set_title(f"Retrieval success vs SUS (ρ = {rho:.2f}, p = {p:.3f})")
    else:
        ax.set_title("Retrieval success vs SUS")
    ax.set_xlabel("Participant retrieval success rate (%)"); ax.set_ylabel("SUS score")
    fig.tight_layout(); fig.savefig(OUT / "fig5_rsr_vs_sus.png"); plt.close(fig)


def table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions, inter, tasks, parts = load()
    T = aggregate.TARGETS
    overall = aggregate.metrics(inter, tasks)
    overall["SUS"] = parts["sus_score"].mean()
    pp = per_participant(inter, tasks, parts)

    # RQ1 / RQ3: per task
    rows = []
    for tid in sorted(tasks["task_id"].unique()):
        m = aggregate.metrics(inter[inter.task_id == tid], tasks[tasks.task_id == tid])
        t = tasks[tasks.task_id == tid]
        rows.append({"task": tid, "name": TASK_NAMES.get(tid, ""), "RSR": m["RSR"], "MRS": m["MRS"], "TCR": m["TCR"],
                     "first_try_%": m["first_try_pct"], "queries_per_task": m["queries_per_task"],
                     "rephrased_%": 100 * (t["n_queries"] > 1).mean(), "task_time_s": m["task_time_s"]})
    per_task = pd.DataFrame(rows).round(2)
    per_task.to_csv(OUT / "per_task.csv", index=False)

    # RQ2: Spearman correlations
    corr = pd.DataFrame([
        spearman(inter["server_response_time_ms"], inter["relevance_rating"], "Response time vs relevance rating", "query"),
        spearman(pp["RSR"], pp["SUS"], "Retrieval success rate vs SUS", "participant"),
        spearman(pp["MRS"], pp["SUS"], "Mean relevance vs SUS", "participant"),
        spearman(pp["MRT"], pp["SUS"], "Mean response time vs SUS", "participant"),
        spearman(pp["TCR"], pp["SUS"], "Task completion vs SUS", "participant"),
        spearman(pp["queries"], pp["SUS"], "Queries needed vs SUS", "participant"),
    ] + ([spearman(pp["trust"], pp["SUS"], "Mean trust vs SUS", "participant")] if "trust" in pp else []))
    corr.to_csv(OUT / "correlations.csv", index=False)

    # RQ3 coding, RQ4 feedback
    sheet = coding_sheet(inter, tasks)
    sheet.to_csv(OUT / "coding_sheet.csv", index=False)
    fb = parts[["participant_id", "fb_helpful", "fb_disappointed", "fb_would_change"]]
    fb.to_csv(OUT / "feedback.csv", index=False)

    try:
        figures(tasks, inter, pp, per_task, T)
        fig_note = "Figures saved as fig1_tasks.png to fig5_rsr_vs_sus.png."
    except ImportError:
        fig_note = "Figures skipped: run  pip install matplotlib  and run this script again."

    def met(name, v):
        ok = v < T[name] if name == "MRT" else v >= T[name]
        return "met" if ok else "not met"

    lines = [
        "# ChatAudit study results", "",
        f"{len(sessions)} completed sessions, {len(inter)} queries, {len(tasks)} participant-tasks.", "",
        "## RQ1: overall metrics against the proposal targets", "",
        "| Metric | Result | Target | Status | Spread across participants |",
        "| --- | --- | --- | --- | --- |",
        f"| Retrieval success rate | {overall['RSR']:.1f}% | > {T['RSR']}% | {met('RSR', overall['RSR'])} | {describe(pp['RSR'])} |",
        f"| Mean relevance score | {overall['MRS']:.2f} / 5 | > {T['MRS']} | {met('MRS', overall['MRS'])} | {describe(pp['MRS'])} |",
        f"| Mean response time | {overall['MRT']:.2f} ms | < {T['MRT']} ms | {met('MRT', overall['MRT'])} | {describe(pp['MRT'])} |",
        f"| Task completion rate | {overall['TCR']:.1f}% | > {T['TCR']}% | {met('TCR', overall['TCR'])} | {describe(pp['TCR'])} |",
        f"| SUS | {overall['SUS']:.1f} (SD {parts['sus_score'].std():.1f}) | ≥ {T['SUS']} | {met('SUS', overall['SUS'])} | {describe(pp['SUS'])} |",
        "",
        f"First-try success (first answer rated 3+): {overall['first_try_pct']:.1f}%. Mean queries per task: {overall['queries_per_task']:.2f}.",
        "",
        "## RQ1 / RQ3: by task", "", table(per_task), "",
        "## RQ2: Spearman correlations", "", table(corr), "",
        "Spearman's rank correlation is used because ratings are ordinal and the sample is small. "
        "With 15–20 participants, only fairly strong relationships (|ρ| > 0.45 or so) can reach significance.", "",
        "## RQ3: failed and rephrased queries", "",
        f"{len(sheet)} queries to code in coding_sheet.csv ({(sheet['relevance_rating'] <= 2).sum()} rated 1–2).",
    ]
    coded = sheet[sheet["code"] != ""]
    if len(coded):
        counts = coded.pivot_table(index="code", columns="task_id", values="query", aggfunc="count", fill_value=0)
        counts["Total"] = counts.sum(axis=1)
        lines += ["", f"Codes applied to {len(coded)} of {len(sheet)} queries:", "", table(counts.reset_index())]
    else:
        lines += ["", "No codes yet. Suggested codes: " + "; ".join(CODES) + "."]
    lines += ["", "## RQ4: open feedback", "",
              f"{fb['fb_helpful'].astype(bool).sum()} 'helpful', {fb['fb_disappointed'].astype(bool).sum()} 'disappointed' "
              f"and {fb['fb_would_change'].astype(bool).sum()} 'would change' answers in feedback.csv, for thematic analysis "
              "(Braun and Clarke, 2006).", "", fig_note]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\nFull report and files in {OUT}")


if __name__ == "__main__":
    main()
