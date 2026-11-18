"""
ChatAudit - System Usability Scale (SUS) report for the dissertation.

Goes further than the single mean in analysis.py, in the way usability studies
such as Iftikhar et al. (2023, JMIR Human Factors) report SUS:
  - mean, SD, median and a 95% confidence interval (t distribution)
  - test against the benchmark of 68 (one-sample t-test and Wilcoxon signed-rank)
  - percentile rank, curved grade (Sauro & Lewis, 2016) and adjective
    (Bangor, Kortum & Miller, 2009), for the mean and for each participant
  - Cronbach's alpha for the 10 items (reliability), item means after recoding

Usage:
    python sus_report.py                     # reads sessions/
    python sus_report.py --sessions sessions_synthetic --out results_synthetic/sus

Writes results/sus/: sus_report.md, sus_participants.csv, sus_items.csv, fig_sus.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent

ITEMS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

# Sauro & Lewis (2016) curved grading scale: (lower bound, grade, percentile range)
GRADES = [(84.1, "A+", "96-100"), (80.8, "A", "90-95"), (78.9, "A-", "85-89"), (77.2, "B+", "80-84"),
          (74.1, "B", "70-79"), (72.6, "B-", "65-69"), (71.1, "C+", "60-64"), (65.0, "C", "41-59"),
          (62.7, "C-", "35-40"), (51.7, "D", "15-34"), (0.0, "F", "0-14")]
# Adjective bands aligned with the curved scale (Bangor et al., 2009; Lewis & Sauro, 2018)
ADJECTIVES = [(84.1, "Best imaginable"), (72.6, "Excellent"), (62.7, "Good"), (51.7, "OK"),
              (25.1, "Poor"), (0.0, "Worst imaginable")]
BENCHMARK, BENCH_SD = 68.0, 12.5  # Sauro (2011): mean and SD over ~500 studies


def grade(s):
    return next((g, p) for lo, g, p in GRADES if s >= lo)


def adjective(s):
    return next(a for lo, a in ADJECTIVES if s >= lo)


def percentile(s):
    return 100 * stats.norm.cdf((s - BENCHMARK) / BENCH_SD)


def cronbach_alpha(items):
    items = np.asarray(items, float)
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def load(sessions_dir):
    rows = []
    for f in sorted(Path(sessions_dir).glob("P-*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("status") == "complete" and s.get("sus"):
            rows.append({"participant_id": s["participant_id"], **{f"q{i + 1}": a for i, a in enumerate(s["sus"]["answers"])}})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=str(BASE / "sessions"))
    ap.add_argument("--out", default=str(BASE / "results" / "sus"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load(args.sessions)
    if raw.empty:
        raise SystemExit(f"No completed sessions with SUS answers in {args.sessions}.")
    q = raw[[f"q{i}" for i in range(1, 11)]].astype(int)
    # Recode to 0-4 "contribution" (Brooke, 1996): odd items a-1, even items 5-a
    contrib = pd.DataFrame({c: (q[c] - 1) if i % 2 == 0 else (5 - q[c]) for i, c in enumerate(q.columns)})
    raw["sus"] = contrib.sum(axis=1) * 2.5
    raw["grade"] = raw["sus"].map(lambda s: grade(s)[0])
    raw["adjective"] = raw["sus"].map(adjective)
    raw["percentile"] = raw["sus"].map(percentile).round(0)
    raw.to_csv(out / "sus_participants.csv", index=False)

    s = raw["sus"]
    n, mean, sd = len(s), s.mean(), s.std(ddof=1)
    half = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else float("nan")
    t_res = stats.ttest_1samp(s, BENCHMARK) if n > 1 else None
    w_p = stats.wilcoxon(s - BENCHMARK).pvalue if n > 1 and (s != BENCHMARK).any() else float("nan")
    alpha = cronbach_alpha(contrib) if n > 2 else float("nan")
    g, pr = grade(mean)

    items = pd.DataFrame({"item": range(1, 11), "statement": ITEMS,
                          "direction": ["positive" if i % 2 == 0 else "negative" for i in range(10)],
                          "mean_raw_1to5": q.mean().round(2).values,
                          "mean_contribution_0to4": contrib.mean().round(2).values})
    items.to_csv(out / "sus_items.csv", index=False)

    lines = [
        "# SUS results", "",
        f"n = {n} participants. Mean SUS = **{mean:.1f}** (SD {sd:.1f}, median {s.median():.1f}, "
        f"range {s.min():.1f}–{s.max():.1f}); 95% CI {mean - half:.1f}–{mean + half:.1f}.", "",
        f"- Grade **{g}** on the Sauro & Lewis (2016) curved scale (percentile band {pr}); "
        f"adjective **{adjective(mean)}** (Bangor et al., 2009); about the {percentile(mean):.0f}th percentile of "
        f"published systems (Sauro, 2011: mean 68, SD 12.5).",
        f"- Against the benchmark of 68: one-sample t({n - 1}) = {t_res.statistic:.2f}, p = {t_res.pvalue:.4f}; "
        f"Wilcoxon signed-rank p = {w_p:.4f}." if t_res is not None else "- Too few participants to test against 68.",
        f"- {100 * (s >= BENCHMARK).mean():.0f}% of participants scored 68 or above.",
        f"- Reliability: Cronbach's alpha = {alpha:.2f} "
        f"({'acceptable' if alpha >= .7 else 'below the usual .70'}; SUS is typically around .90).", "",
        "Is the CI wholly above 68? " + ("Yes: usability is above average with 95% confidence." if mean - half > 68 else
                                          "No: the data cannot rule out average-or-below usability." if mean + half > 68 else
                                          "No, it is wholly below 68: usability is below average."), "",
        "## Grade distribution", "",
        raw["grade"].value_counts().reindex([x[1] for x in GRADES], fill_value=0).loc[lambda v: v > 0].to_frame("participants").to_markdown()
        if _has_tabulate() else raw["grade"].value_counts().to_string(), "",
        "## Item means (contribution 0–4, higher is always better)", "",
        items[["item", "direction", "mean_raw_1to5", "mean_contribution_0to4", "statement"]].to_markdown(index=False)
        if _has_tabulate() else items.to_string(index=False), "",
        "Weakest items (lowest contribution) point to what to discuss: "
        + "; ".join(f"item {r.item} ({r.mean_contribution_0to4:.2f})" for r in items.nsmallest(2, "mean_contribution_0to4").itertuples())
        + ".",
    ]
    try:
        _figure(raw, contrib, out, mean, half)
        lines += ["", "Figure: fig_sus.png"]
    except ImportError:
        pass
    (out / "sus_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"\nSUS report in {out}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _figure(raw, contrib, out, mean, half):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    blue, grey, red = "#1f3a5f", "#9aa5b1", "#b04a3a"
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
    for lo, label in ADJECTIVES[:-1]:
        a.axvline(lo, color=grey, lw=.6)
        a.text(lo + .5, .98, label, rotation=90, va="top", fontsize=7, color=grey, transform=a.get_xaxis_transform())
    a.scatter(raw["sus"], np.random.default_rng(0).uniform(.15, .55, len(raw)), color=blue, s=18, zorder=3)
    a.errorbar(mean, .8, xerr=half, fmt="o", color=red, capsize=4, zorder=4, label=f"Mean {mean:.1f} (95% CI)")
    a.axvline(68, color=red, ls="--", lw=1)
    a.set_xlim(0, 100); a.set_yticks([]); a.set_xlabel("SUS score"); a.legend(frameon=False, fontsize=8, loc="lower left")
    a.set_title("Participants against adjective bands")
    b.barh(range(10, 0, -1), contrib.mean().values, color=[blue if i % 2 == 0 else grey for i in range(10)])
    b.set_yticks(range(10, 0, -1), [f"Q{i}" for i in range(1, 11)]); b.set_xlim(0, 4)
    b.set_xlabel("Mean contribution (0–4, higher = better)"); b.set_title("Items (blue = positive, grey = negative wording)")
    fig.tight_layout(); fig.savefig(out / "fig_sus.png"); plt.close(fig)


if __name__ == "__main__":
    main()
