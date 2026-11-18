"""
ChatAudit - dry run of the whole study pipeline with SYNTHETIC participants.

Use this BEFORE the real study, to prove that every step works end to end:
participant API -> session files -> aggregate.py -> analysis.py -> sus_report.py.
It drives the real Flask app through its test client (no server, no browser), so
the chatbot answers are the real ones; only the "participants" are simulated.

Synthetic data is kept completely apart from real data:
    sessions_synthetic/          fake session files (never sessions/)
    results_synthetic/           metrics, analysis report, figures, SUS report
Delete both folders before the real study. NEVER report these numbers.

Usage:
    python simulate_sessions.py            # 15 participants
    python simulate_sessions.py --n 20 --seed 3
"""

import argparse
import random
import shutil
from pathlib import Path

BASE = Path(__file__).parent
SIM_SESSIONS = BASE / "sessions_synthetic"
SIM_RESULTS = BASE / "results_synthetic"

# A few realistic ways people might phrase each task (first try, then rephrasings).
QUERIES = {
    "T1": ["can i return headphones", "return policy", "my headphones sound bad can I send them back",
           "how do i return a faulty item", "refund for poor quality product"],
    "T2": ["delivery options", "how fast can you deliver", "next day delivery cost",
           "urgent gift delivery", "shipping speeds and prices"],
    "T3": ["student discount", "do you have discounts for students", "how do I get the student discount",
           "uni student offer", "verify student status"],
    "T4": ["laptop wont turn on", "warranty", "is my laptop covered under warranty",
           "how to make a warranty claim", "broken laptop repair"],
    "T5": ["where is my order", "no shipping confirmation", "order not shipped after 5 days",
           "track my order", "order delayed"],
}
# Documents that genuinely answer each task (used only to simulate ratings).
RELEVANT = {
    "T1": {"RET01", "RET02", "RET03"}, "T2": {"SHP01", "SHP03", "SHP04"},
    "T3": {"DIS01", "DIS02"}, "T4": {"WAR01", "WAR02", "WAR04"}, "T5": {"ORD02", "ORD04", "ORD01"},
}
TOPIC_PREFIX = {"T1": "RET", "T2": "SHP", "T3": "DIS", "T4": "WAR", "T5": "ORD"}


def typo(q, rng):
    from perturbation_audit import add_typos
    return add_typos(q, "mixed", rng, 0.2)


def simulate(n, seed):
    import app as A  # the real study app
    A.SESSIONS_DIR = SIM_SESSIONS  # path_for() reads this at call time
    SIM_SESSIONS.mkdir(exist_ok=True)
    rng = random.Random(seed)
    client = A.app.test_client()
    doc_of = {d["text"]: d["id"] for d in A.bot.docs}

    for _ in range(n):
        careful = rng.random()  # 0 = sloppy typist, 1 = careful
        pid = client.post("/api/start").get_json()["participant_id"]
        successes = 0
        for tid in A.TASK_IDS:
            client.post("/api/task_start", json={"participant_id": pid, "task_id": tid})
            pool = QUERIES[tid][:]
            rng.shuffle(pool)
            ok = False
            patience = rng.choice([1, 2, 3, 3])  # some people give up after one try
            for attempt, q in enumerate(pool[:patience]):
                if rng.random() > careful:
                    q = typo(q, rng)
                r = client.post("/api/query", json={"participant_id": pid, "task_id": tid, "query": q}).get_json()
                doc = doc_of.get(r["response"])
                if doc in RELEVANT[tid]:
                    rating = rng.choice([3, 4, 5, 5])
                elif doc and doc.startswith(TOPIC_PREFIX[tid]):
                    rating = rng.choice([2, 3, 3])
                else:
                    rating = rng.choice([1, 1, 2])
                client.post("/api/rate", json={"participant_id": pid, "task_id": tid, "n": r["n"],
                                               "rating": rating, "client_response_time_ms": rng.randint(40, 250)})
                if rating >= 4:
                    ok = True
                    break
            successes += ok
            done = ok or rng.random() < .25  # some say "done" without a good answer
            client.post("/api/complete", json={"participant_id": pid, "task_id": tid, "completed": done})
        # SUS loosely tied to how well the tasks went (odd items positive, even negative)
        mood = successes / len(A.TASK_IDS)
        sus = []
        for i in range(10):
            pos = 2 + round(3 * mood + rng.uniform(-1, 1))
            pos = min(5, max(1, pos))
            sus.append(pos if i % 2 == 0 else 6 - pos)
        client.post("/api/finish", json={
            "participant_id": pid, "sus": sus,
            "helpful": "[synthetic] quick answers", "disappointed": "[synthetic] missed some questions",
            "would_change": "[synthetic] understand typos"})
    print(f"Created {n} synthetic sessions in {SIM_SESSIONS}")


def run_pipeline():
    import aggregate
    import analysis
    import phrasing_analysis
    import sus_report

    # Point every script at the synthetic folders (the real ones are not touched).
    def load_sim(include_incomplete=False):
        import json
        return [s for s in (json.loads(f.read_text(encoding="utf-8")) for f in sorted(SIM_SESSIONS.glob("P-*.json")))
                if s["status"] == "complete" or include_incomplete]

    aggregate.load_sessions = load_sim
    analysis.OUT = SIM_RESULTS / "analysis"
    analysis.BASE = SIM_RESULTS  # coding.json lookup goes to results_synthetic/results/
    analysis.main()
    sus_report.main(["--sessions", str(SIM_SESSIONS), "--out", str(SIM_RESULTS / "sus")])
    phrasing_analysis.main(["--sessions", str(SIM_SESSIONS), "--out", str(SIM_RESULTS / "phrasing")])
    print(f"\nSynthetic outputs in {SIM_RESULTS}. Check them, then delete "
          f"{SIM_SESSIONS.name}/ and {SIM_RESULTS.name}/ before the real study.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    for d in (SIM_SESSIONS, SIM_RESULTS):
        shutil.rmtree(d, ignore_errors=True)
    simulate(args.n, args.seed)
    run_pipeline()
