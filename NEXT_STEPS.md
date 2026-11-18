# ChatAudit – plan to the 3 December deadline

## 16-day plan (today: Tue 17 Nov)

**Blocker:** you cannot run any session without departmental ethics approval (proposal section 4.3).
If you don't have it yet, chase it today; everything below assumes it.

| Date | Do | Command / output |
| --- | --- | --- |
| Tue 17 – Wed 18 Nov | Read PILOT_CHANGES.md, dry-run the pipeline, freeze the system, send recruitment messages (aim for 20 so you end up with 15+) | `python simulate_sessions.py` → check → delete `sessions_synthetic/` and `results_synthetic/`; `python freeze_check.py` |
| Thu 19 Nov | Pilot with 2 people. Fix only instructions or wording, never the bot. Delete their sessions | admin dashboard → Sessions → delete |
| Fri 20 – Thu 26 Nov | Study sessions, about 3 a day, 30 min each. Back up `sessions/` after each one. **Meanwhile write chapters 1–4** (intro, background, methods, implementation) – they don't need data | `python app.py` |
| Wed 25 Nov | Checkpoint: fewer than 12 participants? Use the proposal's contingency (more qualitative depth) | |
| Fri 27 – Sat 28 Nov | Run all analyses; code failures; second coder; thematic analysis of feedback | see Step 4 |
| Sun 29 – Mon 30 Nov | Write results (offline + audit + study), discussion, RQ5, recommendations | |
| Tue 1 Dec | Conclusion, abstract, appendices (ethics, materials, PILOT_CHANGES); record the 10-min demo video using chat_app.py and the admin dashboard | |
| Wed 2 Dec | Proofread, references, supervisor check, buffer | |
| Thu 3 Dec | Submit | |

## Which script answers which research question

| RQ | Evidence | Script |
| --- | --- | --- |
| RQ1 needs met? | RSR, MRS, MRT, TCR, SUS vs targets; SUS CI and grade | `analysis.py`, `sus_report.py` |
| RQ2 objective vs satisfaction | Spearman correlations | `analysis.py` |
| RQ3 worst task, phrasing | task failure table, phrasing features vs rating, reformulation types; offline controlled audit | `phrasing_analysis.py`, `perturbation_audit.py` |
| RQ4 interface features | thematic analysis of feedback.csv (Braun & Clarke); kappa for the coding | manual + `coding_agreement.py` |
| RQ5 transferability | same pipeline across models (TF-IDF, BM25, hybrid), audit methods from the literature (Harvey; Uberti-Bona Marin), what is domain-specific (KB, tasks, intent map) | `compare_models.py`, `typo_tolerant.py`, `perturbation_audit.py` |

## Offline results so far (test half, after pilot refinement)

- v1 → v2 → v3: Hit@1 40.7% → 80.8% → **82.8%**
- Perturbation audit: typos cost the study bot about 11 points of Hit@1 on the same questions (all p < .001, BH-corrected);
  case, punctuation and text-speak cost nothing.
- Typo-tolerant hybrid (word + character n-grams, tuned on dev, same out-of-scope refusal):
  **86.1%** clean and **76.8%** with typos, against 82.8% and 70.7% for the study bot – the main recommendation.

---

# Background notes

Where things stand: the chatbot, study platform, admin dashboard, offline Bitext evaluation
(v1 → v2: Hit@1 40.7% → 80.8% on the held-out test half) and the TF-IDF vs BM25 comparison are done.
The user study has **not** run yet (there is no `sessions/` folder). The order below gets you from here to a
finished dissertation.

## How the four papers fit in

| Paper | What it gives your project | Where it's used |
| --- | --- | --- |
| Harvey, Kizilcec & Koenecke (2025), *auditing chatbots for dialect-based quality-of-service harms* | The audit method: perturb the **same** prompts (typos, register, dialect), compare quality with paired tests and Benjamini–Hochberg correction | `perturbation_audit.py`, `dialect_transform.py` (offline RQ3 evidence) |
| Uberti-Bona Marin et al. (2026), *auditing AI product recommendations* | Repeat each request, because LLM chatbots give different answers on repeats; consumer-facing vs API views differ | Stability section of `perturbation_audit.py` (`consistency.csv`); a discussion point: your bot is deterministic by design |
| Iftikhar et al. (2023), *single-page, multipage and conversational forms* (JMIR Human Factors) | How a usability study with a conversational interface reports SUS and task measures | `sus_report.py` (CI, benchmark test, grades, adjectives, item analysis) |
| Hill (2024), MSc dissertation, *LLM pull request descriptions* | A model of the dissertation structure for this degree, and inter-rater agreement (Cohen's kappa) for annotations | Chapter plan below; `coding_agreement.py` |

## Step 1 – Offline audit (do now, about 2 minutes)

    python perturbation_audit.py bitext.csv

Results already produced in `results/audit/` (1,800 clean test questions, 5 typo seeds):

- Lowercase, no punctuation and text-speak: **no loss** (removing punctuation gains 12 questions; p = .0005).
  Text-speak only changes stop words, which TF-IDF ignores anyway.
- Typos: the study bot drops from **81.6% to about 71–72% Hit@1** (−9.4 to −10.4 points; all p < .001 after BH correction).
  "Sorry, no answer" doubles (4.6% → about 9.4%) and the top answer changes for about 23% of questions.
- BM25 behaves the same (−9.9 to −10.8 points), so the cause is exact word matching, not the TF-IDF weighting.
- Hardest hit: delivery_period, delivery_options and cancel_order (−16 to −18 points).
- Stability: 100% identical answers on 5 repeats (deterministic), unlike the LLM chatbots in Uberti-Bona Marin et al.
  Paraphrase consistency differs by intent: track_order lands on the same document only 48% of the time.

Optional extensions:
- `--models tfidf,bm25,embeddings` (after `pip install sentence-transformers`) tests whether meaning-based retrieval is more typo-robust.
- Dialects: see `dialect_transform.py`. Multi-VALUE needs a separate Python 3.10 environment (spaCy fails to build on 3.13).
  Run `--export-base` with the same `--per-intent` value as the main run, so the question IDs match.

## Step 2 – Dry run of the study pipeline (before any participant)

    python simulate_sessions.py

This drives the real app with 15 synthetic participants and runs `analysis.py` and `sus_report.py` on them.
The output goes to `sessions_synthetic/` and `results_synthetic/`, never to `sessions/`. Check the report, figures and coding sheet,
then **delete both folders**. Never report these numbers.

## Step 3 – Freeze, pilot, run

1. `python freeze_check.py`, then commit or copy `SYSTEM_FREEZE.md`. Don't change `app.py`, `retrieval.py` or `knowledge_base.json` after this.
2. Pilot with 1–2 people (their data is discarded). Fix only wording in the instructions, never the bot.
3. Run 15–20 participants. Back up `sessions/` after each one.

## Step 4 – Analysis after the study

    python aggregate.py
    python analysis.py                 # RQ1–RQ4 report + 5 figures
    python sus_report.py               # SUS detail for RQ1/RQ2
    python phrasing_analysis.py        # RQ3: worst task, phrasing features, reformulations
    # code the failures (admin Analysis tab or coding_sheet.csv), rerun analysis.py, then:
    python coding_agreement.py sample  # give the sample to a second coder
    python coding_agreement.py compare results/analysis/coding_sample_coder2.csv

Link the two halves in the discussion: if the study's coding finds many "Typo or spelling" failures, the
offline audit shows the size of that harm under controlled conditions (about −10 points of Hit@1). That's the
argument for a spelling-tolerant fix, such as character n-grams or embeddings, in future work.

## Suggested dissertation structure (after Hill, 2024)

1. **Introduction**: problem, aims, RQ1–RQ4
2. **Background**: retrieval chatbots (TF-IDF/BM25), chatbot evaluation and auditing (Harvey et al.; Uberti-Bona Marin et al.), usability measurement (SUS; Iftikhar et al.)
3. **Methodology**: system design, knowledge base v1/v2, dev/test split, offline protocol, perturbation audit, user-study design and ethics, metrics and targets, analysis plan (Spearman, coding and kappa)
4. **Implementation**: architecture, logging, admin dashboard, system freeze
5. **Results**: (a) offline: v1 vs v2 and model comparison; (b) perturbation audit; (c) user study RQ1–RQ4
6. **Discussion**: offline vs user results, typo harm, determinism vs LLM variability, limits (synthetic perturbations as a proxy for real users, small n, one domain, student sample)
7. **Conclusion and future work**
