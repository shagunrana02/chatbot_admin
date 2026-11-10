# ChatAudit

TF-IDF customer-support chatbot + user study platform with an admin dashboard.

## Setup (once)
    python3 -m venv venv
    source venv/bin/activate
    pip install flask scikit-learn pandas scipy matplotlib
    # optional, for the embeddings model in compare_models.py (about 1 GB):
    pip install sentence-transformers

## Run
    source venv/bin/activate
    export CHATAUDIT_ADMIN_PASSWORD="choose-a-password"   # optional
    python app.py

- Participant page: http://127.0.0.1:5050
- Admin dashboard:  http://127.0.0.1:5050/admin
  If you did not set a password, a new random one is printed in the terminal at each start.

Each participant's data is saved to `sessions/<code>.json` after every click.
If the browser crashes, reopen `http://127.0.0.1:5050/?pid=<code>` to resume.

## Admin dashboard
- Overview: RSR, MRS, MRT, TCR, SUS against the proposal targets, per-task table, SUS distribution, low-rated answers
- Sessions: every participant with the full conversation; delete a session when a participant withdraws
- Feedback: the three open-question answers, for thematic analysis
- Test chatbot: try any query and see the top 3 documents and matched words (not saved)
- Knowledge base: all documents and hidden keywords, plus offline Bitext results
- Analysis: RQ2 Spearman correlations (objective measures vs satisfaction) and the RQ3 failure-coding sheet (pick a failure code for each low-rated or rephrased query; saved to `results/coding.json`)
- Export: interactions.csv, tasks.csv, participants.csv, sessions.json

## API
Participant (no login): `GET /api/tasks`, `POST /api/start`, `GET /api/session/<pid>`,
`POST /api/task_start`, `POST /api/query`, `POST /api/rate`, `POST /api/complete`, `POST /api/finish`

Admin (login required): `POST /api/admin/login`, `POST /api/admin/logout`, `GET /api/admin/summary`,
`GET|DELETE /api/admin/session/<pid>`, `GET /api/admin/export/<interactions.csv|tasks.csv|participants.csv|sessions.json>`,
`POST /api/admin/search`, `GET /api/admin/kb`, `GET /api/admin/offline`,
`GET /api/admin/analysis`, `POST /api/admin/code`

Also `GET /api/config` (no login): tells the participant page whether the trust question is on.

## Optional trust rating
`study_config.json` contains `{"ask_trust": false}`. Set it to `true` only if your ethics
approval covers a trust question. Participants then rate every answer for relevance AND
"How much would you trust this answer?" (1-5). Mean trust (MTS) appears in the admin Overview.
Restart `app.py` after changing it.

## Other scripts
- `python aggregate.py` – metrics report and CSVs in `results/`
- `python offline_eval.py bitext.csv --split test` – offline retrieval test on the Bitext dataset
- `python test_retrieval.py` – quick 20-query check
- `python compare_models.py bitext.csv` – offline comparison of TF-IDF vs BM25 vs sentence embeddings
  (embeddings are skipped if sentence-transformers is not installed); writes `results/offline/model_*.csv`
- `python analysis.py` – after the study: per-task table, Spearman correlations, coding sheet,
  feedback and 5 figures for the dissertation in `results/analysis/`
- `python freeze_check.py` – records versions and file fingerprints in `SYSTEM_FREEZE.md`

Everything runs locally on this laptop; no data leaves it. Back up `sessions/` after every session.
