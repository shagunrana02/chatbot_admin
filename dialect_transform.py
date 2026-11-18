"""
ChatAudit - dialect versions of the audit's base questions with Multi-VALUE.

Multi-VALUE (Ziems et al., 2023) is the validated dialect-transformation tool
Harvey et al. (2025) recommend. It needs an older Python (3.9-3.11) because of
its spaCy pin, so run it in its OWN virtual environment, not the study venv:

    python3.10 -m venv venv_dialect
    source venv_dialect/bin/activate          # Windows: venv_dialect\\Scripts\\activate
    pip install value-nlp pandas
    python -m spacy download en_core_web_sm   # if Multi-VALUE asks for it

Then:
    # 1. in the study venv: export the same base questions the audit uses
    python perturbation_audit.py bitext.csv --export-base
    # 2. in venv_dialect: transform them
    python dialect_transform.py results/audit/base_queries.csv results/audit/dialect_queries.csv
    # 3. back in the study venv: run the audit with the dialect versions included
    python perturbation_audit.py bitext.csv --dialect-csv results/audit/dialect_queries.csv

Ethics note (Harvey et al., section 4): transformed text is a proxy for real
dialect speakers, not a replacement for them; say so as a limitation, and if
possible have a speaker of each dialect check a sample.
"""

import sys

import pandas as pd

DIALECTS = {  # the five dialects of the Rufus case study
    "dialect_AppE": "AppalachianDialect",
    "dialect_ChcE": "ChicanoDialect",
    "dialect_AAE": "AfricanAmericanVernacular",
    "dialect_IndE": "IndianDialect",
    "dialect_SgE": "ColloquialSingaporeDialect",
}


def main(src, dst):
    from multivalue import Dialects

    base = pd.read_csv(src)
    rows = []
    for cond, cls_name in DIALECTS.items():
        cls = getattr(Dialects, cls_name, None)
        if cls is None:
            have = [n for n in dir(Dialects) if not n.startswith("_")]
            print(f"{cls_name} not in this Multi-VALUE version; skipped. Available: {', '.join(have)}")
            continue
        d = cls()
        changed = 0
        for r in base.itertuples():
            try:
                t = d.transform(r.query)
            except Exception:  # noqa: BLE001 - keep the original sentence if one fails
                t = r.query
            changed += t != r.query
            rows.append({"qid": r.qid, "condition": cond, "query": t})
        print(f"{cond}: {changed}/{len(base)} questions changed")
    pd.DataFrame(rows).to_csv(dst, index=False)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python dialect_transform.py base_queries.csv dialect_queries.csv")
    main(sys.argv[1], sys.argv[2])
