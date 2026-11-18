# Pre-study knowledge-base refinement (proposal section 4.2, risk 4)

The proposal allows refining the knowledge base if checks before the study reveal
systematic gaps. Realistic queries for the five tasks were run through the bot.
Several were sent to the wrong document. The causes and changes are below.
Every change was checked on the Bitext **dev** half only; the **test** half is reported once.

| Problem query (task) | Wrong answer | Cause | Change |
| --- | --- | --- | --- |
| "How do I return an item?" (T1) | WAR06 Repair times | the bigram "return item" appears only in WAR06's text | RET02 keywords: return an item, return item, return product… |
| "How long does delivery take?" (T2) | WAR03 Warranty exclusions | "does" is not in sklearn's stop-word list and matched "does not cover" | added does/did/doing and contraction fragments (isn, doesn, don, won…) to `DOMAIN_STOP_WORDS` in retrieval.py |
| "shipping speeds and prices" (T2) | DIS06 Price match | "prices" only in DIS06 | SHP01 keywords: prices, speeds, shipping cost, how much, fastest, express, standard |
| "My discount code isn't working" | DIS03 Using a code | DIS04 had no keywords | DIS04 keywords: not working, invalid, rejected, error, declined… |
| repair-time / exclusion questions | – | WAR03, WAR06 had no keywords | WAR03: not covered, exclusions, damage; WAR06: repair time, how long repair |
| "Where is my order?" (T5) | ORD05 Cancelling | after stop words only "order" is left, and ORD05 repeats it | ORD01 keywords: where is my order, my order. **Still fails:** a one-word query is genuinely ambiguous. Keeping "where" as a content word fixed it but cost 0.8 points on dev, so it was rejected. Kept as a known limitation for RQ3. |

Results (Hit@1, answerable questions, with the deployed "no answer" threshold):

| Version | Dev | Test |
| --- | --- | --- |
| v2 (before refinement) | 81.0% | 80.8% |
| v3 (pilot-refined, used in the study) | 82.6% | 82.8% |

Out-of-scope questions correctly refused: 27.3% → 26.4% (test), a small trade-off.
The previous version is kept as `knowledge_base_v2_prepilot.json`; the results are in
`results/offline/dev_v3_pilot/` and `results/offline/test_v3_pilot/`.
Run `python freeze_check.py` after this, and make no further changes to the bot.
