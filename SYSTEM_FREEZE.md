# ChatAudit system freeze record

Frozen on: 2026-11-21 04:58
Result of final check: PASSED

## Software versions
- Python 3.13.7 on Darwin 25.6.0
- Flask 3.1.3
- scikit-learn 1.9.1
- pandas 3.1.1

## Retrieval model settings
- Model: TF-IDF vectors (scikit-learn TfidfVectorizer) ranked by cosine similarity; top-1 document returned
- Indexed text per document: title + text + hidden keywords (participants only see the text)
- Lowercasing: yes; n-grams: unigrams and bigrams; sublinear term frequency: yes
- Stop words: scikit-learn English list (318 words) plus 44 domain request words: aren, assist, assistance, cant, check, checking, could, did, didn, does, doesn, doing, don, dont, help, helping, id, im, info, information, isn, ive, know, like, ll, need, please, pls, possible, see, seeing, show, tell, try, trying, ve, wanna, want, wasn, way, ways, wish, won, would
- 'No answer' threshold: cosine similarity below 0.1
- Stemming: off

## Knowledge base
- 30 documents in 5 topics: discounts (6), returns (6), shipping (6), tracking (6), warranty (6)
- Documents with hidden keywords: 27
- Version: v2 (synonym keywords added using the Bitext dev split only)

## Task check (one natural question per task)

| Task | Question | Returned | Score | Time (ms) | Correct |
| --- | --- | --- | --- | --- | --- |
| T1 Return Policy | can I return headphones with bad sound | RET02 | 0.293 | 1.25 | yes |
| T2 Shipping | how fast can you deliver and how much does it cost | SHP01 | 0.217 | 0.75 | yes |
| T3 Student Discount | I'm at uni, can I get money off | DIS01 | 0.176 | 0.70 | yes |
| T4 Product Warranty | my laptop won't turn on, is it under warranty | WAR04 | 0.253 | 0.70 | yes |
| T5 Order Tracking | ordered 5 days ago and no shipping confirmation | ORD02 | 0.407 | 0.69 | yes |

Off-topic question ('what is the weather in Glasgow'): fallback given
Mean response time over 100 queries: 0.64 ms (target under 500 ms)

## File fingerprints (SHA-256, first 12 characters)
If any of these change after this date, the system is no longer the frozen version.

- app.py: e82d895538a1
- retrieval.py: db230ee5fffe
- knowledge_base.json: 8ceab1213f6d
- static/index.html: c181e70a9e04
- static/admin.html: 4b9cb8a39d9b
- aggregate.py: 64210e264067

## Data folder
- Session files present: 0 (empty, ready for the study)
