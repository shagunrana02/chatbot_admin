"""
ChatAudit - TF-IDF retrieval core.

Loads the knowledge base, builds a TF-IDF index, and returns the single
most similar document for a user query (cosine similarity).

Usage:
    python retrieval.py            # interactive test loop
    from retrieval import Retriever
"""

import json
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _make_stemming_analyzer():
    """Porter-stemmed analyzer, so 'deliver'/'delivery', 'order'/'ordered' match.
    Requires: pip install nltk"""
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    base = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).build_analyzer()
    return lambda text: [" ".join(stemmer.stem(w) for w in tok.split()) for tok in base(text)]


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Request words that appear in almost every customer question ("I need help to
# check...") but say nothing about the topic. Left in, they match whichever
# document happens to contain "check" or "help" and pull queries to it.
DOMAIN_STOP_WORDS = {
    "help", "helping", "assistance", "assist", "need", "want", "wanna", "like",
    "check", "checking", "see", "seeing", "know", "tell", "show", "try", "trying",
    "wish", "possible", "way", "ways", "info", "information", "please", "pls",
    "could", "would", "id", "im", "ive", "dont", "cant",
}


class Retriever:
    def __init__(self, kb_path="knowledge_base.json", min_score=0.1, use_stemming=False,
                 domain_stop_words=True):
        self.docs = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        # Index title + text + optional hidden keywords (synonyms users might type).
        # Only "text" is ever shown to participants.
        corpus = [f"{d['title']}. {d['text']} {d.get('keywords', '')}" for d in self.docs]
        stops = list(ENGLISH_STOP_WORDS | (DOMAIN_STOP_WORDS if domain_stop_words else set()))

        if use_stemming:
            self.vectorizer = TfidfVectorizer(
                analyzer=_make_stemming_analyzer(), sublinear_tf=True
            )
        else:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=stops,       # drop "the", "is", "how", plus request words
                ngram_range=(1, 2),     # single words + two-word phrases
                sublinear_tf=True,      # dampen repeated terms
            )
        self.doc_matrix = self.vectorizer.fit_transform(corpus)
        self.min_score = min_score  # below this = "no good answer"

    def search(self, query, top_k=1):
        """Return the top_k documents with scores and response time."""
        start = time.perf_counter()
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        results = [
            {
                "id": self.docs[i]["id"],
                "topic": self.docs[i]["topic"],
                "title": self.docs[i]["title"],
                "text": self.docs[i]["text"],
                "score": round(float(scores[i]), 4),
            }
            for i in ranked
        ]
        return results, round(elapsed_ms, 2)

    def answer(self, query):
        """Chatbot behaviour: one answer, or a fallback if nothing matches."""
        results, ms = self.search(query, top_k=1)
        best = results[0]
        if best["score"] < self.min_score:
            return {
                "response": "Sorry, I couldn't find an answer to that. "
                            "Could you try rephrasing your question?",
                "doc_id": None,
                "score": best["score"],
                "response_time_ms": ms,
            }
        return {
            "response": best["text"],
            "doc_id": best["id"],
            "score": best["score"],
            "response_time_ms": ms,
        }

    def explain(self, query, doc_id, top_terms=5):
        """Show which terms drove the match - useful for failure analysis."""
        idx = next(i for i, d in enumerate(self.docs) if d["id"] == doc_id)
        q_vec = self.vectorizer.transform([query]).toarray().ravel()
        d_vec = self.doc_matrix[idx].toarray().ravel()
        contrib = q_vec * d_vec
        terms = self.vectorizer.get_feature_names_out()
        top = contrib.argsort()[::-1][:top_terms]
        return [(terms[i], round(float(contrib[i]), 4)) for i in top if contrib[i] > 0]


if __name__ == "__main__":
    bot = Retriever()
    print(f"Loaded {len(bot.docs)} documents. Type a question (blank to quit).\n")
    while True:
        q = input("You: ").strip()
        if not q:
            break
        r = bot.answer(q)
        print(f"Bot: {r['response']}")
        print(f"     [doc={r['doc_id']} score={r['score']} time={r['response_time_ms']}ms]")
        if r["doc_id"]:
            print(f"     matched on: {bot.explain(q, r['doc_id'])}")
        print()
