"""
Quick sanity test: realistic queries for each task, with the topic
we expect TF-IDF to retrieve. Prints accuracy per task.
"""

from retrieval import Retriever

TEST_QUERIES = {
    "T1 returns": [
        ("can I return my headphones", "returns"),
        ("headphones sound bad want refund", "returns"),
        ("how do I send something back", "returns"),
        ("return policy", "returns"),
    ],
    "T2 shipping": [
        ("how fast can you deliver", "shipping"),
        ("express delivery cost", "shipping"),
        ("I need a gift to arrive tomorrow", "shipping"),
        ("shipping prices", "shipping"),
    ],
    "T3 discounts": [
        ("do you have a student discount", "discounts"),
        ("I'm at uni, can I get money off", "discounts"),
        ("how do I get student code", "discounts"),
        ("cheaper for students?", "discounts"),
    ],
    "T4 warranty": [
        ("my laptop won't turn on", "warranty"),
        ("is my laptop under warranty", "warranty"),
        ("how to claim warranty", "warranty"),
        ("laptop broken after few months", "warranty"),
    ],
    "T5 tracking": [
        ("where is my order", "tracking"),
        ("no shipping confirmation email", "tracking"),
        ("ordered 5 days ago nothing yet", "tracking"),
        ("track my parcel", "tracking"),
    ],
}

if __name__ == "__main__":
    bot = Retriever()
    total = correct = 0
    for task, queries in TEST_QUERIES.items():
        hits = 0
        print(f"\n== {task} ==")
        for q, expected in queries:
            res, ms = bot.search(q)
            top = res[0]
            ok = top["topic"] == expected
            hits += ok
            mark = "OK  " if ok else "MISS"
            print(f"  {mark} '{q}' -> {top['id']} {top['title']} (score {top['score']}, {ms}ms)")
        print(f"  {hits}/{len(queries)} correct")
        total += len(queries)
        correct += hits
    print(f"\nOverall topic accuracy: {correct}/{total} = {correct / total:.0%}")
