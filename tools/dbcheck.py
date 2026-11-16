"""Query the knowledge base and print distances — use it to calibrate RAG_MAX_DISTANCE.

Usage:
    python -m tools.dbcheck                         # built-in sample questions
    python -m tools.dbcheck "jak zrobić brąz" ...   # your own questions
    python -m tools.dbcheck -n 8 --full "..."       # more hits, full chunk text

Ask questions the bot SHOULD answer and some it should not (off-topic). Set the
threshold between the distances of good hits and of junk.
"""

import argparse
from collections import Counter

from vsai import config, kb

SAMPLE_QUESTIONS = [
    "jak zrobić brąz",               # alloy ratios
    "gdzie znaleźć rudę miedzi",     # ore / prospecting
    "jak przetrwać burzę temporalną",
    "how to make steel",
    "jaka jest stolica Francji",     # off-topic: every hit should be FAR
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", nargs="*", default=SAMPLE_QUESTIONS)
    parser.add_argument("-n", type=int, default=5, help="hits per question")
    parser.add_argument("--full", action="store_true", help="print full chunk text")
    args = parser.parse_args()

    collection = kb.open_collection()
    types = Counter(m.get("type", "?") for m in collection.get(include=["metadatas"])["metadatas"])
    print(f"Collection '{config.COLLECTION_NAME}': {collection.count()} chunks, "
          f"model {config.EMBED_MODEL}, metric {kb.SPACE}")
    print("By type:", ", ".join(f"{t}={n}" for t, n in types.most_common()))

    for question in args.questions:
        print(f"\n=== {question}")
        for i, hit in enumerate(kb.search(collection, question, n_results=args.n), 1):
            text = hit.text if args.full else hit.text.replace("\n", " ")[:140]
            print(f"  {i}. {hit.distance:.3f}  {hit.source}\n     {text}")


if __name__ == "__main__":
    main()
