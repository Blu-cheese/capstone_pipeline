#!/usr/bin/env python3
"""
End-to-end question-answering benchmark. Run from capstone_pipeline:

    venv/bin/python benchmarks/report_qa.py --neo4j-password capstone123
    venv/bin/python benchmarks/report_qa.py --neo4j-password capstone123 -n 5

Asks the natural-language interface questions whose answers are known from
the source scripts, and grades the responses.

This is the only evaluation that measures the pipeline end to end — audio
through diarization, extraction, entity resolution, graph insertion, Cypher
generation and answer synthesis. Every other metric measures one stage.

Grading is keyword-based against expected facts, so it is approximate by
construction: it can mark a good answer wrong if it phrases a fact
unusually. Every answer is printed in full so the score can be checked by
eye rather than trusted. Questions marked CROSS-MEETING require joining
facts stated in different meetings weeks apart.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import load_env
load_env()

from graph.neo4j_graph import KnowledgeGraph
import query as query_mod


# Each entry: the question, and the facts an answer must contain.
# `expects` is a list of groups; every group must match at least one of its
# alternatives. Source script is given so any disputed grading is checkable.
QUESTIONS = [
    dict(q="Who proposed the Applied Deep Learning course?",
         expects=[["ramesh"]], src="script_01", cross=False),

    dict(q="What was the original GPU budget proposed for the lab upgrade?",
         expects=[["3 lakh", "three lakh", "300000", "300,000"]],
         src="script_01", cross=False),

    dict(q="Which vendor was quoted for the lab workstations?",
         expects=[["dell"]], src="script_01", cross=False),

    dict(q="How much was the purchase order for phase two of the lab?",
         expects=[["1.9", "1 point 9"]], src="script_04", cross=True),

    dict(q="What assessment component was added to the Applied Deep Learning course?",
         expects=[["mini project", "project"]], src="script_02", cross=False),

    dict(q="How much was approved for phase one of the GPU purchase?",
         expects=[["2.2", "2 point 2"]], src="script_02", cross=True),

    dict(q="Who chaired the Board of Studies meeting?",
         expects=[["keshavan"]], src="script_03", cross=False),

    dict(q="What happened to the Data Mining course across the meetings?",
         expects=[["merg", "replac", "defer", "postpon"]],
         src="script_03+05", cross=True),

    dict(q="How long is phase one of the capstone project now?",
         expects=[["14 week", "fourteen week"]], src="script_03", cross=False),

    dict(q="Which industry collaboration was prioritised first?",
         expects=[["infosys"]], src="script_03", cross=False),

    dict(q="What did the dedicated power line for the lab cost?",
         expects=[["18,000", "18000", "eighteen thousand"]],
         src="script_04", cross=False),

    dict(q="How many students enrolled in the Applied Deep Learning elective?",
         expects=[["47", "forty-seven", "forty seven"]], src="script_04", cross=False),

    dict(q="What were the TCS placement drive results?",
         expects=[["31", "thirty-one", "thirty one"]], src="script_04", cross=False),

    dict(q="Why has the Infosys agreement not been signed yet?",
         expects=[["exclusiv", "clause", "legal", "recruit"]],
         src="script_05", cross=True),

    dict(q="Who will teach the merged Data Mining and Statistical Learning course?",
         expects=[["kumar"], ["meera"]], src="script_05", cross=True),
]


def norm(s):
    """Lowercase and collapse whitespace so phrasing differences don't matter."""
    return re.sub(r"\s+", " ", str(s).lower())


# An answer that declines to answer must never score, however many keywords
# it happens to echo back from the question. Without this, "the information
# does not explain why the cost increased" scored a PASS on the word
# "increase".
NON_ANSWER = [
    "not specified", "not mentioned", "does not specify", "does not explain",
    "does not contain", "no information", "not provided", "couldn't find",
    "could not find", "unable to", "not available", "does not indicate",
    "doesn't specify", "no relevant", "not enough information",
]


def is_non_answer(answer):
    a = norm(answer)
    return any(p in a for p in NON_ANSWER)


def grade(answer, expects):
    """Every group must match at least one alternative. Returns (ok, missing)."""
    if is_non_answer(answer):
        return False, ["declined to answer"]
    a = norm(answer)
    missing = []
    for group in expects:
        if not any(norm(alt) in a for alt in group):
            missing.append(" / ".join(group))
    return not missing, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="capstone123")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("-n", "--limit", type=int, default=0,
                    help="Run only the first N questions")
    ap.add_argument("--show-cypher", action="store_true")
    args = ap.parse_args()

    graph = KnowledgeGraph(uri=args.neo4j_uri, user=args.neo4j_user,
                           password=args.neo4j_password)
    if not graph.connect():
        print("Neo4j is not reachable. Start it and re-run:")
        print("  docker start capstone-neo4j")
        return 1

    llm_kwargs = {"model": args.llm_model} if args.llm_model else {}
    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS

    print("=" * 78)
    print("END-TO-END QA BENCHMARK".center(78))
    print("=" * 78)
    print("Questions answered from the knowledge graph alone. Ground truth is")
    print("the meeting scripts the audio was generated from. The full pipeline")
    print("runs behind every answer: audio -> diarization -> extraction ->")
    print("entity resolution -> graph -> Cypher -> answer.\n")

    correct, results, t0 = 0, [], time.time()
    for i, item in enumerate(questions, 1):
        tag = "CROSS-MEETING" if item["cross"] else "single meeting"
        print("-" * 78)
        print(f"[{i:2d}/{len(questions)}] {item['q']}")
        print(f"         ground truth: {item['src']}  ({tag})")
        try:
            answer = query_mod.ask(item["q"], graph=graph,
                                   verbose=args.show_cypher, **llm_kwargs)
        except Exception as e:
            answer = f"<error: {e}>"
        ok, missing = grade(answer, item["expects"])
        correct += ok
        results.append((item, ok, answer))

        wrapped = re.sub(r"(.{1,66})(\s|$)", r"\1\n                  ",
                         answer.strip())
        print(f"         answer:  {wrapped.strip()}")
        print(f"         {'PASS' if ok else 'FAIL'}"
              + ("" if ok else f"  (missing: {'; '.join(missing)})"))

    elapsed = time.time() - t0
    n = len(questions)
    cross = [r for r in results if r[0]["cross"]]
    cross_ok = sum(1 for r in cross if r[1])

    print("=" * 78)
    print("RESULT".center(78))
    print("=" * 78)
    print(f"  correct                {correct}/{n}   ({100*correct/n:.0f}%)")
    if cross:
        print(f"  cross-meeting correct  {cross_ok}/{len(cross)}   "
              f"({100*cross_ok/len(cross):.0f}%)")
    print(f"  elapsed                {elapsed:.0f}s "
          f"({elapsed/n:.1f}s per question)")

    if correct < n:
        print("\n  Not answered correctly:")
        for item, ok, _ in results:
            if not ok:
                print(f"    - {item['q']}")

    print(f"""
  Cross-meeting questions require joining facts stated weeks apart in
  different meetings. Answering them is the continual-accumulation claim
  demonstrated rather than asserted: no single meeting contains the answer.

  Grading is keyword-based and therefore approximate — it can mark a
  correct answer wrong for unusual phrasing. Every answer is printed above
  so the score can be audited by eye.
""")
    graph.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
