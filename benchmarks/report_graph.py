#!/usr/bin/env python3
"""
Knowledge-graph benchmark and verification. Run from capstone_pipeline:

    venv/bin/python benchmarks/report_graph.py --neo4j-password capstone123
    venv/bin/python benchmarks/report_graph.py --offline   # no Neo4j needed

Prints the numbers behind the §9.1 deliverable and re-checks, from the live
graph, every property the Aug 30 fixes were supposed to establish. Each check
maps to a defect that previously shipped silently — see
results/PIPELINE_FIXES_AUG30.md.

--offline reads output/triples_*.json instead of Neo4j, so the checks still
run on a fresh clone with no database.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import load_env
load_env()

REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "output"
MEETINGS = [f"meeting_{i:03d}" for i in range(1, 6)]

_passed, _failed = 0, 0


def check(name, cond, extra=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}" + (f"  ({extra})" if extra else ""))
    else:
        _failed += 1
        print(f"  FAIL  {name}" + (f"  ({extra})" if extra else ""))


def load_offline():
    """Rebuild the relation list from the committed triple files."""
    rels = []
    for m in MEETINGS:
        p = OUTPUT / f"triples_{m}.json"
        if p.exists():
            rels.extend(json.loads(p.read_text(encoding="utf-8")))
    return rels


def load_neo4j(uri, user, password):
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver(uri, auth=(user, password))
    with drv.session() as s:
        rows = s.run("""
            MATCH (a:Entity)-[r:RELATION]->(b:Entity)
            RETURN a.name AS subject, a.type AS subject_type,
                   r.type AS relation, b.name AS object, b.type AS object_type,
                   r.source_meeting AS source_meeting, r.confidence AS confidence
        """).data()
    drv.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="capstone123")
    ap.add_argument("--offline", action="store_true",
                    help="Read output/triples_*.json instead of Neo4j")
    args = ap.parse_args()

    if args.offline:
        rels, source = load_offline(), "output/triples_*.json"
    else:
        try:
            rels = load_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            source = args.neo4j_uri
        except Exception as e:
            print(f"[WARN] Neo4j unavailable ({e}); falling back to offline.\n")
            rels, source = load_offline(), "output/triples_*.json"

    if not rels:
        print("No graph data found. Run pipeline.py first.")
        return 1

    entities = {}
    for r in rels:
        entities.setdefault(r["subject"], r.get("subject_type"))
        entities.setdefault(r["object"], r.get("object_type"))

    print("=" * 74)
    print("KNOWLEDGE GRAPH — BENCHMARK".center(74))
    print("=" * 74)
    print(f"source: {source}\n")
    print(f"  entities   {len(entities)}")
    print(f"  relations  {len(rels)}")

    per_meeting = Counter(r.get("source_meeting") for r in rels)
    print("\n  relations per meeting")
    for m in MEETINGS:
        print(f"    {m}  {per_meeting.get(m, 0):>4}")

    # ---- entity and relation distribution ------------------------------
    print("\n  entity types")
    for t, n in Counter(v for v in entities.values() if v).most_common():
        print(f"    {t:<12} {n:>4}")
    print("\n  relation types (top 8)")
    rel_counts = Counter(r["relation"] for r in rels)
    for t, n in rel_counts.most_common(8):
        print(f"    {t:<18} {n:>4}")

    # ---- cross-meeting linkage: the §9.1 headline ----------------------
    spans = {}
    for r in rels:
        for e in (r["subject"], r["object"]):
            spans.setdefault(e, set()).add(r.get("source_meeting"))
    multi = {e: len(ms) for e, ms in spans.items() if len(ms) > 1}
    print(f"\n  entities appearing in >1 meeting: {len(multi)}")
    for n in sorted({v for v in multi.values()}, reverse=True):
        names = [e for e, v in multi.items() if v == n]
        print(f"    spanning {n} meetings: {len(names)}")
        if n >= 4:
            for e in sorted(names)[:6]:
                print(f"       - {e}")

    print("\n" + "=" * 74)
    print("VERIFICATION — each maps to a defect fixed on Aug 30".center(74))
    print("=" * 74)

    # Defect 3: SPEAKER_n as PERSON entities (was 43% of triples)
    import re
    ph = re.compile(r'^SPEAKER[_ ]?\d+$', re.IGNORECASE)
    placeholders = [e for e in entities if ph.match(str(e).strip())]
    check("no SPEAKER_n placeholder nodes", not placeholders,
          f"found {placeholders[:3]}" if placeholders else "was 170 of 385 triples")

    # Defect 2: voice collision merged speakers, so a person went missing
    roster_path = REPO / "data" / "rosters.json"
    if roster_path.exists():
        rosters = json.loads(roster_path.read_text(encoding="utf-8"))
        everyone = sorted({n for v in rosters.values() for n in v})
        present = [n for n in everyone if n in entities]
        check("every roster attendee exists as a node",
              len(present) == len(everyone),
              f"{len(present)}/{len(everyone)}: {sorted(set(everyone) - set(present))}")

        # ASR name variants should have collapsed onto the roster spelling
        from preprocessing.entity_resolution import soundex
        by_sound = {}
        for n in everyone:
            by_sound.setdefault(soundex(n), []).append(n)
        dupes = []
        for e in entities:
            s = str(e).strip()
            if " " in s or s in everyone:
                continue
            if soundex(s) in by_sound:
                dupes.append((s, by_sound[soundex(s)][0]))
        check("no unmerged ASR name variants", not dupes,
              f"{dupes[:3]}" if dupes else "e.g. Jaishree/Mira collapsed")

    # All five meetings made it in
    check("all five meetings present",
          all(per_meeting.get(m, 0) > 0 for m in MEETINGS),
          ", ".join(f"{m}={per_meeting.get(m,0)}" for m in MEETINGS))

    # Defect 4: coverage, which adherence cannot see
    cov_p, adh_p = OUTPUT / "coverage.json", OUTPUT / "adherence_constrained.json"
    if adh_p.exists():
        adh = json.loads(adh_p.read_text(encoding="utf-8"))
        check("taxonomy adherence is 100%", adh.get("adherence_pct") == 100.0,
              f"{adh.get('adherence_pct')}% over {adh.get('triples_seen')} relations")
    if cov_p.exists():
        cov = json.loads(cov_p.read_text(encoding="utf-8"))
        check("no extraction windows failed", cov.get("windows_failed") == 0,
              f"{cov.get('windows_extracted')}/{cov.get('windows_total')} windows")
    else:
        print("  note: output/coverage.json absent — re-run pipeline.py to record it")

    # Every relation is in the taxonomy
    from extractors.llm_extractor import RELATION_TYPES
    off = sorted(set(rel_counts) - set(RELATION_TYPES))
    check("every relation is in the taxonomy", not off, f"off-taxonomy: {off}")

    # Cross-meeting linkage is what makes this one graph rather than five
    check("graph is linked across meetings", len(multi) >= 10,
          f"{len(multi)} bridging entities")

    print("\n" + "=" * 74)
    print(f"{_passed} passed, {_failed} failed")
    print("=" * 74)

    print("""
CONTEXT FOR THE SLIDE

  Constrained extraction took taxonomy adherence from 29% to 100% with
  zero invalid labels, and yield ROSE (55 -> 85) — constraining the label
  space cost no recall. But teacher-vs-gold agreement is only ~37%: schema
  constraints guarantee validity, not correctness. That is the headline
  result and it is more interesting than the distillation angle.

  The cross-meeting entity count above is the April before/after: the old
  graph showed ~50 disconnected components because meeting_001's relations
  were 96% off-taxonomy, so nothing joined up.
""")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
