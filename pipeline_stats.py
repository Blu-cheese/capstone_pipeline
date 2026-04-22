#!/usr/bin/env python3
"""
Pipeline statistics and sanity checks.

Computes health metrics from pipeline output WITHOUT ground-truth annotations.
These are NOT proper evaluation metrics — for precision/recall/F1 you need
annotated ground truth.

Usage:
    python pipeline_stats.py
    python pipeline_stats.py --output-dir ./output
    python pipeline_stats.py --compare meeting_001 meeting_002
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def load_all_triples(output_dir: str):
    """Load all per-meeting triple files."""
    triples_by_meeting = {}
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith("triples_") and fname.endswith(".json"):
            meeting_id = fname[len("triples_"):-len(".json")]
            with open(os.path.join(output_dir, fname)) as f:
                triples_by_meeting[meeting_id] = json.load(f)
    return triples_by_meeting


def load_graph(output_dir: str):
    """Load the flat file graph."""
    entities = {}
    relations = []

    e_path = os.path.join(output_dir, "entities.json")
    r_path = os.path.join(output_dir, "relations.json")

    if os.path.exists(e_path):
        with open(e_path) as f:
            entities = json.load(f)
    if os.path.exists(r_path):
        with open(r_path) as f:
            relations = json.load(f)

    return entities, relations


def confidence_bucket(c: float) -> str:
    if c >= 0.9:
        return "high (≥0.9)"
    if c >= 0.7:
        return "medium (0.7-0.9)"
    if c >= 0.5:
        return "low (0.5-0.7)"
    return "very low (<0.5)"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)


def analyze_single_meeting(meeting_id: str, triples: list):
    """Stats for a single meeting."""
    print_section(f"Meeting: {meeting_id}")

    if not triples:
        print("  (no triples extracted)")
        return

    print(f"  Total triples: {len(triples)}")

    # Entity counts
    subjects = {t["subject"] for t in triples}
    objects = {t["object"] for t in triples}
    all_entities = subjects | objects
    print(f"  Unique entities: {len(all_entities)}")
    print(f"    Appearing as subject: {len(subjects)}")
    print(f"    Appearing as object: {len(objects)}")
    print(f"    Appearing as both: {len(subjects & objects)}")

    # Entity type distribution
    type_counts = Counter()
    for t in triples:
        type_counts[t["subject_type"]] += 1
        type_counts[t["object_type"]] += 1

    print(f"\n  Entity type distribution:")
    total = sum(type_counts.values())
    for etype, count in type_counts.most_common():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {etype:15s} {count:4d} ({pct:5.1f}%) {bar}")

    # Relation type distribution
    rel_counts = Counter(t["relation"] for t in triples)
    print(f"\n  Relation type distribution (top 10):")
    for rel, count in rel_counts.most_common(10):
        pct = 100 * count / len(triples)
        bar = "█" * int(pct)
        print(f"    {rel:25s} {count:4d} ({pct:5.1f}%) {bar}")

    # Confidence distribution
    conf_counts = Counter(confidence_bucket(t.get("confidence", 1.0)) for t in triples)
    print(f"\n  Confidence distribution:")
    for bucket in ["high (≥0.9)", "medium (0.7-0.9)", "low (0.5-0.7)", "very low (<0.5)"]:
        count = conf_counts.get(bucket, 0)
        pct = 100 * count / len(triples) if triples else 0
        bar = "█" * int(pct / 2)
        print(f"    {bucket:20s} {count:4d} ({pct:5.1f}%) {bar}")

    avg_conf = sum(t.get("confidence", 1.0) for t in triples) / len(triples)
    print(f"  Average confidence: {avg_conf:.2f}")


def analyze_cross_meeting(triples_by_meeting: dict):
    """Entity evolution and overlap across meetings."""
    if len(triples_by_meeting) < 2:
        return

    print_section("Cross-meeting analysis")

    # Entity overlap between meetings
    entities_by_meeting = {}
    for mid, triples in triples_by_meeting.items():
        ents = set()
        for t in triples:
            ents.add(t["subject"].lower())
            ents.add(t["object"].lower())
        entities_by_meeting[mid] = ents

    meeting_ids = sorted(triples_by_meeting.keys())

    print(f"\n  Entity overlap matrix (Jaccard similarity):")
    print(f"    {'':20s}", end="")
    for mid in meeting_ids:
        print(f" {mid[:12]:>12s}", end="")
    print()
    for m1 in meeting_ids:
        print(f"    {m1[:20]:20s}", end="")
        for m2 in meeting_ids:
            e1 = entities_by_meeting[m1]
            e2 = entities_by_meeting[m2]
            if not e1 or not e2:
                print(f" {'--':>12s}", end="")
                continue
            jaccard = len(e1 & e2) / len(e1 | e2)
            print(f" {jaccard:>12.2f}", end="")
        print()

    print(f"\n  Interpretation:")
    print(f"    1.00 = identical entities (likely same meeting or bug)")
    print(f"    0.00 = no overlap (entity resolution may be broken OR topics genuinely different)")
    print(f"    ~0.2-0.5 = healthy overlap of recurring people/topics")

    # Entities appearing in multiple meetings (potential evolution tracking)
    entity_appearances = defaultdict(set)
    for mid, ents in entities_by_meeting.items():
        for e in ents:
            entity_appearances[e].add(mid)

    recurring = {e: mids for e, mids in entity_appearances.items() if len(mids) >= 2}

    print(f"\n  Recurring entities (appearing in ≥2 meetings): {len(recurring)}")
    if recurring:
        sorted_recurring = sorted(recurring.items(), key=lambda x: -len(x[1]))
        print(f"  Top 10 most recurring:")
        for entity, mids in sorted_recurring[:10]:
            print(f"    {entity:30s} appears in {len(mids)} meetings: {', '.join(sorted(mids))}")


def analyze_health(triples_by_meeting: dict):
    """Pipeline health indicators."""
    print_section("Pipeline health check")

    all_triples = [t for triples in triples_by_meeting.values() for t in triples]
    if not all_triples:
        print("  ⚠ No triples found. Pipeline may have failed.")
        return

    total_meetings = len(triples_by_meeting)
    empty_meetings = sum(1 for t in triples_by_meeting.values() if not t)
    avg_triples_per_meeting = len(all_triples) / total_meetings if total_meetings else 0

    print(f"  Total meetings processed: {total_meetings}")
    print(f"  Empty meetings (zero triples): {empty_meetings}")
    print(f"  Average triples per meeting: {avg_triples_per_meeting:.1f}")

    # Health flags
    print(f"\n  Health indicators:")

    # Triple volume
    if avg_triples_per_meeting < 5:
        print(f"    ⚠ LOW triple count ({avg_triples_per_meeting:.1f} per meeting)")
        print(f"      Likely causes: prompt issues, transcripts too short, extractor failing silently")
    elif avg_triples_per_meeting > 100:
        print(f"    ⚠ HIGH triple count ({avg_triples_per_meeting:.1f} per meeting)")
        print(f"      Likely cause: LLM extracting spurious relations. Tighten the prompt.")
    else:
        print(f"    ✓ Triple count looks reasonable ({avg_triples_per_meeting:.1f} per meeting)")

    # Entity type diversity
    type_counts = Counter()
    for t in all_triples:
        type_counts[t["subject_type"]] += 1
        type_counts[t["object_type"]] += 1

    unique_types = len(type_counts)
    dominant_type, dominant_count = type_counts.most_common(1)[0]
    dominant_pct = 100 * dominant_count / sum(type_counts.values())

    if unique_types < 3:
        print(f"    ⚠ LOW type diversity (only {unique_types} entity types used)")
        print(f"      Prompt may be too narrow. Check if all 10 types are being extracted.")
    elif dominant_pct > 80:
        print(f"    ⚠ ONE type dominates: {dominant_type} is {dominant_pct:.0f}% of entities")
        print(f"      LLM is biased — other entity types being missed.")
    else:
        print(f"    ✓ Entity types look diverse ({unique_types} types, max {dominant_pct:.0f}%)")

    # UNKNOWN types (indicates extractor failure)
    unknown_count = type_counts.get("UNKNOWN", 0)
    if unknown_count > 0:
        unknown_pct = 100 * unknown_count / sum(type_counts.values())
        print(f"    ⚠ {unknown_pct:.1f}% of entities have UNKNOWN type")
        print(f"      LLM isn't following the taxonomy. Review the prompt.")

    # Confidence health
    avg_conf = sum(t.get("confidence", 1.0) for t in all_triples) / len(all_triples)
    very_low_conf = sum(1 for t in all_triples if t.get("confidence", 1.0) < 0.5)

    if avg_conf < 0.6:
        print(f"    ⚠ LOW average confidence ({avg_conf:.2f})")
        print(f"      Either LLM is uncertain or extractor prompt is unclear.")
    elif very_low_conf / len(all_triples) > 0.3:
        print(f"    ⚠ {100*very_low_conf/len(all_triples):.0f}% of triples have very low confidence")
    else:
        print(f"    ✓ Confidence scores look healthy (avg {avg_conf:.2f})")

    # Self-referential triples (subject == object, usually a bug)
    self_ref = sum(1 for t in all_triples if t["subject"].lower() == t["object"].lower())
    if self_ref > 0:
        print(f"    ⚠ {self_ref} triples have subject == object (likely bug)")

    # Speaker-ID leakage (SPEAKER_XX showing up as entity — entity resolution failure)
    speaker_leak = sum(1 for t in all_triples
                       if "SPEAKER_" in t["subject"] or "SPEAKER_" in t["object"])
    if speaker_leak > 0:
        pct = 100 * speaker_leak / len(all_triples)
        print(f"    ⚠ {speaker_leak} triples ({pct:.0f}%) contain raw SPEAKER_XX labels")
        print(f"      LLM isn't resolving speaker IDs to names. Check the prompt.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Specific meeting IDs to compare")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Output directory not found: {args.output_dir}")
        print("Run the pipeline first.")
        return

    triples_by_meeting = load_all_triples(args.output_dir)

    if not triples_by_meeting:
        print(f"No triple files found in {args.output_dir}")
        print("Run the pipeline first.")
        return

    if args.compare:
        triples_by_meeting = {k: v for k, v in triples_by_meeting.items() if k in args.compare}

    # Per-meeting analysis
    for mid, triples in triples_by_meeting.items():
        analyze_single_meeting(mid, triples)

    # Cross-meeting
    if len(triples_by_meeting) >= 2:
        analyze_cross_meeting(triples_by_meeting)

    # Health check
    analyze_health(triples_by_meeting)

    print_section("Reminder")
    print("""
  These are SANITY CHECKS, not evaluation metrics.
  Proper metrics (precision/recall/F1) require human-annotated ground truth.

  To run proper evaluation, you need:
    1. Annotate a subset of meetings (spreadsheet with correct triples)
    2. Build an evaluation script that compares extracted vs annotated
    3. Compute per-relation-type F1 and aggregate
""")


if __name__ == "__main__":
    main()
