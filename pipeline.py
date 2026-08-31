#!/usr/bin/env python3
"""
Continual Knowledge Graph Construction from Meeting Audio
=========================================================

Main pipeline orchestrator.

Usage:
    # Process a single meeting (text transcript)
    python pipeline.py --input meeting1.txt --meeting-id "staff_meeting_jan"

    # Process a single meeting (SRT from whisper-diarization)
    python pipeline.py --input meeting1.srt --meeting-id "staff_meeting_jan"

    # Process multiple meetings sequentially (continual mode)
    python pipeline.py --input meeting1.txt meeting2.txt meeting3.txt

    # Use flat file graph instead of Neo4j (for development)
    python pipeline.py --input meeting1.txt --no-neo4j

    # Use mock extractor (no API key needed, for testing pipeline flow)
    python pipeline.py --input meeting1.txt --extractor mock
    
    # Use DHGAT extractor
    python pipeline.py --input meeting1.txt --extractor dhgat --dhgat-ckpt runs/best.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from utils.env import load_env
load_env()  # populate os.environ from .env before defaults are computed

from utils.models import Triple, MeetingTranscript
from preprocessing.transcript_parser import parse_transcript
from preprocessing.chunking import chunk_transcript
from preprocessing.entity_resolution import EntityResolver
from preprocessing.speaker_naming import (
    resolve_speakers, drop_placeholder_triples, load_rosters,
)
from extractors.llm_extractor import LLMExtractor, MockLLMExtractor, DEFAULT_MODEL
from graph.neo4j_graph import KnowledgeGraph, FlatFileGraph


def process_meeting(
    transcript: MeetingTranscript,
    extractor,
    resolver: EntityResolver,
    graph,
    window_size: int = 15,
    overlap: int = 5,
    drop_placeholders: bool = True,
    meeting_date: str = "",
) -> List[Triple]:
    """
    Process a single meeting through the full pipeline.
    
    Returns the list of resolved triples that were inserted.
    """
    meeting_id = transcript.meeting_id
    print(f"\n{'='*60}")
    print(f"Processing meeting: {meeting_id}"
          + (f" ({meeting_date})" if meeting_date else ""))
    print(f"  Utterances: {len(transcript.utterances)}")
    print(f"  Duration: {transcript.duration:.1f}s")
    print(f"{'='*60}")

    # Step 1: Chunk into windows
    windows = chunk_transcript(transcript, window_size=window_size, overlap=overlap)
    print(f"\n[1/4] Chunked into {len(windows)} windows "
          f"(size={window_size}, overlap={overlap})")

    # Step 2: Extract triples
    print(f"\n[2/4] Extracting triples...")
    raw_triples = extractor.extract_meeting(windows, meeting_date=meeting_date)
    print(f"  Raw triples extracted: {len(raw_triples)}")

    # Any speaker the naming step could not identify is still a bare
    # diarization label, which is not a real entity.
    if drop_placeholders:
        raw_triples = drop_placeholder_triples(raw_triples)

    if not raw_triples:
        print("  No triples found. Skipping this meeting.")
        return []

    # Step 3: Entity resolution
    print(f"\n[3/4] Resolving entities...")
    resolved_triples = resolver.resolve_triples(raw_triples)
    print(f"  After resolution: {len(resolved_triples)} triples "
          f"(deduped {len(raw_triples) - len(resolved_triples)})")

    # Step 4: Insert into graph
    print(f"\n[4/4] Inserting into knowledge graph...")
    graph.insert_triples(resolved_triples, meeting_id=meeting_id,
                         meeting_date=meeting_date)

    # Print summary
    stats = graph.get_graph_stats()
    print(f"\n  Graph state: {stats.get('total_entities', '?')} entities, "
          f"{stats.get('total_relations', '?')} relations")

    return resolved_triples


def save_triples_json(triples: List[Triple], output_path: str):
    """Save extracted triples to JSON for inspection/evaluation."""
    data = [t.to_dict() for t in triples]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Triples saved to {output_path}")


def build_extractor(args):
    """Build the appropriate extractor based on CLI args."""
    if args.extractor == "mock":
        print("[Extractor] Using MockLLMExtractor (no API needed)")
        return MockLLMExtractor()

    elif args.extractor == "llm":
        print(f"[Extractor] Using LLM: {args.llm_model}")
        print(f"[Extractor] Mode: {'constrained (schema enums)' if args.constrained else 'freeform'}")
        print(f"[Extractor] Pacing: {args.request_delay}s between windows")
        return LLMExtractor(
            model=args.llm_model,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=args.api_base or os.getenv("OPENAI_BASE_URL"),
            constrained=args.constrained,
            request_delay=args.request_delay,
        )

    elif args.extractor == "dhgat":
        from extractors.dhgat_extractor import DHGATExtractor
        print(f"[Extractor] Using DHGAT from {args.dhgat_repo}")
        ext = DHGATExtractor(
            repo_path=args.dhgat_repo,
            ckpt_path=args.dhgat_ckpt,
        )
        if not ext.load():
            print("[FATAL] DHGAT failed to load. Falling back to mock.")
            return MockLLMExtractor()
        return ext

    else:
        raise ValueError(f"Unknown extractor: {args.extractor}")


def build_graph(args):
    """Build the appropriate graph backend."""
    if args.no_neo4j:
        graph = FlatFileGraph(output_dir=args.output_dir)
    else:
        graph = KnowledgeGraph(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )

    if not graph.connect():
        if not args.no_neo4j:
            print("[WARN] Neo4j connection failed. Falling back to flat file graph.")
            graph = FlatFileGraph(output_dir=args.output_dir)
            graph.connect()

    graph.setup_constraints()
    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Continual KG Construction from Meeting Audio"
    )

    # Input
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="Transcript file(s) to process. Supports .txt, .srt, .json"
    )
    parser.add_argument(
        "--meeting-id", nargs="+", default=None,
        help="Meeting ID(s) for the input files. Defaults to filename."
    )

    # Extractor
    parser.add_argument(
        "--extractor", choices=["llm", "dhgat", "mock"], default="llm",
        help="Which triple extractor to use (default: llm)"
    )
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
                        help=f"LLM for extraction (default: {DEFAULT_MODEL}). "
                             f"The API endpoint is derived from the model name.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--request-delay", type=float,
                        default=float(os.getenv("LLM_REQUEST_DELAY", "5.0")),
                        help="Seconds between extraction calls. Default 5.0 suits "
                             "a free tier (~12 RPM). On a paid tier use 0.1-0.5; "
                             "429s are retried with backoff regardless.")

    # Taxonomy constraint. Constrained is the default: freeform extraction
    # produced 4% taxonomy adherence on meeting_001, which is unusable as
    # training labels. --freeform keeps the original path for comparison.
    constraint = parser.add_mutually_exclusive_group()
    constraint.add_argument("--constrained", dest="constrained", action="store_true",
                            default=True,
                            help="Force relations/types to the taxonomy via JSON-schema enums (default)")
    constraint.add_argument("--freeform", dest="constrained", action="store_false",
                            help="Original unconstrained prompt; the model may invent relations")
    parser.add_argument("--dhgat-repo", default="./capstone_dialouge-re")
    parser.add_argument("--dhgat-ckpt", default=None)

    # Graph
    parser.add_argument("--no-neo4j", action="store_true",
                        help="Use flat file graph instead of Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")

    # Speaker naming. Diarization emits anonymous SPEAKER_n labels, which
    # the extractor otherwise turns into PERSON entities — 43% of triples
    # in the first five-meeting run. Resolving them is the default.
    parser.add_argument("--no-resolve-speakers", dest="resolve_speakers",
                        action="store_false", default=True,
                        help="Skip mapping SPEAKER_n to the names spoken in the dialogue")
    parser.add_argument("--keep-placeholders", dest="drop_placeholders",
                        action="store_false", default=True,
                        help="Keep triples anchored to unresolved speaker labels")
    parser.add_argument("--roster", default="data/rosters.json",
                        help="JSON of {meeting_id: [attendee names]}. Constrains "
                             "speaker naming to a closed set, the way a calendar "
                             "invite would. Pass '' to disable.")
    parser.add_argument("--meeting-dates", default="data/meeting_dates.json",
                        help="JSON of {meeting_id: {date: YYYY-MM-DD}}. Gives the "
                             "graph metric time (stated_at) and lets the extractor "
                             "resolve relative deadlines. Pass '' to disable.")

    # Chunking
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--overlap", type=int, default=5)

    # Output
    parser.add_argument("--output-dir", default="./output",
                        help="Directory for output files")
    parser.add_argument("--clear-graph", action="store_true",
                        help="Clear the graph before processing")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Rosters feed two things: the closed set for speaker naming, and the
    # canonical spellings the resolver collapses ASR name variants onto.
    rosters = load_rosters(args.roster) if args.resolve_speakers else {}
    if rosters:
        print(f"[Rosters] Loaded attendee lists for {len(rosters)} meetings "
              f"from {args.roster}")
    all_attendees = sorted({n for names in rosters.values() for n in names})

    # Meeting dates: {meeting_id: "YYYY-MM-DD"}. Missing file or entry is
    # never fatal — the graph just stays ordinal for that meeting.
    meeting_dates = {}
    if args.meeting_dates and os.path.exists(args.meeting_dates):
        try:
            with open(args.meeting_dates, encoding="utf-8") as f:
                raw_dates = json.load(f)
            meeting_dates = {
                k: v["date"] for k, v in raw_dates.items()
                if isinstance(v, dict) and v.get("date")
            }
            print(f"[Dates] Loaded meeting dates for {len(meeting_dates)} meetings "
                  f"from {args.meeting_dates}")
        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"[WARN] Could not read meeting dates: {e}")

    # Build components
    extractor = build_extractor(args)
    resolver = EntityResolver(roster_names=all_attendees)
    graph = build_graph(args)

    if args.clear_graph:
        graph.clear_all()

    # Process each meeting sequentially
    meeting_ids = args.meeting_id or [None] * len(args.input)
    if len(meeting_ids) < len(args.input):
        meeting_ids.extend([None] * (len(args.input) - len(meeting_ids)))

    all_triples = {}

    for filepath, mid in zip(args.input, meeting_ids):
        # Parse transcript
        transcript = parse_transcript(filepath, meeting_id=mid or "")
        if not transcript.utterances:
            print(f"[WARN] No utterances found in {filepath}. Skipping.")
            continue

        # Name the speakers before chunking, so the extractor prompt
        # carries real names rather than diarization labels.
        if args.resolve_speakers:
            print(f"\n[0/4] Resolving speaker names for {transcript.meeting_id}...")
            resolve_speakers(
                transcript,
                model=args.llm_model,
                api_key=args.api_key or os.getenv("OPENAI_API_KEY", ""),
                base_url=args.api_base or os.getenv("OPENAI_BASE_URL", ""),
                roster=rosters.get(transcript.meeting_id),
            )

        # Process through pipeline
        triples = process_meeting(
            transcript=transcript,
            extractor=extractor,
            resolver=resolver,
            graph=graph,
            window_size=args.window_size,
            overlap=args.overlap,
            drop_placeholders=args.drop_placeholders,
            meeting_date=meeting_dates.get(transcript.meeting_id, ""),
        )

        # Save triples for this meeting
        meeting_output = os.path.join(
            args.output_dir,
            f"triples_{transcript.meeting_id}.json"
        )
        save_triples_json(triples, meeting_output)
        all_triples[transcript.meeting_id] = triples

    # Final summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Meetings processed: {len(all_triples)}")
    total = sum(len(t) for t in all_triples.values())
    print(f"Total triples extracted: {total}")
    print(f"Graph stats: {graph.get_graph_stats()}")
    print(f"Entity resolver stats: {resolver.get_stats()}")

    # Extraction coverage. Reported separately from adherence because they
    # answer different questions: adherence is "are the labels legal",
    # coverage is "did we actually look at the whole meeting".
    if hasattr(extractor, "coverage_report"):
        cov = extractor.coverage_report()
        print(f"\nExtraction coverage: {cov['windows_extracted']}/{cov['windows_total']} "
              f"windows ({cov['coverage_pct']}%), "
              f"{cov['windows_failed']} failed, {cov['windows_empty']} yielded no triples")
        if cov["windows_failed"]:
            print(f"  *** {cov['windows_failed']} window(s) were never extracted. "
                  f"That content is missing from the graph. Re-run to recover it.")
        cov_path = os.path.join(args.output_dir, "coverage.json")
        with open(cov_path, "w", encoding="utf-8") as f:
            json.dump(cov, f, indent=2)

    # Taxonomy adherence — the gating metric for using these triples as
    # training labels downstream.
    if hasattr(extractor, "adherence_report"):
        report = extractor.adherence_report()
        print(f"\nTaxonomy adherence [{report['mode']}]: "
              f"{report['adherence_pct']}% "
              f"({report['triples_seen'] - report['off_taxonomy_count']}"
              f"/{report['triples_seen']} relations in taxonomy)")
        if report["off_taxonomy_labels"]:
            print(f"  Off-taxonomy labels: {report['off_taxonomy_labels']}")
        report_path = os.path.join(args.output_dir, f"adherence_{report['mode']}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  Adherence report saved to {report_path}")

    print(f"Output directory: {args.output_dir}")

    graph.close()


if __name__ == "__main__":
    main()
