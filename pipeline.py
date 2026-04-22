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

from utils.models import Triple, MeetingTranscript
from preprocessing.transcript_parser import parse_transcript
from preprocessing.chunking import chunk_transcript
from preprocessing.entity_resolution import EntityResolver
from extractors.llm_extractor import LLMExtractor, MockLLMExtractor
from graph.neo4j_graph import KnowledgeGraph, FlatFileGraph


def process_meeting(
    transcript: MeetingTranscript,
    extractor,
    resolver: EntityResolver,
    graph,
    window_size: int = 15,
    overlap: int = 5,
) -> List[Triple]:
    """
    Process a single meeting through the full pipeline.
    
    Returns the list of resolved triples that were inserted.
    """
    meeting_id = transcript.meeting_id
    print(f"\n{'='*60}")
    print(f"Processing meeting: {meeting_id}")
    print(f"  Utterances: {len(transcript.utterances)}")
    print(f"  Duration: {transcript.duration:.1f}s")
    print(f"{'='*60}")

    # Step 1: Chunk into windows
    windows = chunk_transcript(transcript, window_size=window_size, overlap=overlap)
    print(f"\n[1/4] Chunked into {len(windows)} windows "
          f"(size={window_size}, overlap={overlap})")

    # Step 2: Extract triples
    print(f"\n[2/4] Extracting triples...")
    raw_triples = extractor.extract_meeting(windows)
    print(f"  Raw triples extracted: {len(raw_triples)}")

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
    graph.insert_triples(resolved_triples, meeting_id=meeting_id)

    # Print summary
    stats = graph.get_graph_stats()
    print(f"\n  Graph state: {stats.get('total_entities', '?')} entities, "
          f"{stats.get('total_relations', '?')} relations")

    return resolved_triples


def save_triples_json(triples: List[Triple], output_path: str):
    """Save extracted triples to JSON for inspection/evaluation."""
    data = [t.to_dict() for t in triples]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Triples saved to {output_path}")


def build_extractor(args):
    """Build the appropriate extractor based on CLI args."""
    if args.extractor == "mock":
        print("[Extractor] Using MockLLMExtractor (no API needed)")
        return MockLLMExtractor()

    elif args.extractor == "llm":
        print(f"[Extractor] Using LLM: {args.llm_model}")
        return LLMExtractor(
            model=args.llm_model,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=args.api_base or os.getenv("OPENAI_BASE_URL"),
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
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--dhgat-repo", default="./capstone_dialouge-re")
    parser.add_argument("--dhgat-ckpt", default=None)

    # Graph
    parser.add_argument("--no-neo4j", action="store_true",
                        help="Use flat file graph instead of Neo4j")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")

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

    # Build components
    extractor = build_extractor(args)
    resolver = EntityResolver()
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

        # Process through pipeline
        triples = process_meeting(
            transcript=transcript,
            extractor=extractor,
            resolver=resolver,
            graph=graph,
            window_size=args.window_size,
            overlap=args.overlap,
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
    print(f"Output directory: {args.output_dir}")

    graph.close()


if __name__ == "__main__":
    main()
