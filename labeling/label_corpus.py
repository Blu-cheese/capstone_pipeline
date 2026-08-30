"""
Teacher labelling at scale (METHOD_A_SPEC Phase 1).

Gemini classifies the relation for each annotated entity pair in a DialogRE
dialogue. Those teacher labels - NOT the gold labels - are what the continual
student trains on. Gold is retained alongside purely to measure teacher
quality, which is the student's accuracy ceiling.

Design notes:

  * One API call per DIALOGUE, not per pair. Dialogues average 5.6 annotated
    pairs, so this is a ~5.6x reduction against a rate-limited quota with no
    loss of context - the model sees the same dialogue either way.

  * Disk cache is mandatory and keyed on
    sha256(dialogue_text + pair list + model + PROMPT_VERSION).
    Bump PROMPT_VERSION whenever the prompt changes, or you will silently
    reuse labels produced by a different instruction.

  * Resumable: killing and restarting re-requests nothing already cached.

Usage:
    venv/bin/python -m labeling.label_corpus --limit 400
    venv/bin/python -m labeling.label_corpus --limit 400 --dry-run
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.env import load_env
load_env()

from preprocessing.dialogre_parser import (
    load_dialogre, DialogueUnit, DIALOGRE_RELATIONS,
)
from extractors.llm_extractor import default_base_url, DEFAULT_MODEL

# Bump on any prompt change - the cache key depends on it.
PROMPT_VERSION = "v2"

CACHE_DIR = Path("data/cache/teacher")
OUTPUT_DIR = Path("data/labeled")


SYSTEM_PROMPT = """You are a relation classifier for multi-party dialogue.

You are given a dialogue and a list of entity pairs appearing in it. For EACH
pair, decide which single relation holds between argument x and argument y,
reading the relation as "x is the <relation> of y" where applicable.

You MUST choose exactly one label per pair, verbatim, from this list:
{relations}

Work through these checks IN ORDER and take the first that applies:

1. SAME PERSON? If x and y are two ways of referring to the same individual
   - nickname, first name vs full name, surname, or a form of address
   ("Chandler" / "Chandler Bing" / "Mr. Bing") - answer
   `per:alternate_names`. Check this FIRST; it is the single most common
   relation in this corpus and is easy to miss.

2. FAMILY OR ROMANCE? Use the specific kinship or romantic label:
   `per:spouse`, `per:parents` (x is the PARENT of y), `per:children`
   (x is the CHILD of y), `per:siblings`, `per:other_family`,
   `per:girl/boyfriend`, `per:dates`.

3. STRUCTURED TIE? Use `per:friends`, `per:roommate`, `per:neighbor`,
   `per:boss` (x is the BOSS of y), `per:subordinate`, `per:client`,
   `per:employee_or_member_of`, `per:alumni`, `per:schools_attended`,
   `per:works`, `per:title`, `per:age`, `per:pet`, or a place/origin
   relation, whenever the dialogue supports one.

4. EXPRESSED FEELING? If a speaker voices liking, admiration, affection or
   praise toward the other, answer `per:positive_impression`. Dislike,
   irritation, contempt or criticism -> `per:negative_impression`. These are
   COMMON in this corpus - do not skip them just because a social tie also
   exists; use the impression label when the feeling is what the dialogue
   actually conveys.

5. NO EVIDENCE? Answer `unanswerable`.

CRITICAL - `per:acquaintance` is RARE. It appears in well under 1% of cases.
It means only "these two know each other and NONE of the above applies".
NEVER use it because you are unsure or because the relation is ambiguous.
If you are hesitating between `per:acquaintance` and something specific,
choose the specific one; if there is genuinely no evidence, choose
`unanswerable`. Treat any urge to answer `per:acquaintance` as a signal that
you have not worked through checks 1-4 carefully enough.

Direction matters throughout: read every label as "x is the <relation> of y".
Base decisions only on the dialogue, not outside knowledge of the characters.

Return ONLY a JSON object mapping each pair's index to its label:
{{"labels": [{{"index": 0, "relation": "per:friends"}}, ...]}}
Include every index exactly once."""


USER_TEMPLATE = """Dialogue:
---
{dialogue}
---

Entity pairs to classify:
{pairs}

Return the JSON object with one label per index."""


def build_messages(unit: DialogueUnit) -> tuple:
    pair_lines = "\n".join(
        f"  {i}. x={p.x!r} ({p.x_type})  y={p.y!r} ({p.y_type})"
        for i, p in enumerate(unit.pairs)
    )
    return (
        SYSTEM_PROMPT.format(relations=", ".join(DIALOGRE_RELATIONS)),
        USER_TEMPLATE.format(dialogue=unit.text, pairs=pair_lines),
    )


def cache_key(unit: DialogueUnit, model: str) -> str:
    payload = "\n".join([
        unit.text,
        "|".join(f"{p.x}::{p.y}" for p in unit.pairs),
        model,
        PROMPT_VERSION,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(key: str) -> Path:
    # Shard by first two hex chars so no directory holds thousands of files.
    return CACHE_DIR / key[:2] / f"{key}.json"


def read_cache(key: str) -> Optional[dict]:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None  # corrupt entry - re-request rather than crash


def write_cache(key: str, value: dict) -> None:
    p = cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2), encoding="utf-8")


# The label schema. Enums make the teacher physically unable to emit a class
# outside the DialogRE label space, which is what keeps the student's 37-way
# head well-defined.
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relation": {"type": "string", "enum": DIALOGRE_RELATIONS},
                },
                "required": ["index", "relation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def call_teacher(system: str, user: str, model: str, api_key: str,
                 base_url: str, max_retries: int = 5) -> str:
    """Single chat-completion call with 429 backoff."""
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 2000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "relation_labels", "strict": True,
                            "schema": LABEL_SCHEMA},
        },
    }).encode("utf-8")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    [rate limited] waiting {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("max retries exhausted")


def parse_labels(raw: str, n_pairs: int) -> Dict[int, str]:
    """Map pair index -> teacher relation, ignoring malformed entries."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}

    out: Dict[int, str] = {}
    for item in data.get("labels", []):
        try:
            idx = int(item["index"])
            rel = str(item["relation"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < n_pairs and rel in DIALOGRE_RELATIONS:
            out[idx] = rel
    return out


def label_corpus(split: str = "train", limit: Optional[int] = None,
                 model: str = "", sleep: float = 1.0,
                 dry_run: bool = False) -> List[dict]:
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL") or default_base_url(model)

    if not api_key and not dry_run:
        raise SystemExit("OPENAI_API_KEY not set (check .env)")

    units = load_dialogre(f"data/raw/dialogre/{split}.json", limit=limit)

    total_pairs = sum(len(u.pairs) for u in units)
    cached = sum(1 for u in units if read_cache(cache_key(u, model)))
    print(f"\n{len(units)} dialogues / {total_pairs} pairs | "
          f"cache hits {cached}/{len(units)} | to request: {len(units)-cached}")
    print(f"model={model}  prompt={PROMPT_VERSION}\n")

    if dry_run:
        print("[dry-run] no API calls made")
        return []

    rows: List[dict] = []
    hits = misses = failed = 0
    multilabel_dropped = 0
    t0 = time.time()

    for n, unit in enumerate(units, 1):
        key = cache_key(unit, model)
        entry = read_cache(key)

        if entry is not None:
            hits += 1
        else:
            try:
                system, user = build_messages(unit)
                raw = call_teacher(system, user, model, api_key, base_url)
                entry = {"raw": raw, "model": model, "prompt_version": PROMPT_VERSION}
                write_cache(key, entry)
                misses += 1
                if sleep:
                    time.sleep(sleep)
            except Exception as exc:
                failed += 1
                print(f"  [FAIL] {unit.dialogue_id}: {str(exc)[:120]}")
                continue

        labels = parse_labels(entry["raw"], len(unit.pairs))
        for i, pair in enumerate(unit.pairs):
            teacher = labels.get(i)
            if teacher is None:
                continue

            # Multi-label pairs are DROPPED, not collapsed to the first gold
            # relation (V2 global config). "First" is annotation order, not
            # salience. Note the filter is applied HERE, when building rows -
            # not when loading the corpus - so the teacher still sees every
            # pair in context and the cache key is unchanged. Filtering at
            # load time would alter the key and force a full re-request.
            if pair.is_multilabel:
                multilabel_dropped += 1
                continue
            rows.append({
                "dialogue_id": unit.dialogue_id,
                "window_text": unit.text,
                "subject": pair.x, "subject_type": pair.x_type,
                "object": pair.y, "object_type": pair.y_type,
                "relation": teacher,          # teacher label - student trains on this
                "gold_relation": pair.gold,   # human label - evaluation only
                "gold_relations": pair.gold_relations,
                "teacher_correct": teacher in pair.gold_relations,
                "source_meeting": unit.dialogue_id,
            })

        if n % 25 == 0 or n == len(units):
            rate = n / max(time.time() - t0, 1e-6) * 60
            print(f"  {n}/{len(units)} dialogues | {len(rows)} labelled pairs | "
                  f"hits={hits} new={misses} failed={failed} | {rate:.0f} dlg/min")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dialogre_{split}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    _report(rows, out_path, hits, misses, failed, multilabel_dropped)
    return rows


def _report(rows, out_path, hits, misses, failed, multilabel_dropped=0):
    from collections import Counter

    print(f"\n{'='*62}\nLABELLING COMPLETE -> {out_path}")
    print(f"{'='*62}")
    print(f"pairs labelled : {len(rows)}")
    print(f"cache hits     : {hits}   new requests: {misses}   failures: {failed}")
    print(f"multi-label dropped: {multilabel_dropped} (V2 policy: dropped, not collapsed to first)")
    if not rows:
        return

    agree = sum(1 for r in rows if r["teacher_correct"])
    print(f"teacher vs gold: {agree}/{len(rows)} = {100*agree/len(rows):.1f}% "
          f"(this is the student's accuracy ceiling)")

    counts = Counter(r["relation"] for r in rows)
    print(f"\nteacher label distribution ({len(counts)}/{len(DIALOGRE_RELATIONS)} types used):")
    for rel, c in counts.most_common():
        print(f"  {c:5d}  {rel}")

    missing = [r for r in DIALOGRE_RELATIONS if r not in counts]
    if missing:
        print(f"\nabsent from teacher output ({len(missing)}): {missing}")
    thin = [r for r, c in counts.items() if c < 20]
    if thin:
        print(f"\nfewer than 20 examples ({len(thin)}) - Phase 3 must merge or drop: {thin}")


def main():
    ap = argparse.ArgumentParser(description="Teacher-label DialogRE with Gemini")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="max dialogues")
    ap.add_argument("--model", default="")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds between requests (0 for paid tier)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report cache state and exit without calling the API")
    args = ap.parse_args()
    label_corpus(args.split, args.limit, args.model, args.sleep, args.dry_run)


if __name__ == "__main__":
    main()
