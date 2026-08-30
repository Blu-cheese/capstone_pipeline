"""
Map diarization speaker IDs to the real names spoken in the dialogue.

whisper-diarization emits anonymous labels (`SPEAKER_0`, `SPEAKER_1`, ...).
`ConversationWindow.text` renders those straight into the extractor prompt,
so the LLM extracts them as PERSON entities: in the first five-meeting run,
170 of 385 triples (43%) had `SPEAKER_0` as subject or object, making a
placeholder the highest-degree node in the graph.

The real names are present, but only inside the dialogue ("Ramesh, start
with the syllabus submission"). One LLM call per meeting reads the
transcript and returns the ID -> name mapping. Names are rewritten before
chunking, so the extractor never sees a placeholder.

Speakers that cannot be identified keep their original ID. Callers decide
what to do with those; see `drop_placeholder_triples`.
"""

import json
import os
import re
import ssl
import urllib.request
from typing import Dict, List, Optional

from utils.models import MeetingTranscript, Triple
from extractors.llm_extractor import DEFAULT_MODEL, default_base_url


PLACEHOLDER = re.compile(r'^SPEAKER[_ ]?\d+$', re.IGNORECASE)

SYSTEM_PROMPT = """You identify who each anonymous speaker in a meeting transcript is.

The transcript labels speakers as SPEAKER_0, SPEAKER_1, and so on. The real
names are never attached to the labels, but they are spoken in the dialogue —
people address each other by name, introduce each other, and refer to who is
presenting.

Return a JSON object mapping each speaker label to a name:

  {"SPEAKER_0": "Jayashree", "SPEAKER_1": "Ramesh"}

RULES:
1. Use ONLY names actually spoken in the transcript. Never invent one.
2. Use the bare first name as spoken, without titles. "Jayashree", not
   "Dr. Jayashree" or "Jayashree ma'am".
3. Evidence for a label is what that speaker is CALLED by others, or what
   they say about themselves — not merely names they mention.
4. If you cannot identify a speaker with reasonable confidence, map it to
   null. A wrong name is far worse than an unresolved one.
5. Include every speaker label that appears. Return only the JSON object."""


# Closed-set variant. A real deployment knows its attendees from the calendar
# invite, so constraining names to a roster is realistic, not a shortcut — and
# it is the same move that took taxonomy adherence from 29% to 100%: shrink the
# label space to the set of legal answers. The roster does NOT say which label
# is whom; that still has to be inferred from the dialogue.
ROSTER_SYSTEM_PROMPT = """You identify who each anonymous speaker in a meeting transcript is.

The transcript labels speakers as SPEAKER_0, SPEAKER_1, and so on. You are
given the list of people attending this meeting. Assign each speaker label to
one of those people.

Return a JSON object mapping each speaker label to an attendee name:

  {"SPEAKER_0": "Jayashree", "SPEAKER_1": "Ramesh"}

RULES:
1. Use ONLY names from the attendee list, spelled exactly as given. Never use
   a name that is not on the list, even if it is spoken in the transcript.
2. Never assign the same person to two different speaker labels.
3. Evidence for a label is what that speaker is CALLED by others, how others
   respond to them, and what they say about themselves — not merely names
   they mention. Someone who says "Pavan, start with the lab" is not Pavan.
4. There may be fewer speaker labels than attendees; not everyone speaks, and
   diarization sometimes merges two people into one label. Assign only what
   the evidence supports.
5. If you cannot identify a speaker with reasonable confidence, map it to
   null. A wrong name is far worse than an unresolved one.
6. Include every speaker label that appears. Return only the JSON object."""


def _call_api(system_prompt: str, user_prompt: str, model: str = "",
              api_key: str = "", base_url: str = "", timeout: int = 60) -> str:
    """Minimal OpenAI-compatible call. Mirrors the extractor's transport."""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or default_base_url(model)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    ctx = ssl.create_default_context()
    if "localhost" in base_url or "127.0.0.1" in base_url:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def load_rosters(path: str) -> Dict[str, List[str]]:
    """
    Load {meeting_id: [attendee names]} from a JSON file.

    Missing or malformed files are not fatal — the caller falls back to
    open-ended inference.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Could not read roster file {path}: {e}")
        return {}

    rosters = {}
    for meeting_id, names in data.items():
        if isinstance(names, list):
            clean = [str(n).strip() for n in names if str(n).strip()]
            if clean:
                rosters[meeting_id] = clean
    return rosters


def infer_speaker_names(
    transcript: MeetingTranscript,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    roster: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Ask the LLM which real name belongs to each speaker label.

    If `roster` is given, names are constrained to that closed set and no
    two labels may share a person. Returns {speaker_id: name} containing
    only confidently-resolved speakers. Unresolvable ones are omitted, and
    on any API or parse failure this returns {} so the caller proceeds
    with raw IDs rather than crashing a pipeline run.
    """
    labels = sorted({u.speaker_id for u in transcript.utterances})
    if not labels:
        return {}

    lines = [f"[{u.speaker_id}] {u.text}" for u in transcript.utterances]
    if roster:
        user_prompt = (
            f"Attendees of this meeting: {', '.join(roster)}\n"
            f"Speaker labels in this transcript: {', '.join(labels)}\n\n"
            f"Transcript:\n" + "\n".join(lines)
        )
        system = ROSTER_SYSTEM_PROMPT
    else:
        user_prompt = (
            f"Speaker labels in this transcript: {', '.join(labels)}\n\n"
            f"Transcript:\n" + "\n".join(lines)
        )
        system = SYSTEM_PROMPT

    try:
        raw = _call_api(system, user_prompt, model=model,
                        api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"  [WARN] Speaker-name inference failed ({e}). Keeping raw IDs.")
        return {}

    # Strip markdown fences if the model adds them despite json_object mode.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Speaker-name response was not JSON. Keeping raw IDs.")
        return {}

    # Roster lookup is case-insensitive but the roster's spelling wins, so
    # the graph gets one canonical node per person.
    roster_lookup = {n.lower(): n for n in (roster or [])}

    mapping = {}
    for label, name in parsed.items():
        if label not in labels:
            continue                      # hallucinated label
        if not name or not str(name).strip():
            continue                      # explicit null = unidentified
        name = str(name).strip()
        if PLACEHOLDER.match(name):
            continue                      # refused to resolve; don't rename
        if roster:
            canonical = roster_lookup.get(name.lower())
            if canonical is None:
                print(f"  [WARN] '{name}' is not on the roster; leaving {label} unresolved")
                continue
            name = canonical
        mapping[label] = name

    # No two labels may be the same person. When the model doubles up, keep
    # the label with more utterances — it has more evidence behind it — and
    # leave the other unresolved rather than guessing.
    utt_counts = {}
    for u in transcript.utterances:
        utt_counts[u.speaker_id] = utt_counts.get(u.speaker_id, 0) + 1

    by_name: Dict[str, List[str]] = {}
    for label, name in mapping.items():
        by_name.setdefault(name, []).append(label)

    for name, dupes in by_name.items():
        if len(dupes) < 2:
            continue
        winner = max(dupes, key=lambda l: utt_counts.get(l, 0))
        for loser in dupes:
            if loser != winner:
                print(f"  [WARN] '{name}' was assigned to both {winner} and "
                      f"{loser}; keeping {winner} ({utt_counts.get(winner, 0)} "
                      f"utterances), leaving {loser} unresolved")
                mapping.pop(loser, None)

    # Closed-set completion. When the label count matches the roster exactly
    # and only one of each is left over, the assignment is forced — there is
    # no other person it could be. This is deduction, not guessing, and it
    # recovers the case where the model returns "null" or reaches for a name
    # from another meeting's cast. Anything less determined is left alone.
    if roster and len(labels) == len(roster):
        unassigned_labels = [l for l in labels if l not in mapping]
        unassigned_names = [n for n in roster if n not in mapping.values()]
        if len(unassigned_labels) == 1 and len(unassigned_names) == 1:
            label, name = unassigned_labels[0], unassigned_names[0]
            mapping[label] = name
            print(f"  Assigned {label} -> {name} by elimination "
                  f"(only remaining attendee for the only remaining label)")

    return mapping


def apply_speaker_names(transcript: MeetingTranscript,
                        mapping: Dict[str, str]) -> int:
    """Rewrite speaker_id in place. Returns the number of utterances renamed."""
    if not mapping:
        return 0
    renamed = 0
    for u in transcript.utterances:
        if u.speaker_id in mapping:
            u.speaker_id = mapping[u.speaker_id]
            renamed += 1
    return renamed


def resolve_speakers(
    transcript: MeetingTranscript,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    roster: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Infer names and apply them. Returns the mapping used."""
    labels = sorted({u.speaker_id for u in transcript.utterances})

    # Fewer labels than attendees means diarization merged two people into
    # one, and every utterance of the merged pair will carry a single name.
    # Surface it — it is a limitation of the transcript, not of the naming.
    if roster:
        print(f"  Roster ({len(roster)}): {', '.join(roster)}")
        if len(labels) < len(roster):
            print(f"  [WARN] {len(labels)} speaker labels for {len(roster)} "
                  f"attendees — diarization merged speakers, so attribution "
                  f"for the merged label covers more than one person")

    mapping = infer_speaker_names(transcript, model=model, api_key=api_key,
                                  base_url=base_url, roster=roster)
    renamed = apply_speaker_names(transcript, mapping)

    resolved = [f"{k} -> {v}" for k, v in sorted(mapping.items())]
    unresolved = [l for l in labels if l not in mapping]
    print(f"  Speakers resolved: {len(mapping)}/{len(labels)}"
          f" ({renamed} utterances renamed)")
    for r in resolved:
        print(f"    {r}")
    if unresolved:
        print(f"    unresolved (kept as-is): {', '.join(unresolved)}")

    return mapping


def drop_placeholder_triples(triples: List[Triple]) -> List[Triple]:
    """
    Remove triples still anchored to an unresolved speaker label.

    A triple like `SPEAKER_2 discussed budget` carries no usable
    information into the graph — the subject is not a real entity. Kept
    separate from resolution so the two decisions stay independent.
    """
    kept = [t for t in triples
            if not PLACEHOLDER.match(str(t.subject).strip())
            and not PLACEHOLDER.match(str(t.object).strip())]
    dropped = len(triples) - len(kept)
    if dropped:
        print(f"  Dropped {dropped} triples still anchored to an "
              f"unresolved speaker label")
    return kept
