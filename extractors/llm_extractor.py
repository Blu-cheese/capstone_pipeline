"""
LLM-based triple extraction.

Takes a ConversationWindow, prompts an LLM to extract
(subject, relation, object) triples with entity types,
returns List[Triple].

Supports any OpenAI-compatible API (OpenAI, Groq, local vLLM, etc).
"""

import json
import re
import os
from typing import List, Optional

from utils.models import ConversationWindow, Triple


# --- API endpoint selection ---
#
# The project runs Gemini through its OpenAI-compatible shim. Previously the
# base URL defaulted to OpenAI's while the model name defaulted to a Gemini
# one, so any run without a hand-exported OPENAI_BASE_URL sent "gemini-*" to
# api.openai.com and failed. Derive the endpoint from the model name instead.

DEFAULT_MODEL = "gemini-2.5-flash-lite"

OPENAI_BASE = "https://api.openai.com/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def default_base_url(model: str) -> str:
    """Pick the OpenAI-compatible endpoint that matches the model family."""
    return GEMINI_BASE if model.lower().startswith("gemini") else OPENAI_BASE


# --- Entity taxonomy for college staff meetings ---
ENTITY_TYPES = [
    "PERSON",       # faculty, students, administrators
    "COURSE",       # course codes, course names
    "DEPARTMENT",   # CSE, AI&ML, Mechanical, etc.
    "COMMITTEE",    # Board of Studies, Exam Committee, etc.
    "PROJECT",      # capstone projects, research projects
    "RESOURCE",     # labs, budgets, infrastructure
    "EVENT",        # exams, workshops, conferences
    "DEADLINE",     # dates, submission deadlines
    "POLICY",       # rules, regulations, curriculum changes
    "TOPIC",        # agenda items, discussion topics
]

RELATION_TYPES = [
    "teaches", "heads", "member_of", "assigned_to",
    "proposed", "approved", "rejected", "postponed",
    "deadline_for", "reports_to", "depends_on", "blocked_by",
    "discussed", "decided_on", "scheduled_for", "budget_for",
    "supervises", "enrolled_in", "part_of", "replaced_by",
]

CONSTRAINED_SYSTEM_PROMPT = """You are a knowledge extraction system for college staff meeting transcripts.

Given a conversation window, extract structured knowledge triples.

ENTITY TYPES - `subject_type` and `object_type` MUST be exactly one of these: {entity_types}

RELATION TYPES - `relation` MUST be exactly one of these, verbatim: {relation_types}

RULES:
0. NEVER invent a relation or entity type outside the lists above. If a
   statement does not fit any listed relation, omit the triple entirely
   rather than inventing a label.
   Use the exact underscore spelling, e.g. `deadline_for`, never `deadline for`.
0a. PREFER THE MOST SPECIFIC RELATION. `discussed` is a last resort, not a
   default: use it only when someone raises a topic with no decision,
   assignment, proposal, schedule, dependency or ownership attached. If a
   statement carries any of those, the specific relation is required.
   Before emitting `discussed`, re-read the list and confirm nothing fits.
0b. A speaker merely mentioning a topic is usually NOT worth a triple.
   Prefer few high-value triples (decisions, assignments, deadlines,
   ownership) over exhaustive coverage of everything said.
1. Extract only factual statements, decisions, and assignments - not opinions or filler.
2. Each triple must have: subject, subject_type, relation, object, object_type.
3. Normalize entity names: use full names for people (not pronouns), official names for courses/departments.
4. If a pronoun refers to a known speaker, resolve it to the speaker's name.
5. Skip utterances that are purely social (greetings, thanks) unless they contain information.
6. Confidence should be 0.0-1.0: 1.0 for explicit statements, 0.7-0.9 for implied, below 0.7 for uncertain.
7. Subjects and objects must be concrete named entities, not sentence
   fragments. Reject things like "three changes" or "twelve weeks to
   fourteen weeks" - if there is no nameable entity, omit the triple.
8. `utterance_time` is the (Ns) start time of the utterance where the fact
   is stated, copied from the transcript. Use the utterance that asserts the
   fact, not one that merely mentions the entity.
9. `deadline_date` is "" unless the object is a DEADLINE whose calendar date
   is determinable. When the meeting date is given, resolve relative phrases
   against it: "next Friday", "month-end", "the 25th" all resolve. Format
   YYYY-MM-DD. Never guess a date the dialogue does not support - "" is
   always acceptable.

Respond with ONLY a JSON object of the form {{"triples": [...]}}. No other text.
Example:
{{"triples": [
  {{
    "subject": "Dr. Jayashree",
    "subject_type": "PERSON",
    "relation": "approved",
    "object": "New AI elective",
    "object_type": "COURSE",
    "confidence": 0.95
  }}
]}}

If no meaningful triples can be extracted, return {{"triples": []}}."""


SYSTEM_PROMPT = """You are a knowledge extraction system for college staff meeting transcripts.

Given a conversation window, extract structured knowledge triples.

ENTITY TYPES: {entity_types}

COMMON RELATION TYPES (you may use others if they fit better): {relation_types}

RULES:
1. Extract only factual statements, decisions, and assignments — not opinions or filler.
2. Each triple must have: subject, subject_type, relation, object, object_type.
3. Normalize entity names: use full names for people (not pronouns), official names for courses/departments.
4. If a pronoun refers to a known speaker, resolve it to the speaker's name.
5. Skip utterances that are purely social (greetings, thanks) unless they contain information.
6. Confidence should be 0.0-1.0: 1.0 for explicit statements, 0.7-0.9 for implied, below 0.7 for uncertain.

Respond with ONLY a JSON array of objects. No other text.
Example:
[
  {{
    "subject": "Dr. Jayashree",
    "subject_type": "PERSON",
    "relation": "approved",
    "object": "New AI elective",
    "object_type": "COURSE",
    "confidence": 0.95
  }}
]

If no meaningful triples can be extracted, return an empty array: []"""


USER_PROMPT_TEMPLATE = """Extract knowledge triples from this meeting segment:

Meeting: {meeting_id}{meeting_date_line}
---
{transcript_text}
---

Extract all entities and relationships. Return JSON only."""


def build_prompt(window: ConversationWindow, constrained: bool = False,
                 meeting_date: str = "") -> tuple:
    """Build system + user prompts for the LLM."""
    template = CONSTRAINED_SYSTEM_PROMPT if constrained else SYSTEM_PROMPT
    system = template.format(
        entity_types=", ".join(ENTITY_TYPES),
        relation_types=", ".join(RELATION_TYPES),
    )
    # The date is what makes "next Friday" resolvable to a calendar day.
    date_line = f"\nMeeting date: {meeting_date}" if meeting_date else ""
    user = USER_PROMPT_TEMPLATE.format(
        meeting_id=window.meeting_id,
        meeting_date_line=date_line,
        transcript_text=window.text,
    )
    return system, user


# JSON schema handed to the API in constrained mode. The enums are the whole
# point: they move taxonomy adherence from a prompt-level request the model
# may ignore to a decoding-level constraint. Wrapped in an object because
# several OpenAI-compatible endpoints reject a bare array at the root.
TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "subject_type": {"type": "string", "enum": ENTITY_TYPES},
                    "relation": {"type": "string", "enum": RELATION_TYPES},
                    "object": {"type": "string"},
                    "object_type": {"type": "string", "enum": ENTITY_TYPES},
                    "confidence": {"type": "number"},
                    # Start time (seconds) of the utterance that states the
                    # fact — the (Ns) values are right there in the window
                    # text. Without this every triple inherited the window's
                    # first timestamp, collapsing time to ~40s buckets.
                    "utterance_time": {"type": "number"},
                    # YYYY-MM-DD when the object is a DEADLINE and the
                    # dialogue pins it down ("next Friday" + meeting date),
                    # else "". Deadlines otherwise exist only as strings the
                    # graph cannot order.
                    "deadline_date": {"type": "string"},
                },
                "required": [
                    "subject", "subject_type", "relation",
                    "object", "object_type", "confidence",
                    "utterance_time", "deadline_date",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["triples"],
    "additionalProperties": False,
}


def normalize_relation(relation: str) -> str:
    """
    Canonicalise a relation label to the taxonomy's snake_case spelling.

    The freeform extractor emitted space-separated near-duplicates of valid
    types ("deadline for" alongside "deadline_for"), which inflate the
    apparent off-taxonomy rate without being genuinely new relations.
    """
    return "_".join(relation.strip().lower().split())


def parse_llm_response(
    response_text: str,
    window: ConversationWindow,
    constrained: bool = False,
    stats: Optional[dict] = None,
) -> List[Triple]:
    """
    Parse LLM JSON response into Triple objects.
    Handles common LLM quirks: markdown fences, trailing commas, etc.

    Accepts either a bare array (freeform mode) or {"triples": [...]}
    (constrained mode). In constrained mode, relations are canonicalised and
    anything still outside the taxonomy is dropped, since the whole point of
    that mode is that downstream training labels stay inside the 20 classes.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = _extract_json_payload(text)
    if data is None:
        print(f"  [WARN] Failed to parse LLM response for window {window.window_id}")
        return []

    # Constrained mode returns {"triples": [...]}; freeform returns a array.
    if isinstance(data, dict):
        data = data.get("triples", [])
    if not isinstance(data, list):
        return []

    triples = []
    first_time = window.utterances[0].start_time if window.utterances else 0.0
    last_time = window.utterances[-1].end_time if window.utterances else 0.0

    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            relation = normalize_relation(str(item.get("relation", "")))

            # Per-utterance attribution, trusted only within window bounds.
            # A cited time outside the window is a hallucinated citation, so
            # it falls back to the window start (the old behaviour).
            try:
                cited = float(item.get("utterance_time", first_time))
            except (TypeError, ValueError):
                cited = first_time
            if not (first_time <= cited <= last_time + 1.0):
                cited = first_time

            # Deadline resolution: accept only a well-formed ISO date.
            deadline_date = str(item.get("deadline_date", "") or "").strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", deadline_date):
                deadline_date = ""

            if stats is not None:
                stats["seen"] = stats.get("seen", 0) + 1
                if relation not in RELATION_TYPES:
                    stats.setdefault("off_taxonomy", []).append(relation)

            # In constrained mode the enum should make this unreachable. If a
            # label still slips through, the endpoint is not honouring the
            # schema - drop it rather than let it into training labels.
            if constrained and relation not in RELATION_TYPES:
                continue

            triples.append(Triple(
                subject=str(item.get("subject", "")).strip(),
                subject_type=str(item.get("subject_type", "UNKNOWN")).strip().upper(),
                relation=relation,
                object=str(item.get("object", "")).strip(),
                object_type=str(item.get("object_type", "UNKNOWN")).strip().upper(),
                confidence=float(item.get("confidence", 0.8)),
                source_meeting=window.meeting_id,
                timestamp=cited,
                deadline_date=deadline_date,
                source_utterance=window.text[:200],
                window_text=window.text,
            ))
        except (ValueError, KeyError) as e:
            continue

    # Filter out empty triples
    triples = [t for t in triples if t.subject and t.object and t.relation]
    return triples


def _extract_json_payload(text: str):
    """
    Pull the JSON object or array out of a model response, tolerating
    surrounding prose and trailing commas. Returns None if nothing parses.
    """
    import re

    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append((start, text[start:end + 1]))

    # Prefer whichever structure appears first in the response.
    for _, snippet in sorted(candidates):
        for attempt in (snippet, re.sub(r',\s*([}\]])', r'\1', snippet)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


class LLMExtractor:
    """
    Extract triples using an OpenAI-compatible API.
    
    Works with: OpenAI, Groq, Together, local vLLM/Ollama, etc.
    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL in env.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        constrained: bool = True,
        request_delay: float = 5.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or default_base_url(model)
        self.temperature = temperature
        self.constrained = constrained
        # Seconds between extraction calls. The 5s default is sized for a
        # free tier (~12 RPM); on a paid tier this is the single biggest
        # component of wall-clock time and should be dropped accordingly.
        # _call_api still backs off on 429, so a too-low value degrades to
        # retries rather than failures.
        self.request_delay = request_delay
        # Extraction coverage. Taxonomy adherence divides parsed triples by
        # parsed triples, so a window that fails contributes to neither and
        # the run still reports 100%. Two windows lost to HTTP 503 went
        # unnoticed that way once; these counters make the loss visible.
        self.coverage = {"windows": 0, "failed": 0, "empty": 0}
        # Adherence bookkeeping, reported by the extractor at end of run.
        self.stats = {"seen": 0, "off_taxonomy": []}

        if not self.api_key:
            print("[WARN] No API key set. Set OPENAI_API_KEY or pass api_key=...")

    def extract(self, window: ConversationWindow) -> List[Triple]:
        """Extract triples from a single conversation window."""
        system_prompt, user_prompt = build_prompt(
            window, constrained=self.constrained,
            meeting_date=getattr(self, "meeting_date", ""),
        )

        try:
            response_text = self._call_api(system_prompt, user_prompt)
            return parse_llm_response(
                response_text, window,
                constrained=self.constrained,
                stats=self.stats,
            )
        except Exception as e:
            print(f"  [ERROR] LLM extraction failed for window {window.window_id}: {e}")
            self.coverage["failed"] += 1
            return []

    def adherence_report(self) -> dict:
        """Taxonomy adherence for the run so far."""
        seen = self.stats["seen"]
        off = self.stats["off_taxonomy"]
        return {
            "mode": "constrained" if self.constrained else "freeform",
            "model": self.model,
            "triples_seen": seen,
            "off_taxonomy_count": len(off),
            "adherence_pct": round(100.0 * (seen - len(off)) / seen, 1) if seen else 0.0,
            "off_taxonomy_labels": sorted(set(off)),
        }

    def extract_meeting(self, windows: List[ConversationWindow],
                        meeting_date: str = "") -> List[Triple]:
        """Extract triples from all windows of a meeting."""
        import time
        self.meeting_date = meeting_date
        all_triples = []
        for i, window in enumerate(windows):
            print(f"  Extracting window {i+1}/{len(windows)}...")
            # Pace requests to stay under the account's rate limit. See
            # request_delay in __init__ for how to size this.
            if i > 0 and self.request_delay > 0:
                time.sleep(self.request_delay)
            failed_before = self.coverage["failed"]
            triples = self.extract(window)
            all_triples.extend(triples)
            self.coverage["windows"] += 1
            if not triples and self.coverage["failed"] == failed_before:
                # Produced nothing, but the call itself succeeded — a window
                # of pure agreement or small talk. Distinct from a failure.
                self.coverage["empty"] += 1
            print(f"    Found {len(triples)} triples")
        return all_triples

    def coverage_report(self) -> dict:
        """
        Extraction completeness, which adherence cannot measure.

        `failed` is the number that matters: those windows were never seen
        by the model, so their content is missing from the graph entirely.
        """
        c = self.coverage
        seen = c["windows"] - c["failed"]
        return {
            "windows_total": c["windows"],
            "windows_extracted": seen,
            "windows_failed": c["failed"],
            "windows_empty": c["empty"],
            "coverage_pct": round(100.0 * seen / c["windows"], 1) if c["windows"] else 0.0,
        }

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call OpenAI-compatible chat completion API with retry logic.
        Handles 429 (rate limit) errors via exponential backoff.
        Uses urllib so we don't add requests/openai as dependencies.
        """
        import urllib.request
        import urllib.error
        import ssl
        import time

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }
        if self.constrained:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "meeting_triples",
                    "strict": True,
                    "schema": TRIPLE_SCHEMA,
                },
            }
        payload = json.dumps(body).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Allow self-signed certs for local APIs
        ctx = ssl.create_default_context()
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

        # Retry logic: 5 attempts with exponential backoff for rate limits
        max_retries = 5
        base_wait = 5  # seconds

        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                # 429 = rate limit, 5xx = transient server trouble. Both are
                # worth retrying: a 503 burst once cost meeting_002 five of
                # its eleven windows because only 429 retried here (the
                # query layer already retried 5xx — the two transports had
                # drifted). 4xx other than 429 will never succeed; raise.
                if e.code == 429 or e.code >= 500:
                    wait = base_wait * (2 ** attempt)  # 5, 10, 20, 40, 80 seconds
                    if attempt < max_retries - 1:
                        print(f"    [HTTP {e.code}] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(
                        f"HTTP {e.code} after {max_retries} retries. "
                        f"Server or quota trouble that outlasted the backoff."
                    )
                # Client error — retrying cannot help
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {e.code}: {e.reason}. {error_body}")
            except urllib.error.URLError as e:
                # Network error — retry once
                if attempt < max_retries - 1:
                    print(f"    [Network error] {e}. Retrying in {base_wait}s...")
                    time.sleep(base_wait)
                    continue
                raise

        raise RuntimeError("Max retries exhausted")


# --- Offline/mock extractor for testing without an API ---

class MockLLMExtractor:
    """
    For testing the pipeline without an API key.
    Returns hardcoded triples that exercise the downstream components.
    """

    def extract(self, window: ConversationWindow) -> List[Triple]:
        # Generate plausible triples from the speaker IDs
        triples = []
        speakers = list(window.speaker_ids)
        if len(speakers) >= 2:
            triples.append(Triple(
                subject=speakers[0],
                subject_type="PERSON",
                relation="discussed_with",
                object=speakers[1],
                object_type="PERSON",
                confidence=0.9,
                source_meeting=window.meeting_id,
                timestamp=window.utterances[0].start_time if window.utterances else 0,
            ))
        return triples

    def extract_meeting(self, windows: List[ConversationWindow],
                        meeting_date: str = "") -> List[Triple]:
        all_triples = []
        for window in windows:
            all_triples.extend(self.extract(window))
        return all_triples
