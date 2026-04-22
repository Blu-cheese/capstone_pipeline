"""
LLM-based triple extraction.

Takes a ConversationWindow, prompts an LLM to extract
(subject, relation, object) triples with entity types,
returns List[Triple].

Supports any OpenAI-compatible API (OpenAI, Groq, local vLLM, etc).
"""

import json
import os
from typing import List, Optional

from utils.models import ConversationWindow, Triple


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

Meeting: {meeting_id}
---
{transcript_text}
---

Extract all entities and relationships. Return JSON only."""


def build_prompt(window: ConversationWindow) -> tuple:
    """Build system + user prompts for the LLM."""
    system = SYSTEM_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relation_types=", ".join(RELATION_TYPES),
    )
    user = USER_PROMPT_TEMPLATE.format(
        meeting_id=window.meeting_id,
        transcript_text=window.text,
    )
    return system, user


def parse_llm_response(response_text: str, window: ConversationWindow) -> List[Triple]:
    """
    Parse LLM JSON response into Triple objects.
    Handles common LLM quirks: markdown fences, trailing commas, etc.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []

    json_str = text[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing trailing commas
        import re
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"  [WARN] Failed to parse LLM response for window {window.window_id}")
            return []

    if not isinstance(data, list):
        return []

    triples = []
    first_time = window.utterances[0].start_time if window.utterances else 0.0

    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            triples.append(Triple(
                subject=str(item.get("subject", "")).strip(),
                subject_type=str(item.get("subject_type", "UNKNOWN")).strip().upper(),
                relation=str(item.get("relation", "")).strip().lower(),
                object=str(item.get("object", "")).strip(),
                object_type=str(item.get("object_type", "UNKNOWN")).strip().upper(),
                confidence=float(item.get("confidence", 0.8)),
                source_meeting=window.meeting_id,
                timestamp=first_time,
                source_utterance=window.text[:200],
            ))
        except (ValueError, KeyError) as e:
            continue

    # Filter out empty triples
    triples = [t for t in triples if t.subject and t.object and t.relation]
    return triples


class LLMExtractor:
    """
    Extract triples using an OpenAI-compatible API.
    
    Works with: OpenAI, Groq, Together, local vLLM/Ollama, etc.
    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL in env.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature

        if not self.api_key:
            print("[WARN] No API key set. Set OPENAI_API_KEY or pass api_key=...")

    def extract(self, window: ConversationWindow) -> List[Triple]:
        """Extract triples from a single conversation window."""
        system_prompt, user_prompt = build_prompt(window)

        try:
            response_text = self._call_api(system_prompt, user_prompt)
            triples = parse_llm_response(response_text, window)
            return triples
        except Exception as e:
            print(f"  [ERROR] LLM extraction failed for window {window.window_id}: {e}")
            return []

    def extract_meeting(self, windows: List[ConversationWindow]) -> List[Triple]:
        """Extract triples from all windows of a meeting."""
        import time
        all_triples = []
        for i, window in enumerate(windows):
            print(f"  Extracting window {i+1}/{len(windows)}...")
            # Pace requests to stay under free-tier rate limits (e.g. 10-15 RPM)
            # 5 seconds between requests = max 12 RPM, safe for all free tiers
            if i > 0:
                time.sleep(5)
            triples = self.extract(window)
            all_triples.extend(triples)
            print(f"    Found {len(triples)} triples")
        return all_triples

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
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }).encode("utf-8")

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
                if e.code == 429:
                    # Rate limit — back off and retry
                    wait = base_wait * (2 ** attempt)  # 5, 10, 20, 40, 80 seconds
                    if attempt < max_retries - 1:
                        print(f"    [Rate limited] Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                        time.sleep(wait)
                        continue
                    else:
                        raise RuntimeError(
                            f"Rate limited after {max_retries} retries. "
                            f"Free tier Gemini allows ~10-15 RPM. "
                            f"Wait 1 minute and try again, or switch model."
                        )
                else:
                    # Other HTTP error — don't retry
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

    def extract_meeting(self, windows: List[ConversationWindow]) -> List[Triple]:
        all_triples = []
        for window in windows:
            all_triples.extend(self.extract(window))
        return all_triples
