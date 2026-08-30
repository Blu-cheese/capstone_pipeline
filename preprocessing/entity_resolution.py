"""
Entity resolution: normalize entity names and merge duplicates.

Handles:
- Title stripping (Dr., Prof., Mr., Mrs.)
- Case normalization
- Fuzzy string matching (Levenshtein)
- Manual alias map for known entities in your domain
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from utils.models import Triple


# Titles to strip when normalizing person names
TITLES = re.compile(
    r'^(dr\.?|prof\.?|professor|mr\.?|mrs\.?|ms\.?|sir|madam)\s+',
    re.IGNORECASE
)

# Common abbreviations in college settings
DEFAULT_ALIASES = {
    # Add your institution-specific aliases here
    "cse": "Computer Science and Engineering",
    "ai&ml": "Artificial Intelligence and Machine Learning",
    "cse(ai&ml)": "CSE AI and ML",
    "ece": "Electronics and Communication Engineering",
    "mech": "Mechanical Engineering",
    "bos": "Board of Studies",
    "hod": "Head of Department",
    # Example person aliases — fill these from your actual transcripts
    # "jayashree ma'am": "Dr. Jayashree R",
    # "jayashree": "Dr. Jayashree R",
}


def normalize_entity(name: str, entity_type: str = "") -> str:
    """Basic normalization: strip titles, fix case, trim whitespace."""
    name = name.strip()

    if entity_type == "PERSON":
        # Strip titles for matching but we'll keep canonical form
        name = TITLES.sub("", name).strip()

    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)

    return name


def soundex(name: str) -> str:
    """
    Classic Soundex code for a single word.

    ASR errors on names are phonetic, not orthographic: whisper hears
    "Jaishree" for "Jayashree" and "Mira" for "Meera". Those sit at 0.78
    and 0.60 Levenshtein — below any threshold that is safe for general
    entity matching — but they are the same sound. Used only to match
    PERSON mentions against a known roster, where the candidate set is
    four names and a false merge is therefore very unlikely.
    """
    name = re.sub(r'[^a-z]', '', name.lower())
    if not name:
        return ""

    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), **dict.fromkeys("l", "4"),
             **dict.fromkeys("mn", "5"), **dict.fromkeys("r", "6")}

    first = name[0].upper()
    encoded = []
    prev = codes.get(name[0], "")
    for ch in name[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            encoded.append(code)
        # h and w are transparent: they do not reset the previous code
        if ch not in "hw":
            prev = code

    return (first + "".join(encoded) + "000")[:4]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute Levenshtein similarity ratio between two strings.
    Returns 0.0 to 1.0 (1.0 = identical).
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    # Quick length check — very different lengths can't be similar
    if abs(len1 - len2) / max(len1, len2) > 0.5:
        return 0.0

    # Standard DP Levenshtein
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost,
            )

    distance = matrix[len1][len2]
    return 1.0 - distance / max(len1, len2)


class EntityResolver:
    """
    Resolve entity mentions to canonical forms.
    
    Priority:
    1. Exact match in alias map
    2. Normalized exact match against known entities
    3. Fuzzy match above threshold against known entities
    4. If no match, register as new entity
    """

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        fuzzy_threshold: float = 0.85,
        roster_names: Optional[List[str]] = None,
    ):
        self.aliases: Dict[str, str] = {}  # lowercase -> canonical
        self.canonical_entities: Dict[str, str] = {}  # normalized -> canonical
        self.entity_types: Dict[str, str] = {}  # canonical -> type
        self.fuzzy_threshold = fuzzy_threshold

        # Known attendees across all meetings, indexed by sound. Lets an ASR
        # variant of a name collapse onto the roster spelling so one person
        # is one node.
        self.roster_by_sound: Dict[str, str] = {}
        for n in (roster_names or []):
            code = soundex(n)
            if code:
                self.roster_by_sound.setdefault(code, n)

        # Load default aliases
        for alias, canonical in DEFAULT_ALIASES.items():
            self.add_alias(alias, canonical)

        # Load custom aliases
        if aliases:
            for alias, canonical in aliases.items():
                self.add_alias(alias, canonical)

    def add_alias(self, alias: str, canonical: str):
        """Register an alias -> canonical mapping."""
        self.aliases[alias.lower().strip()] = canonical
        self.canonical_entities[canonical.lower().strip()] = canonical

    def resolve(self, name: str, entity_type: str = "") -> str:
        """
        Resolve an entity mention to its canonical form.
        Returns the canonical name.
        """
        original = name
        normalized = normalize_entity(name, entity_type).lower()

        # 1. Check alias map
        if normalized in self.aliases:
            canonical = self.aliases[normalized]
            self.entity_types[canonical] = entity_type or self.entity_types.get(canonical, "")
            return canonical

        # 1b. A PERSON who sounds like someone on the roster IS that person.
        # Single-token names only: "Ramesh" should collapse onto the roster,
        # but "Ramesh Kumar Committee" should not.
        if entity_type == "PERSON" and self.roster_by_sound and " " not in normalized:
            canonical = self.roster_by_sound.get(soundex(normalized))
            if canonical and canonical.lower() != normalized:
                self.aliases[normalized] = canonical
                self.entity_types[canonical] = "PERSON"
                return canonical

        # 2. Check exact match against known canonical entities
        if normalized in self.canonical_entities:
            canonical = self.canonical_entities[normalized]
            self.entity_types[canonical] = entity_type or self.entity_types.get(canonical, "")
            return canonical

        # 3. Fuzzy match against known entities
        best_match = None
        best_score = 0.0
        for known_norm, known_canonical in self.canonical_entities.items():
            score = levenshtein_ratio(normalized, known_norm)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = known_canonical

        if best_match:
            # Register this as a new alias for future lookups
            self.aliases[normalized] = best_match
            return best_match

        # 4. New entity — register it
        # Use the original casing as canonical form
        canonical = name.strip()
        self.canonical_entities[normalized] = canonical
        self.entity_types[canonical] = entity_type
        return canonical

    def resolve_triples(self, triples: List[Triple]) -> List[Triple]:
        """
        Resolve all entity mentions in a list of triples.
        Also deduplicates identical triples.
        """
        resolved = []
        seen = set()

        for t in triples:
            new_subject = self.resolve(t.subject, t.subject_type)
            new_object = self.resolve(t.object, t.object_type)

            # Deduplicate: same subject-relation-object
            dedup_key = (new_subject.lower(), t.relation.lower(), new_object.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            resolved.append(Triple(
                subject=new_subject,
                subject_type=t.subject_type,
                relation=t.relation,
                object=new_object,
                object_type=t.object_type,
                confidence=t.confidence,
                source_meeting=t.source_meeting,
                timestamp=t.timestamp,
                source_utterance=t.source_utterance,
            ))

        return resolved

    def get_stats(self) -> Dict:
        """Return resolver statistics."""
        return {
            "num_canonical_entities": len(self.canonical_entities),
            "num_aliases": len(self.aliases),
            "entities_by_type": dict(
                sorted(
                    defaultdict(int, {
                        t: sum(1 for v in self.entity_types.values() if v == t)
                        for t in set(self.entity_types.values()) if t
                    }).items()
                )
            ),
        }
