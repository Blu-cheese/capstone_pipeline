"""
Parse the DialogRE corpus into teacher-labelling units.

DialogRE (Yu et al., 2020 - https://github.com/nlpdata/dialogre) is the
standard benchmark for dialogue-level relation extraction. Released for
research use; cite the paper in any published work.

Why this corpus rather than AMI, for Track B:

    Measured directly on both. AMI yields 4-14 distinct relations with the
    most frequent class taking 36-77% of labels, and 16 of our 20 meeting
    relation types never occur at all - meeting speech is overwhelmingly
    discussion, so no taxonomy repairs it. DialogRE has 36 relation types
    (plus `unanswerable`), the most frequent takes 21%, and 31 of 35 have
    >=20 training examples. Phase 3 needs a balanced 3-task split; only
    DialogRE can supply one.

    It also ships 5,963 human-annotated pairs, so teacher quality can be
    measured against gold rather than against a hand-annotated sample.

Format: each record is [dialogue_lines, annotations], where annotations carry
the entity pair (x, y), their types, and the gold relation list `r`.

LIMITATION to state in the report: DialogRE is transcribed sitcom dialogue,
not meetings. The continual-learning method is domain-agnostic, but the
Track B corpus no longer matches Track A's meeting domain.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# The full DialogRE label space, sorted for a stable class-index mapping.
# Derived from train+dev+test; frozen here so the student's 37-way head and
# the task split never shift because a split was resampled.
DIALOGRE_RELATIONS = [
    "gpe:births_in_place", "gpe:residents_of_place", "gpe:visitors_of_place",
    "org:employees_or_members", "org:students",
    "per:acquaintance", "per:age", "per:alternate_names", "per:alumni",
    "per:boss", "per:children", "per:client", "per:date_of_birth",
    "per:dates", "per:employee_or_member_of", "per:friends",
    "per:girl/boyfriend", "per:major", "per:negative_impression",
    "per:neighbor", "per:origin", "per:other_family", "per:parents",
    "per:pet", "per:place_of_birth", "per:place_of_residence",
    "per:place_of_work", "per:positive_impression", "per:roommate",
    "per:schools_attended", "per:siblings", "per:spouse", "per:subordinate",
    "per:title", "per:visited_place", "per:works", "unanswerable",
]

RELATION_TO_INDEX = {r: i for i, r in enumerate(DIALOGRE_RELATIONS)}


@dataclass
class EntityPair:
    """One entity pair inside a dialogue, with its gold relation."""
    x: str
    x_type: str
    y: str
    y_type: str
    gold_relations: List[str]
    trigger: str = ""

    @property
    def gold(self) -> str:
        """
        Single gold label. Only meaningful for single-label pairs - guard with
        `is_multilabel` first; multi-label pairs are DROPPED, not collapsed.

        Collapse-to-first was the earlier policy and was wrong. "First"
        reflects annotation order, not salience, so it silently assigns an
        arbitrary label. On eval that is actively harmful: a model predicting
        the pair's *second* gold relation is scored wrong despite being right,
        which injects spurious errors that depress every regime unevenly.
        """
        return self.gold_relations[0] if self.gold_relations else "unanswerable"

    @property
    def is_multilabel(self) -> bool:
        return len(self.gold_relations) > 1

    def to_dict(self) -> Dict:
        return {
            "x": self.x, "x_type": self.x_type,
            "y": self.y, "y_type": self.y_type,
            "gold_relations": self.gold_relations,
            "gold": self.gold,
            "trigger": self.trigger,
        }


@dataclass
class DialogueUnit:
    """
    One dialogue plus every annotated entity pair in it.

    This is the unit of teacher labelling: the dialogue is sent to the LLM
    once with all its pairs, rather than one request per pair. That cuts API
    calls by ~5.6x (the mean pairs-per-dialogue) against a rate-limited
    quota, and gives the model the same context either way.
    """
    dialogue_id: str
    lines: List[str]
    pairs: List[EntityPair] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def to_dict(self) -> Dict:
        return {
            "dialogue_id": self.dialogue_id,
            "text": self.text,
            "pairs": [p.to_dict() for p in self.pairs],
        }


def load_dialogre(
    path: str,
    limit: Optional[int] = None,
    skip_unanswerable_only: bool = False,
) -> List[DialogueUnit]:
    """
    Load a DialogRE split into DialogueUnits.

    Args:
        path: train.json / dev.json / test.json
        limit: cap the number of dialogues loaded.
        skip_unanswerable_only: drop dialogues whose pairs are all
            `unanswerable`. Off by default - `unanswerable` is a real class
            in the label space, and dropping it would bias the student
            toward always predicting a positive relation.

    Raises ValueError if nothing usable parsed, rather than returning empty.
    """
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)

    units: List[DialogueUnit] = []
    stem = Path(path).stem

    for i, record in enumerate(records):
        if limit is not None and len(units) >= limit:
            break
        if not isinstance(record, list) or len(record) < 2:
            continue

        lines, annotations = record[0], record[1]
        if not lines or not annotations:
            continue

        pairs = [
            EntityPair(
                x=str(a.get("x", "")).strip(),
                x_type=str(a.get("x_type", "")).strip(),
                y=str(a.get("y", "")).strip(),
                y_type=str(a.get("y_type", "")).strip(),
                gold_relations=list(a.get("r", [])),
                trigger=str(a.get("t", [""])[0] if isinstance(a.get("t"), list) else a.get("t", "")),
            )
            for a in annotations
            if a.get("x") and a.get("y")
        ]
        if not pairs:
            continue

        if skip_unanswerable_only and all(p.gold == "unanswerable" for p in pairs):
            continue

        units.append(DialogueUnit(f"{stem}_{i:04d}", list(lines), pairs))

    if not units:
        raise ValueError(f"No usable DialogRE dialogues parsed from {path}")

    total_pairs = sum(len(u.pairs) for u in units)
    print(f"Loaded {len(units)} DialogRE dialogues from {path} "
          f"({total_pairs} annotated pairs, {total_pairs/len(units):.1f}/dialogue)")
    return units
