"""
RQ3 pre-registration: the run guard and the machine-readable predictions
(RQ3_SPEC §4).

The guard. RQ3 runs refuse to start unless PREREG.md *at HEAD* (i.e.
committed) has a "## RQ3 predictions" section, no "____" / "YYYY-MM-DD" left
anywhere, and a parseable scoring block in that section.

The scoring block. Lines 10, 14, 15 and 17 contain numbers chosen by Adithya,
lines 11 to 13 a choice of words, and lines 14, 15 and 17 an expectation, so make_tables can only score them
mechanically if the choices are stated in a fixed form. The block sits inside
the RQ3 section, exactly like this (values are examples):

    rq3-scoring:
      line10_gold_I_minus_gold   = -0.05
      line10_gold_I_real         = yes
      line10_gold_sub_minus_gold = -0.04
      line10_ratio               = 0.75
      line11                     = lower
      line12                     = same
      line13                     = partly: teacher_I, perm_I
      line13_controls            = not_real
      line14_pct                 = 50
      line14_expect              = fails
      line15_pp                  = 5
      line15_expect              = holds
      line17_pct                 = 50
      line17_expect              = holds

Allowed values:
  line10_*_minus_gold, line10_ratio, line14_pct, line15_pp, line17_pct   a number
  line10_gold_I_real                yes | no
  line11, line12                    lower | higher | same
  line13                            holds | fails | partly: <conditions>
                                    (conditions from teacher_I, perm_I, rand_I, gold_I)
  line13_controls                   not_real | not_real_positive
      how "not real for rand_I and gold_I" is read: |delta| not real in
      either direction, or delta not real-positive (a real negative counts as
      consistent with the hypothesis)
  line14_expect, line15_expect, line17_expect   holds | fails
      whether the line is expected to hold; the scorecard reports the outcome
      and whether this expectation was correct

Line 17 (Gate 2) has no blank in prose; it is defined by its scoring rule in
make_tables.py and exists in the PREREG only through line17_pct / line17_expect
and Adithya's wording of it.

How each line is then scored is fixed in make_tables.py's module docstring.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

REPO = Path(__file__).resolve().parent.parent
HEADING = "## RQ3 predictions"
CL_CONDITIONS = ("teacher_I", "perm_I", "rand_I", "gold_I")

_NUMBER_KEYS = ("line10_gold_I_minus_gold", "line10_gold_sub_minus_gold", "line10_ratio",
                "line14_pct", "line15_pp", "line17_pct")
EXPECT_KEYS = ("line14_expect", "line15_expect", "line17_expect")
KEYS = _NUMBER_KEYS + ("line10_gold_I_real", "line11", "line12", "line13",
                       "line13_controls") + EXPECT_KEYS


def head_prereg() -> str:
    return subprocess.run(["git", "show", "HEAD:PREREG.md"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def rq3_section(text: str) -> str:
    i = text.find(HEADING)
    if i < 0:
        raise ValueError(f"no '{HEADING}' section")
    j = text.find("\n## ", i + len(HEADING))
    return text[i:] if j < 0 else text[i:j]


def parse_scoring(section: str) -> Dict:
    """Parse the rq3-scoring block. Raises ValueError on anything malformed."""
    lines = section.splitlines()
    try:
        start = next(k for k, l in enumerate(lines) if l.strip() == "rq3-scoring:")
    except StopIteration:
        raise ValueError("no 'rq3-scoring:' block in the RQ3 section")
    raw: Dict[str, str] = {}
    for l in lines[start + 1:]:
        m = re.fullmatch(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*", l)
        if not m:
            break
        if m.group(1) in raw:
            raise ValueError(f"{m.group(1)} given twice")
        raw[m.group(1)] = m.group(2)
    missing = [k for k in KEYS if k not in raw]
    unknown = [k for k in raw if k not in KEYS]
    if missing or unknown:
        raise ValueError(f"scoring block: missing {missing}, unknown {unknown}")

    out: Dict = {}
    for k in _NUMBER_KEYS:
        try:
            out[k] = float(raw[k].replace("−", "-").rstrip("%"))
        except ValueError:
            raise ValueError(f"{k} must be a number, got {raw[k]!r}")
    if raw["line10_gold_I_real"] not in ("yes", "no"):
        raise ValueError("line10_gold_I_real must be yes or no")
    out["line10_gold_I_real"] = raw["line10_gold_I_real"] == "yes"
    for k in ("line11", "line12"):
        if raw[k] not in ("lower", "higher", "same"):
            raise ValueError(f"{k} must be lower, higher or same")
        out[k] = raw[k]
    v = raw["line13"]
    if v in ("holds", "fails"):
        out["line13"] = (v, ())
    else:
        m = re.fullmatch(r"partly:\s*(.+)", v)
        conds = tuple(sorted(c.strip() for c in m.group(1).split(","))) if m else ()
        if not conds or any(c not in CL_CONDITIONS for c in conds):
            raise ValueError(f"line13 must be holds, fails or 'partly: <conditions>', got {v!r}")
        out["line13"] = ("partly", conds)
    if raw["line13_controls"] not in ("not_real", "not_real_positive"):
        raise ValueError("line13_controls must be not_real or not_real_positive")
    out["line13_controls"] = raw["line13_controls"]
    for k in EXPECT_KEYS:
        if raw[k] not in ("holds", "fails"):
            raise ValueError(f"{k} must be holds or fails")
        out[k] = raw[k]
    return out


def prereg_rq3_ready(text: str = None) -> Tuple[bool, str]:
    """(ok, why). Reads PREREG.md at HEAD unless text is given (tests)."""
    try:
        text = head_prereg() if text is None else text
    except Exception:
        return False, "PREREG.md is not committed"
    if "____" in text or "YYYY-MM-DD" in text:
        return False, "PREREG.md at HEAD still has blanks"
    try:
        parse_scoring(rq3_section(text))
    except ValueError as e:
        return False, f"PREREG.md at HEAD: {e}"
    return True, ""
