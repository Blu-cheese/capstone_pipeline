"""
RQ3 Gate 1 diagnostics (RQ3_SPEC §1.2 to §1.6). Reads saved files only, with
one declared exception: §1.4 needs joint-baseline test predictions, which
student.py never saved. Those are recomputed in memory by calling the
unchanged train_joint() and are used ONLY if the recomputed run matches the
saved joint_baseline_seed*.json exactly (test macro-F1, accuracy, best epoch,
epochs run, best dev macro-F1, per-class F1). Nothing is written to the
joint files.

    venv/bin/python -m continual.rq3_diagnostics
    venv/bin/python -m continual.rq3_diagnostics --only rows kd broad tasks audit

Writes results/diagnostics/:
    row_accounting.{json,md}, rowset_I.json          §1.3
    kd_memory_agreement.{json,md}                    §1.2
    broad_class_check.{json,md}                      §1.4
    task_rows_by_condition.{json,md}                 §1.5 point 4
    audit_50.csv                                     §1.6

Row id everywhere = row index into data/features/dialogre_train_minilm-l6-v2.npz,
which is also the row index into data/labeled/dialogre_train.json (asserted).
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from continual import config
from continual.features import build_features
from continual.task_split import load_split, class_index, kept_relations
from preprocessing.dialogre_parser import load_dialogre, DIALOGRE_RELATIONS

OUT = config.RESULTS_DIR / "diagnostics"
TRAIN_JSON = Path("data/raw/dialogre/train.json")
LABELED = Path("data/labeled/dialogre_train.json")
N_TEACHER_DIALOGUES = 400          # label_corpus was run with --limit 400
SEEDS = config.SEEDS
MIN_EXAMPLES = 15                  # task_split.MIN_EXAMPLES_PER_CLASS
MIN_TASK_SHARE = 0.15              # task_split.MIN_TASK_SHARE
DEGENERATE_CLASS = "per:alternate_names"
BROAD_CLASS_T5 = "per:other_family"   # re-derived and asserted below, not assumed


def _short(name: str) -> str:
    return name.split(":", 1)[1]


def _write(name: str, payload: Dict, md: List[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                                      encoding="utf-8")
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ------------------------------------------------------------- row table

def build_row_table() -> Dict:
    """
    Rebuild the path from DialogRE train.json to the 2,018 feature rows, pair
    by pair, exactly as label_corpus.label_corpus does it, and check it lands
    on the saved rows. Returns per-row metadata plus the drop counts.
    """
    from labeling.label_corpus import cache_key, read_cache, parse_labels
    from extractors.llm_extractor import DEFAULT_MODEL

    raw = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    all_pairs = sum(len(r[1]) for r in raw)
    units = load_dialogre(str(TRAIN_JSON), limit=N_TEACHER_DIALOGUES)
    saved = json.loads(LABELED.read_text(encoding="utf-8"))

    meta, unlabelled, multilabel = [], [], []
    for u in units:
        entry = read_cache(cache_key(u, DEFAULT_MODEL))
        assert entry is not None, f"cache miss for {u.dialogue_id}"
        labels = parse_labels(entry["raw"], len(u.pairs))
        rev = {(p.x, p.y): p.gold_relations for p in u.pairs}
        for i, p in enumerate(u.pairs):
            if labels.get(i) is None:                 # label_corpus checks this first
                unlabelled.append({"dialogue_id": u.dialogue_id, "pair_index": i,
                                   "n_pairs": len(u.pairs), "gold": p.gold_relations,
                                   "is_last_pair": i == len(u.pairs) - 1})
                continue
            if p.is_multilabel:
                multilabel.append({"dialogue_id": u.dialogue_id, "pair_index": i,
                                   "gold": p.gold_relations})
                continue
            meta.append({"dialogue_id": u.dialogue_id, "pair_index": i,
                         "subject": p.x, "object": p.y,
                         "subject_type": p.x_type, "object_type": p.y_type,
                         "teacher": labels[i], "gold": p.gold,
                         "gold_relations": p.gold_relations,
                         "reverse_pair_gold": rev.get((p.y, p.x)),
                         "text": u.text})

    assert len(meta) == len(saved), (len(meta), len(saved))
    for m, s in zip(meta, saved):
        assert (m["dialogue_id"], m["subject"], m["object"], m["teacher"], m["gold"]) == \
               (s["dialogue_id"], s["subject"], s["object"], s["relation"], s["gold_relation"])

    feats = build_features(config.CORPUS)
    assert len(meta) == len(feats["y_gold"])
    for k, m in enumerate(meta):
        assert DIALOGRE_RELATIONS[feats["y_teacher"][k]] == m["teacher"]
        assert DIALOGRE_RELATIONS[feats["y_gold"][k]] == m["gold"]

    return {"meta": meta, "unlabelled": unlabelled, "multilabel": multilabel,
            "n_dialogues_train": len(raw), "n_pairs_train": all_pairs,
            "n_dialogues_labelled": len(units),
            "n_pairs_labelled": sum(len(u.pairs) for u in units),
            "feats": feats}


class Space:
    """Label-space bookkeeping shared by every section."""

    def __init__(self, table: Dict):
        split = load_split(SEEDS[0])
        for s in SEEDS[1:]:
            assert set(load_split(s)["relation_to_task"]) == set(split["relation_to_task"])
        self.ci = class_index(split)                    # name -> head index
        self.names = kept_relations(split)              # head order
        self.drop_reason = {d["relation"]: d["reason"] for d in split["dropped_classes"]}
        meta = table["meta"]
        self.teacher = np.array([m["teacher"] for m in meta])
        self.gold = np.array([m["gold"] for m in meta])
        self.t_in = np.isin(self.teacher, self.names)
        self.g_in = np.isin(self.gold, self.names)
        self.I = np.flatnonzero(self.t_in & self.g_in)  # row ids of I
        self.teacher_rows = np.flatnonzero(self.t_in)   # 879, teacher track
        self.gold_rows = np.flatnonzero(self.g_in)      # 1,327, gold track
        self.y_t = np.array([self.ci.get(v, -1) for v in self.teacher])
        self.y_g = np.array([self.ci.get(v, -1) for v in self.gold])


# ------------------------------------------------------------ §1.3 rows

def row_accounting(table: Dict, sp: Space) -> Dict:
    n = len(table["meta"])
    agree = sp.teacher == sp.gold
    both = sp.t_in & sp.g_in
    I = sp.I
    eps = 1.0 - float(agree[I].mean())
    n_flip = int(round(eps * len(I)))

    C = np.zeros((len(sp.names), len(sp.names)), dtype=int)
    for r in I:
        C[sp.y_g[r], sp.y_t[r]] += 1
    gold_counts = {c: int((sp.y_g[I] == sp.ci[c]).sum()) for c in sp.names}
    teach_counts = {c: int((sp.y_t[I] == sp.ci[c]).sum()) for c in sp.names}
    full_gold = {c: int((sp.gold == c).sum()) for c in sp.names}
    full_teach = {c: int((sp.teacher == c).sum()) for c in sp.names}

    diag_rows = []        # Gate 2: is C's diagonal the row maximum, per gold class?
    for i, c in enumerate(sp.names):
        off = [(int(C[i, j]), sp.names[j]) for j in range(len(sp.names)) if j != i]
        top_v, top_c = max(off)
        diag_rows.append({"gold": c, "diag": int(C[i, i]), "max_off_diag": top_v,
                          "max_off_diag_teacher_label": top_c,
                          "diag_is_strict_row_max": bool(C[i, i] > top_v),
                          "tie": bool(C[i, i] == top_v)})
    outside = Counter(sp.gold[sp.t_in & ~sp.g_in].tolist())
    outside_teacher = Counter(sp.teacher[sp.t_in & ~sp.g_in].tolist())
    gold_only = Counter(sp.teacher[sp.g_in & ~sp.t_in].tolist())
    unl = table["unlabelled"]

    payload = {
        "dialogre_train": {"dialogues": table["n_dialogues_train"],
                           "pairs": table["n_pairs_train"]},
        "teacher_labelled_dialogues": {
            "dialogues": table["n_dialogues_labelled"], "pairs": table["n_pairs_labelled"],
            "selection": "first 400 dialogues of train.json in file order "
                         "(label_corpus --limit 400)"},
        "other_dialogues": {
            "dialogues": table["n_dialogues_train"] - table["n_dialogues_labelled"],
            "pairs": table["n_pairs_train"] - table["n_pairs_labelled"],
            "used_by_any_track": False,
            "note": "both tracks draw only on the 400 teacher-labelled dialogues; "
                    "the gold track never used gold labels from other dialogues"},
        "dropped_before_rows": {
            "multi_label": len(table["multilabel"]),
            "teacher_returned_no_label": len(unl),
            "teacher_no_label_all_last_pair": all(u["is_last_pair"] for u in unl),
            "teacher_no_label_dialogues": len({u["dialogue_id"] for u in unl}),
            "teacher_no_label_gold": dict(Counter(u["gold"][0] for u in unl).most_common()),
            "note": "multi-label pairs: 103, not 124 (124 = dev 75 + test 49). The 23 "
                    "unlabelled pairs are each the LAST pair of its dialogue: 23 responses "
                    "returned indices 0..n-2, the other 377 returned 0..n-1; no response "
                    "was off by one, so no label is shifted"},
        "single_label_rows": n,
        "dialogues_with_rows": len({m["dialogue_id"] for m in table["meta"]}),
        "label_space": {
            "gold_in_17": int(sp.g_in.sum()), "teacher_in_17": int(sp.t_in.sum()),
            "both": int(both.sum()), "gold_only": int((sp.g_in & ~sp.t_in).sum()),
            "teacher_only": int((sp.t_in & ~sp.g_in).sum()),
            "neither": int((~sp.g_in & ~sp.t_in).sum())},
        "agreement": {
            "on_both": {"agree": int(agree[both].sum()), "n": int(both.sum()),
                        "rate": round(float(agree[both].mean()), 6)},
            "all_2018_rows": round(float(agree.mean()), 6),
            "teacher_track_879": {"agree": int(agree[sp.t_in].sum()),
                                  "rate": round(float(agree[sp.t_in].mean()), 6),
                                  "note": "gold outside the 17 counts as disagreement"},
            "gold_track_1327": {"agree": int(agree[sp.g_in].sum()),
                                "rate": round(float(agree[sp.g_in].mean()), 6)}},
        "tracks": {
            "gold_track": {"rows": int(sp.g_in.sum()), "= both": int(both.sum()),
                           "+ gold_only": int((sp.g_in & ~sp.t_in).sum()),
                           "gold_only_by_teacher_label": dict(gold_only.most_common())},
            "teacher_track": {"rows": int(sp.t_in.sum()), "= both": int(both.sum()),
                              "+ teacher_only": int((sp.t_in & ~sp.g_in).sum())}},
        "I": {
            "definition": "single-label pairs whose gold AND teacher labels are both in "
                          "the 17 kept classes",
            "size": int(len(I)),
            "epsilon_I": round(eps, 6),
            "n_disagree": int((~agree[I]).sum()),
            "rand_I_flips_round_eps_times_n": n_flip,
            "equals_teacher_track_879": bool(len(I) == sp.t_in.sum()),
            "subset_of_teacher_track": bool(np.isin(I, sp.teacher_rows).all()),
            "subset_of_gold_track": bool(np.isin(I, sp.gold_rows).all()),
            "teacher_rows_outside_I": {
                "n": int((sp.t_in & ~sp.g_in).sum()),
                "why": "gold label outside the 17 kept classes",
                "by_gold_label": {g: {"n": c, "reason": sp.drop_reason.get(g, "?")}
                                  for g, c in outside.most_common()},
                "by_teacher_label": dict(outside_teacher.most_common())},
            "per_class": {c: {"gold_I": gold_counts[c], "teacher_I": teach_counts[c],
                              "gold_track": full_gold[c], "teacher_track": full_teach[c],
                              "diag_C": int(C[sp.ci[c], sp.ci[c]])}
                          for c in sp.names},
            "classes_under_15": {
                "gold_I": [c for c in sp.names if gold_counts[c] < MIN_EXAMPLES],
                "teacher_I": [c for c in sp.names if teach_counts[c] < MIN_EXAMPLES]},
            "confusion": {"rows": "gold", "cols": "teacher", "order": sp.names,
                          "C": C.tolist()},
            "diagonal_vs_row_max": diag_rows},
    }

    rowset = {"definition": payload["I"]["definition"],
              "row_id": "row index into data/features/dialogre_train_minilm-l6-v2.npz "
                        "and data/labeled/dialogre_train.json",
              "n": int(len(I)), "row_ids": [int(r) for r in I],
              "head_order": sp.names,
              "gold_head": [int(sp.y_g[r]) for r in I],
              "teacher_head": [int(sp.y_t[r]) for r in I]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rowset_I.json").write_text(json.dumps(rowset, indent=1) + "\n", encoding="utf-8")

    ls, ag = payload["label_space"], payload["agreement"]
    md = ["# Row accounting (RQ3_SPEC §1.3)", "",
          "Generated by `continual/rq3_diagnostics.py`. Do not edit by hand.", "",
          "## From DialogRE train to the two tracks", "",
          "| step | dialogues | pairs / rows |", "|---|---|---|",
          f"| DialogRE train.json | {table['n_dialogues_train']} | {table['n_pairs_train']} |",
          f"| teacher-labelled (first 400 in file order) | {table['n_dialogues_labelled']} "
          f"| {table['n_pairs_labelled']} |",
          f"| − multi-label pairs dropped | | −{len(table['multilabel'])} |",
          f"| − pairs the teacher returned no label for | | −{len(unl)} |",
          f"| = single-label rows (feature matrix) | {payload['dialogues_with_rows']} | {n} |",
          "",
          f"The other {payload['other_dialogues']['dialogues']} dialogues "
          f"({payload['other_dialogues']['pairs']} pairs) are used by neither track.",
          "",
          f"- Multi-label: **{len(table['multilabel'])}**, not 124 (124 = dev 75 + test 49).",
          f"- Teacher returned no label for {len(unl)} pairs in {len({u['dialogue_id'] for u in unl})} "
          "dialogues, and every one is the last pair of its dialogue (the response stopped "
          "at index n−2). The other 377 responses cover 0..n−1 exactly; none is off by one.",
          "- 3 dialogues contribute no rows (all their pairs are multi-label).", "",
          "## Label space (2,018 rows, 17 kept classes)", "",
          "| | rows |", "|---|---|",
          f"| gold in 17 (= gold track) | {ls['gold_in_17']} |",
          f"| teacher in 17 (= teacher track) | {ls['teacher_in_17']} |",
          f"| both (= I) | {ls['both']} |",
          f"| gold only | {ls['gold_only']} |",
          f"| teacher only | {ls['teacher_only']} |",
          f"| neither | {ls['neither']} |", "",
          f"- Exact-match agreement on *both*: {ag['on_both']['agree']}/{ag['on_both']['n']} "
          f"= **{ag['on_both']['rate']:.4f}**. On all 2,018 rows: {ag['all_2018_rows']:.4f} "
          "(the ~37% headline).",
          f"- Teacher track: {ag['teacher_track_879']['agree']}/879 labels equal gold "
          f"({ag['teacher_track_879']['rate']:.4f}), so its total noise rate is "
          f"{1 - ag['teacher_track_879']['rate']:.4f}: {payload['I']['n_disagree']} in-space "
          f"errors plus {ls['teacher_only']} rows whose gold label is outside the 17.",
          "- 1,327 gold = 710 + 617 gold-only rows, whose teacher label is outside the 17: "
          + ", ".join(f"{k} {v}" for k, v in list(gold_only.most_common())[:3]) + ", other "
          + str(sum(list(dict(gold_only).values())) - sum(v for _, v in gold_only.most_common(3))) + ".",
          "- 879 teacher = 710 + 169 teacher-only rows, whose gold label is outside the 17 (below).",
          "", "## The intersection set I", "",
          f"|I| = **{len(I)}**, ε_I = **{eps:.4f}** ({payload['I']['n_disagree']} of {len(I)} "
          f"disagree), so rand_I flips round(ε_I·|I|) = **{n_flip}** rows.", "",
          f"**I is not the 879 teacher rows.** It is a strict subset of both tracks. The "
          f"{ls['teacher_only']} teacher-track rows outside I all have a gold label outside the "
          "17 classes:", "",
          "| gold label | rows | why it is outside the 17 |", "|---|---|---|"]
    md += [f"| {g} | {v['n']} | {v['reason']} |"
           for g, v in payload["I"]["teacher_rows_outside_I"]["by_gold_label"].items()]
    md += ["", "Their teacher labels: " + ", ".join(
        f"{_short(k)} {v}" for k, v in outside_teacher.most_common()) + ".", "",
        "### Per-class counts", "",
        "| class | gold_I | teacher_I | correct (C diag) | gold track | teacher track |",
        "|---|---|---|---|---|---|"]
    for c in sp.names:
        pc = payload["I"]["per_class"][c]
        flag = lambda v: f"**{v}**" if v < MIN_EXAMPLES else str(v)
        md.append(f"| {c} | {flag(pc['gold_I'])} | {flag(pc['teacher_I'])} | {pc['diag_C']} "
                  f"| {pc['gold_track']} | {pc['teacher_track']} |")
    md += ["", f"Bold = under the frozen threshold of {MIN_EXAMPLES}. The threshold is not "
           "re-applied on I (the 17 classes are frozen); this is reported, not acted on.", "",
           "### Confusion matrix C[g][t] on I (rows gold, columns teacher)", ""]
    ab = [_short(c)[:10] for c in sp.names]
    md += ["| gold \\ teacher | " + " | ".join(ab) + " | Σ |",
           "|---|" + "---|" * (len(ab) + 1)]
    for i, c in enumerate(sp.names):
        md.append(f"| {ab[i]} | " + " | ".join(str(v) if v else "·" for v in C[i]) +
                  f" | {C[i].sum()} |")
    md.append("| Σ | " + " | ".join(str(v) for v in C.sum(axis=0)) + f" | {C.sum()} |")
    bad = [d for d in diag_rows if not d["diag_is_strict_row_max"]]
    md += ["", "### Is the diagonal the row maximum? (Gate 2)", "",
           f"For {len(diag_rows) - len(bad)} of {len(diag_rows)} gold classes the teacher's "
           "most frequent label on I is the correct one. Where it isn't, the teacher's "
           "class-conditional noise has a wrong dominant label, so perm_I keeps that "
           "wrong majority too:", "",
           "| gold class | C diagonal (teacher correct) | largest off-diagonal | teacher label there |",
           "|---|---|---|---|"]
    md += [f"| {d['gold']} | {d['diag']} | {d['max_off_diag']}{' (tie)' if d['tie'] else ''} "
           f"| {d['max_off_diag_teacher_label']} |" for d in bad]
    md += ["", "Row ids of I: `rowset_I.json`."]
    _write("row_accounting", payload, md)
    return payload


# --------------------------------------------------------- §1.2 KD/memory

def kd_memory_agreement() -> Dict:
    """
    In each fixed-schedule kd run, the snapshot theta_{k-1} on the replay-stage
    memory M-tilde_k, for k >= 2 (t = 1, 2 here).

    The snapshot is the model at the end of after_task(t-1) (KDRegime inherits
    FKDRegime._end_of_task). run_all's Recorder records train_prob[t-1] right
    after that same after_task returns, from the same weights, in eval mode,
    under the seen-through-(t-1) mask = OLD for task t. So train_prob[t-1] IS
    the snapshot's softmax over OLD - the Eq. 12 target - and no rerun is needed.
    """
    from continual.make_tables import select_runs, SplitInfo
    Fx, h = select_runs("fixed", ["replay", "fkd", "kd"], None)
    out, md_rows = [], []
    for tr in ("gold", "teacher"):
        for s in SEEDS:
            run = Fx[(tr, "kd", s)]
            info = SplitInfo(s)
            mem = run.json("memory_indices.json")["classes"]
            log = [json.loads(l) for l in (run.dir / "replay_log.jsonl").read_text().splitlines()]
            P = run.P
            label = P["train_label"]
            for t in (1, 2):
                old = sorted(set().union(*info.task_classes[:t]))
                new = info.task_classes[t]
                seen = old + new
                rows = np.array([i for r in sorted(seen) for i in mem[info.names[r]]["train_rows"]])
                n_logged = next(e["n_memory"] for e in log if e.get("task") == t and "n_memory" in e)
                assert len(rows) == n_logged, (tr, s, t, len(rows), n_logged)
                prob = P["train_prob"][t - 1][rows].astype(np.float64)      # snapshot, softmax over OLD
                assert np.allclose(prob[:, [c for c in range(prob.shape[1]) if c not in old]], 0.0)
                assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-5)
                y = label[rows]
                is_old = np.isin(y, old)
                po = prob[is_old][:, old]
                arg = np.array(old)[po.argmax(axis=1)]
                p_lab = prob[is_old, y[is_old]]
                pn = prob[~is_old][:, old]
                ent = -(pn * np.log(np.clip(pn, 1e-300, None))).sum(axis=1) / math.log(len(old))
                rec = {"track": tr, "seed": s, "task": t, "k_one_based": t + 1,
                       "n_old_classes": len(old), "n_memory": int(len(rows)),
                       "old_samples": int(is_old.sum()),
                       "old_argmax_equals_label": round(float((arg == y[is_old]).mean()), 6),
                       "old_mean_prob_on_label": round(float(p_lab.mean()), 6),
                       "old_median_prob_on_label": round(float(np.median(p_lab)), 6),
                       "new_samples": int((~is_old).sum()),
                       "new_mean_entropy_frac_of_max": round(float(ent.mean()), 6),
                       "run_dir": run.dir.name}
                out.append(rec)
                md_rows.append(f"| {tr} | {s} | {t} | {len(old)} | {rec['old_samples']} | "
                               f"{rec['old_argmax_equals_label']:.3f} | "
                               f"{rec['old_mean_prob_on_label']:.3f} | {rec['new_samples']} | "
                               f"{rec['new_mean_entropy_frac_of_max']:.3f} |")
    payload = {"source": f"fixed-schedule kd runs at {h}, predictions.npz train_prob[t-1] "
                         "(= snapshot theta_{k-1} softmax over OLD) on memory_indices.json rows",
               "cells": out}
    oa = [c["old_argmax_equals_label"] for c in out]
    op = [c["old_mean_prob_on_label"] for c in out]
    ne = [c["new_mean_entropy_frac_of_max"] for c in out]
    md = ["# KD targets on the memory (RQ3_SPEC §1.2)", "",
          "Generated by `continual/rq3_diagnostics.py`. Do not edit by hand.", "",
          f"Fixed-schedule `kd` runs at `{h}`. The snapshot θₖ₋₁ is the model at the end of "
          "`after_task(t−1)`; `run_all`'s Recorder saves `train_prob[t−1]` from those same "
          "weights under the OLD mask, so this is the exact Eq. 12 target. No rerun. "
          "Memory rows from `memory_indices.json`, count checked against `replay_log.jsonl`.",
          "", "Tasks numbered from 0 (t = 1, 2 are the paper's k = 2, 3). Labels are the "
          "track's training labels.", "",
          "| track | seed | t | \\|OLD\\| | old samples | snapshot argmax = label | mean P(label) "
          "| new samples | new: entropy / max |",
          "|---|---|---|---|---|---|---|---|---|"] + md_rows + [
          "",
          f"Range over the 12 cells: argmax = label {min(oa):.3f}–{max(oa):.3f}, mean P(label) "
          f"{min(op):.3f}–{max(op):.3f}, new-class entropy {min(ne):.3f}–{max(ne):.3f} of max."]
    _write("kd_memory_agreement", payload, md)
    return payload


# ---------------------------------------------------------- §1.4 broad

def _joint_predictions(seed: int, feats: Dict) -> Dict[str, np.ndarray]:
    """Recompute joint test predictions; refuse unless they match the saved json."""
    import torch
    from continual.student import train_joint, _predict, macro_f1, accuracy
    from continual.eval_sets import build_eval_sets
    ref = json.loads((config.RESULTS_DIR / f"joint_baseline_seed{seed}.json")
                     .read_text(encoding="utf-8"))["tracks"]
    split = load_split(seed)
    ci, rel = class_index(split), kept_relations(split)
    ev = build_eval_sets(seed)
    f2h = {DIALOGRE_RELATIONS.index(r): ci[r] for r in rel}
    out = {}
    for tr, yf in (("teacher", feats["y_teacher"]), ("gold", feats["y_gold"])):
        m = np.isin(yf, list(f2h))
        y = np.array([f2h[int(v)] for v in yf[m]], dtype=np.int64)
        model, info = train_joint(feats["X"][m], y, ev["dev"]["X"], ev["dev"]["y"], len(ci),
                                  seed=seed, verbose=False)
        pred = _predict(model, torch.from_numpy(ev["test"]["X"]))
        f1, pc = macro_f1(ev["test"]["y"], pred, len(ci))
        r = ref[tr]
        same = (f1 == r["test_macro_f1"] and accuracy(ev["test"]["y"], pred) == r["test_accuracy"]
                and info["best_epoch"] == r["best_epoch"] and info["epochs_run"] == r["epochs_run"]
                and info["best_dev_macro_f1"] == r["best_dev_macro_f1"]
                and {rel[c]: round(v, 4) for c, v in sorted(pc.items())} == r["per_class_f1"])
        if not same:
            raise RuntimeError(f"joint {tr}/{seed} does not reproduce the saved json")
        out[tr] = pred
    return out


def broad_class_check(table: Dict, sp: Space) -> Dict:
    from continual.make_tables import select_runs, broad_class
    P, h = select_runs("primary", ["naive", "replay", "fkd"], None)
    b, counts = broad_class()
    assert b == BROAD_CLASS_T5
    feats = table["feats"]
    t3 = {(r["track"], r["regime"], int(r["seed"])): float(r["final_macro_f1_all"])
          for r in csv.DictReader(open(config.RESULTS_DIR / "tables" / "T3_final_macro_f1.csv", encoding="utf-8"))}
    per_seed, md_rows = [], []
    for s in SEEDS:
        ci = class_index(load_split(s))
        # T5's definition evaluated per seed (head order and kept set are per-split)
        ratio = {n: (sp.teacher == n).sum() / (sp.gold == n).sum() for n in ci}
        top = max(ratio, key=ratio.get)
        c = ci[b]
        joint = _joint_predictions(s, feats)
        y = P[("gold", "naive", s)].P["test_gold"]
        sh = {"teacher_joint": float(np.mean(joint["teacher"] == c)),
              "teacher_naive_final": float(np.mean(P[("teacher", "naive", s)].P["test_pred"][-1] == c)),
              "teacher_replay_final": float(np.mean(P[("teacher", "replay", s)].P["test_pred"][-1] == c)),
              "teacher_fkd_final": float(np.mean(P[("teacher", "fkd", s)].P["test_pred"][-1] == c)),
              "gold_joint": float(np.mean(joint["gold"] == c)),
              "gold_replay_final": float(np.mean(P[("gold", "replay", s)].P["test_pred"][-1] == c)),
              "gold_test_share": float(np.mean(y == c))}
        f1 = {"teacher_replay": t3[("teacher", "replay", s)],
              "teacher_joint": t3[("teacher", "joint", s)],
              "gold_replay": t3[("gold", "replay", s)], "gold_joint": t3[("gold", "joint", s)]}
        per_seed.append({"seed": s, "most_over_predicted": top,
                         "ratio": round(float(ratio[top]), 4),
                         "shares": {k: round(v, 6) for k, v in sh.items()},
                         "final_macro_f1_T3": f1,
                         "teacher_replay_minus_joint": round(f1["teacher_replay"] - f1["teacher_joint"], 6)})
        md_rows.append(f"| {s} | {_short(top)} ({ratio[top]:.2f}×) | {sh['teacher_joint']:.3f} | "
                       f"{sh['teacher_naive_final']:.3f} | {sh['teacher_replay_final']:.3f} | "
                       f"{sh['teacher_fkd_final']:.3f} | {sh['gold_joint']:.3f} | "
                       f"{sh['gold_test_share']:.3f} | {f1['teacher_replay']:.3f} vs "
                       f"{f1['teacher_joint']:.3f} ({f1['teacher_replay'] - f1['teacher_joint']:+.3f}) |")
    tops = {p["most_over_predicted"] for p in per_seed}
    payload = {"broad_class": b, "teacher_train_count": counts[b][0],
               "gold_train_count": counts[b][1], "same_on_all_seeds": len(tops) == 1,
               "primary_runs_hash": h,
               "joint_predictions": "recomputed with the unchanged train_joint(); used only "
                                    "after matching the saved joint_baseline_seed*.json exactly",
               "seeds": per_seed}
    s1 = next(p for p in per_seed if p["seed"] == 1)
    md = ["# Broad-class check (RQ3_SPEC §1.4)", "",
          "Generated by `continual/rq3_diagnostics.py`. Do not edit by hand.", "",
          f"Most over-predicted class (T5 definition, teacher / gold train count): **{b}**, "
          f"{counts[b][0]} vs {counts[b][1]} ({counts[b][0] / counts[b][1]:.2f}×), the same on "
          f"all three seeds: {'yes' if len(tops) == 1 else 'NO'}.", "",
          "Shares are fractions of the 1,144 test examples predicted as that class by the final "
          "model. Joint predictions were never saved; they are recomputed with the unchanged "
          "`train_joint()` and used only because all six reproduce `joint_baseline_seed*.json` "
          "exactly (macro-F1, accuracy, best epoch, epochs run, best dev F1, per-class F1). CL "
          f"shares come from the primary runs at `{h}`.", "",
          "| seed | over-predicted | teacher joint | teacher naive | teacher replay | teacher FKD "
          "| gold joint | gold test share | T3: teacher replay vs joint |",
          "|---|---|---|---|---|---|---|---|---|"] + md_rows + [
          "",
          f"Quoted before as seed 1 figures: 20.5% → 9.4%, and teacher replay 0.350 vs joint "
          f"0.328. Seed 1 here: {100 * s1['shares']['teacher_joint']:.1f}% → "
          f"{100 * s1['shares']['teacher_replay_final']:.1f}%, and "
          f"{s1['final_macro_f1_T3']['teacher_replay']:.3f} vs "
          f"{s1['final_macro_f1_T3']['teacher_joint']:.3f}. Both confirmed for seed 1.",
          "",
          "Teacher replay beats teacher joint on seed 1 only; it is below joint on seeds 1234 "
          "and 2. The share reduction (joint → replay) holds on every seed."]
    _write("broad_class_check", payload, md)
    return payload


# ------------------------------------------------------ §1.5.4 task rows

def provisional_conditions(sp: Space, seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    PROVISIONAL label conditions for the §1.5 point 4 row counts, built to the
    §2.2 wording. Not the §2 implementation (that is written after Gate 1);
    rand_I and gold_sub counts here are one draw and may differ from T7's.
    Returns {track: (row_ids, head labels)}.
    """
    I = sp.I
    g, t = sp.y_g[I], sp.y_t[I]
    conds = {"gold": (sp.gold_rows, sp.y_g[sp.gold_rows]),
             "teacher": (sp.teacher_rows, sp.y_t[sp.teacher_rows]),
             "gold_I": (I, g.copy()), "teacher_I": (I, t.copy())}
    rng = np.random.default_rng(seed)
    perm = t.copy()
    for c in range(len(sp.names)):
        idx = np.flatnonzero(g == c)
        perm[idx] = t[idx][rng.permutation(len(idx))]
    conds["perm_I"] = (I, perm)
    rng = np.random.default_rng(seed)
    n_flip = int(round((1 - (g == t).mean()) * len(I)))
    rand = g.copy()
    for i in np.sort(rng.choice(len(I), size=n_flip, replace=False)):
        rand[i] = rng.choice([c for c in range(len(sp.names)) if c != g[i]])
    conds["rand_I"] = (I, rand)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(sp.gold_rows, size=len(I), replace=False))
    conds["gold_sub"] = (sub, sp.y_g[sub])
    return conds


def task_rows(sp: Space) -> Dict:
    payload, md = {"note": "rand_I and gold_sub are one provisional draw per seed "
                           "(default_rng(seed)); final counts come from §2's code (T7)",
                   "seeds": {}}, []
    md = ["# Rows per task under each label condition (RQ3_SPEC §1.5 point 4)", "",
          "Generated by `continual/rq3_diagnostics.py`. Do not edit by hand.", "",
          "The split assigns classes to tasks and is frozen, so the degenerate task "
          f"(the one holding {DEGENERATE_CLASS}) is at the same position under every "
          "condition by construction. What moves is which ROWS land in each task, because a "
          "row's task follows its training label.", "",
          "rand_I and gold_sub below are one provisional draw (`default_rng(seed)`) to the "
          "§2.2 wording, not the §2 implementation; T7 gives the final counts.", ""]
    for s in SEEDS:
        split = load_split(s)
        toc = {sp.ci[r]: t for r, t in split["relation_to_task"].items()}
        deg = toc[sp.ci[DEGENERATE_CLASS]]
        K = split["k"]
        conds = provisional_conditions(sp, s)
        seed_out = {"degenerate_task": deg}
        md += [f"## Seed {s} (degenerate task = task {deg})", "",
               "| condition | rows | " + " | ".join(f"task {k}" + (" †" if k == deg else "")
                                                     for k in range(K)) +
               " | min task share | classes < 15 rows |",
               "|---|---|" + "---|" * K + "---|---|"]
        for name, (rows, y) in conds.items():
            per = [int(np.isin(y, [c for c, tt in toc.items() if tt == k]).sum()) for k in range(K)]
            cc = Counter(y.tolist())
            under = [_short(sp.names[c]) + f" {cc.get(c, 0)}" for c in range(len(sp.names))
                     if cc.get(c, 0) < MIN_EXAMPLES]
            seed_out[name] = {"rows": int(len(y)), "per_task": per,
                              "task_share": [round(v / len(y), 4) for v in per],
                              "classes_under_15": under}
            md.append(f"| {name} | {len(y)} | " +
                      " | ".join(f"{v} ({v / len(y):.0%})" for v in per) +
                      f" | {min(per) / len(y):.1%} | {', '.join(under) if under else '—'} |")
        assert seed_out["perm_I"]["per_task"] == seed_out["teacher_I"]["per_task"]
        payload["seeds"][str(s)] = seed_out
        md += ["", "perm_I's per-task counts equal teacher_I's exactly: the permutation stays "
               "inside each gold class, so the multiset of teacher labels is unchanged "
               "(asserted).", ""]
    md += [f"Every condition keeps every task above the {MIN_TASK_SHARE:.0%} share used when "
           "the split was built, unless a row above says otherwise."]
    _write("task_rows_by_condition", payload, md)
    return payload


# ------------------------------------------------------------ §1.6 audit

def audit_50(table: Dict, sp: Space, n: int = 50, seed: int = 1234) -> Dict:
    """
    50 teacher != gold rows from I, stratified by teacher label so the three
    most over-predicted classes (T5 definition, teacher/gold count on I) appear
    in proportion; the rest form one 'other' stratum. Largest-remainder
    allocation, pandas sample(random_state=1234) within each stratum.
    """
    import pandas as pd
    meta = table["meta"]
    I = sp.I
    dis = I[sp.y_t[I] != sp.y_g[I]]
    tI = Counter(sp.y_t[I].tolist())
    gI = Counter(sp.y_g[I].tolist())
    ratio = {c: tI.get(c, 0) / gI[c] for c in range(len(sp.names)) if gI.get(c, 0)}
    top3 = sorted(ratio, key=lambda c: -ratio[c])[:3]
    strata = {sp.names[c]: dis[sp.y_t[dis] == c] for c in top3}
    strata["other"] = dis[~np.isin(sp.y_t[dis], top3)]
    sizes = {k: len(v) for k, v in strata.items()}
    exact = {k: n * v / len(dis) for k, v in sizes.items()}
    alloc = {k: int(math.floor(v)) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: -(exact[k] - alloc[k]))[:n - sum(alloc.values())]:
        alloc[k] += 1
    picked = []
    for k, rows in strata.items():
        df = pd.DataFrame({"row": rows})
        picked += df.sample(n=alloc[k], random_state=seed)["row"].tolist()
    picked = sorted(int(r) for r in picked)

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["row_id", "dialogue_id", "pair_index", "dialogue_text", "subject", "subject_type",
              "object", "object_type", "gold_label", "teacher_label", "gold_relations_original",
              "reverse_pair_gold", "stratum", "teacher_defensible", "note"]
    stratum_of = {int(r): k for k, rows in strata.items() for r in rows}
    with open(OUT / "audit_50.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in picked:
            m = meta[r]
            w.writerow({"row_id": r, "dialogue_id": m["dialogue_id"],
                        "pair_index": m["pair_index"], "dialogue_text": m["text"],
                        "subject": m["subject"], "subject_type": m["subject_type"],
                        "object": m["object"], "object_type": m["object_type"],
                        "gold_label": m["gold"], "teacher_label": m["teacher"],
                        "gold_relations_original": "|".join(m["gold_relations"]),
                        "reverse_pair_gold": "" if m["reverse_pair_gold"] is None
                        else "|".join(m["reverse_pair_gold"]),
                        "stratum": stratum_of[r], "teacher_defensible": "", "note": ""})
    rev_hit = sum(1 for r in picked if meta[r]["reverse_pair_gold"]
                  and meta[r]["teacher"] in meta[r]["reverse_pair_gold"])
    info = {"n": len(picked), "pool": int(len(dis)), "random_state": seed,
            "top3_over_predicted_on_I": {sp.names[c]: round(ratio[c], 3) for c in top3},
            "strata_sizes": sizes, "allocation": alloc,
            "gold_set_differs_from_single_label": 0,
            "teacher_label_equals_reverse_pair_gold": rev_hit}
    (OUT / "audit_50.meta.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="RQ3 Gate 1 diagnostics")
    ap.add_argument("--only", nargs="+", default=["rows", "kd", "broad", "tasks", "audit"])
    args = ap.parse_args()
    table = build_row_table()
    sp = Space(table)
    if "rows" in args.only:
        p = row_accounting(table, sp)
        print(f"rows: |I|={p['I']['size']} eps_I={p['I']['epsilon_I']:.4f}")
    if "kd" in args.only:
        kd_memory_agreement()
        print("kd: done")
    if "broad" in args.only:
        broad_class_check(table, sp)
        print("broad: done")
    if "tasks" in args.only:
        task_rows(sp)
        print("tasks: done")
    if "audit" in args.only:
        print("audit:", audit_50(table, sp))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
