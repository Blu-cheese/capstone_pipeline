"""
Central configuration for every continual-learning run.

METHOD_A_SPEC_V2 §Global config: unstated choices are how regimes end up
differing by accident. Every module reads from here; nothing hard-codes a
hyperparameter locally.

CPU only throughout. No CUDA, no MPS, no device-selection logic.
"""

from pathlib import Path

# --------------------------------------------------------------- paths

FEATURE_DIR = Path("data/features")
SPLIT_DIR = Path("data/task_splits")
EVAL_DIR = Path("data/eval")
RUN_DIR = Path("runs/cl")
RESULTS_DIR = Path("results")

CORPUS = "dialogre_train"
ENCODER_SLUG = "minilm-l6-v2"

# --------------------------------------------------------------- model

FEATURE_DIM = 1152          # 384 x 3 (window | subject | object), frozen MiniLM
HIDDEN_DIM = 256
DROPOUT = 0.2

# N_CLASSES is deliberately absent. It is read from the frozen task-split file
# (see task_split.class_index). It has already changed twice - 20 in the stale
# meeting taxonomy, 18 before the multi-label fix, 17 now - and hard-coding it
# is exactly how a stale constant silently corrupts a head.

# ------------------------------------------------------------ training
#
# Identical across all regimes and both tracks. V2: "If one regime needs
# different values, that is a finding to report, not a knob to turn quietly."

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 32
# Raised 30 -> 60 in Phase 5 pre-flight (PHASE5_SPEC 0.1), then 60 -> 120
# before Phase 6: across all 18 task-trainings in the naive seed sweep, 17
# early-stopped and exactly one hit the 60 cap (teacher/seed 1234/task 0,
# best dev @ ep 59). Early-stopped trajectories are invariant under a higher
# cap, so the raise changes that single run and nothing else; both joint
# baselines stopped at eps 32/26 and are untouched. Patience = 5 governs.
# STOP RULE: if a run hits 120 still improving, that is bouncing, not
# under-capping - fix patience, never raise the cap again.
MAX_EPOCHS_PER_TASK = 120
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"   # on D_dev_t, task t only
RESTORE_BEST_WEIGHTS = True

# Fresh optimizer at every task boundary. Carrying Adam state from task t-1
# changes how fast task t overwrites it, which silently confounds the
# forgetting measurement.
FRESH_OPTIMIZER_PER_TASK = True

SEEDS = [1234, 1, 2]        # each yields a different task sequence AND init

# ---------------------------------------------------------------- FKD
#
# Zhao, Cui & Hu, "Improving Continual Relation Extraction by Distinguishing
# Analogous Semantics", ACL 2023, pp. 1162-1175. Appendix B.
# Equations and extracted values: docs/PAPER_NOTES.md §1.
#
# WHY THE TACRED ROW, NOT FEWREL'S:
#
#   The paper reports two hyperparameter rows, one per dataset:
#       FewRel : gamma=1.25, lambda_2=0.5/1.1, tau_2=0.5
#       TACRED : gamma=2.00, lambda_2=0.7,     tau_2=0.5
#
#   FewRel is balanced - 80 relations x 700 samples each. TACRED is
#   class-imbalanced with far fewer samples per relation. DialogRE as we use
#   it is emphatically the latter: 17 classes ranging from 15 to ~200 training
#   examples, 879 total on the teacher track. TACRED is the closer analogue,
#   so we take its row rather than averaging or picking arbitrarily.
#
#   tau_2 = 0.5 is fixed across both their datasets, so it carries no choice.
#
#   These are the paper's tuned values, adopted as defaults - NOT tuned by us.
#   If FKD underperforms, report it. Tuning lambda until FKD wins is
#   fabrication (CLAUDE.md §8).

FKD_TAU2 = 0.5      # temperature in the prototype-similarity softmax (Eq. 10)
FKD_GAMMA = 2.0     # focal exponent (Eq. 11), TACRED value
FKD_LAMBDA = 0.7    # weight on L_fkd in L_replay, TACRED lambda_2 (linear head)

# Search ranges, should tuning ever be justified and declared:
FKD_GAMMA_RANGE = (1.0, 2.0)
FKD_LAMBDA_RANGE = (0.5, 1.5)

# We implement the FKD component in isolation: linear head only, so
# L_replay = L_cls + lambda * L_fkd rather than the paper's Eq. 14, which
# combines contrastive and linear FKD terms. Declare this in the report.

# ------------------------------------------------------------- replay

REPLAY_MEMORY_PER_RELATION = 10   # Zhao et al. main-experiment memory size
REPLAY_SELECTION = "kmeans"       # cre2.pdf Algorithm 1: samples nearest centroids
REPLAY_NEW_TO_BUFFER_RATIO = 0.5  # 50/50 new:buffer in each batch


def summary() -> dict:
    """Everything a run should record in its config.json."""
    return {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs_per_task": MAX_EPOCHS_PER_TASK,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_metric": EARLY_STOPPING_METRIC,
        "fresh_optimizer_per_task": FRESH_OPTIMIZER_PER_TASK,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "feature_dim": FEATURE_DIM,
        "fkd": {"tau2": FKD_TAU2, "gamma": FKD_GAMMA, "lambda": FKD_LAMBDA,
                "source": "Zhao et al. ACL 2023 Appendix B, TACRED row"},
        "replay": {"memory_per_relation": REPLAY_MEMORY_PER_RELATION,
                   "selection": REPLAY_SELECTION,
                   "new_to_buffer_ratio": REPLAY_NEW_TO_BUFFER_RATIO},
    }
