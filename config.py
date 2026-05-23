"""
config.py

Global configuration for FairSurvival_CreditRisk.
All parameters are defined here and imported by experiments and src modules.
Override per-experiment via experiments/configs/*.yaml or CLI arguments.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Fairness penalty coefficients ────────────────────────────────────────────
BETA  = 0.0    # M_STATIC  EO penalty weight
ALPHA = 0.0    # M_DYNAMIC EO penalty weight
GAMMA = 0.0    # M_PP      EO penalty weight

# ── EO penalty mode ───────────────────────────────────────────────────────────
# Options: "mean" | "weighted" | "trend_aware" | "weighted+trend"
EO_MODE_D = "trend_aware"   # dynamic model
EO_MODE_P = "trend_aware"   # person-period model

# ── Time schedule mode (alpha_schedule) ──────────────────────────────────────
# Options: "flat" | "decay" | "growth" | "u_shaped" | "early_focus"
SCHEDULE_MODE_D = "decay"
SCHEDULE_MODE_P = "decay"

# ── MLP architecture ──────────────────────────────────────────────────────────
HIDDEN1  = 64
HIDDEN2  = 32
DROPOUT  = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 200
PATIENCE     = 30
MIN_LR       = 1e-5
PW_CLIP      = 20.0    # cap on pos_weight

# ── Cross-validation ──────────────────────────────────────────────────────────
N_FOLDS = 5

# ── Simulation dataset ────────────────────────────────────────────────────────
SCENARIO         = "fair"          # "fair" | "unfair"
HORIZON          = 3               # prediction horizon (periods)
LANDMARKS        = list(range(1, 13))
N_TEST_LANDMARKS = 12

# Column names in simulation dataset
ID_COL    = "ID"
TIME_COL  = "Time"
EVENT_COL = "Event"
SENS_COL  = "S"

STATIC_COLS  = ["X1", "X2"]
TVC_COLS     = ["X3", "X4", "X6"]
CAT_COLS     = ["X5"]
ALL_NUM_COLS = STATIC_COLS + TVC_COLS

ATTR_NAME   = "S"
GROUP_NAMES = {0: "S0", 1: "S1"}

# ── FNMA real dataset ─────────────────────────────────────────────────────────
HORIZON_MONTHS = 12
LANDMARKS_FNMA = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48]

STATIC_COLS_FNMA = [
    "credit_score", "original_dti", "original_ltv",
    "interest_rate", "loan_term", "num_borrowers",
]
TVC_COLS_FNMA = [
    "current_upb", "current_interest_rate", "estimated_ltv", "bd_pct",
]
CAT_COLS_FNMA = ["occupancy_status_orig", "loan_purpose_orig"]

FAIR_ATTR   = "SEX"    # "SEX" | "RACE" | "AGE"
GROUP_NAMES_FNMA = {
    "SEX":  {0: "Male",        1: "Female"},
    "RACE": {0: "White/Asian", 1: "Black/Indian"},
    "AGE":  {0: "Old",         1: "Young"},
}

# ── Grid search sweep values ──────────────────────────────────────────────────
GRID_BETAS  = [0.0, 0.3, 0.5, 0.7, 1.0]
GRID_ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
GRID_GAMMAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

# ── W&B ───────────────────────────────────────────────────────────────────────
USE_WANDB    = False
WANDB_ENTITY = "alessia-ciccaglione02-"
WANDB_PROJECT = "ThesisFairness"
