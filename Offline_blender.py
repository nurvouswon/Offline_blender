# offline_tuner.py
# ============================================================
# 🧪 MLB HR/TB/RBI Offline Tuner (Streamlit Cloud–friendly)
# - Inputs: Event-level data (+ hr_outcome), Season batter profile, Season pitcher profile
# - No TODAY file, No weather dependence
# - Base meta-ensemble (XGB/LGB/CB → LR) with early stopping
# - Isotonic calibration + top-K temp tuning (K=30 fixed)
# - Weather-free overlay (batter/pitcher/platoon/park) from profiles
# - Auto-derive TB≥2 (tb_ge_2) and RBI≥2 (rbi_ge_2) if missing
# - Multiplier Tuner (overlay exponents) — per target
# - Blended Tuner (prob + overlay + ranker + RRF – penalty) — per target
# - 5 folds, no plots, tuned for Streamlit Cloud memory/CPU
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, time, gc
from datetime import timedelta
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.special import logit, expit

# ============= UI HEADER =============
st.set_page_config(page_title="🧪 Offline Tuner (HR / TB≥2 / RBI≥2)", layout="wide")
st.title("🧪 Offline Tuner — HR / TB≥2 / RBI≥2 (Cloud-friendly)")

# ============= CONSTANTS (Cloud-safe) =============
TOPK = 30
N_FOLDS = 5
SEEDS = [42, 101, 202, 404]
SAMPLES_MULT = 4000   # overlay exponent search
SAMPLES_BLEND = 6000  # final blend search

# ============= SESSION DEFAULTS =============
DEFAULT_BLEND = dict(
    w_prob=0.30,
    w_overlay=0.20,
    w_ranker=0.20,
    w_rrf=0.10,
    w_penalty=0.20,
)
DEFAULT_MULT = dict(
    a_batter=0.80,
    b_pitcher=0.80,
    c_platoon=0.60,
    d_park=0.40,
)
if "best_by_target" not in st.session_state:
    st.session_state.best_by_target = {
        "HR":   {"blend": DEFAULT_BLEND.copy(), "mult": DEFAULT_MULT.copy()},
        "TB2":  {"blend": DEFAULT_BLEND.copy(), "mult": DEFAULT_MULT.copy()},
        "RBI2": {"blend": DEFAULT_BLEND.copy(), "mult": DEFAULT_MULT.copy()},
    }

# ============= HELPERS =============
@st.cache_data(show_spinner=False, max_entries=4)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "O":
            try:
                # Try numeric; if it doesn't coerce, keep as object
                v = pd.to_numeric(df[c], errors="coerce")
                # Keep as numeric only if many values actually convert
                if v.notna().sum() >= max(5, int(0.5 * len(v))):
                    df[c] = v
            except Exception:
                pass
    return df

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def embargo_time_splits(dates_series, n_splits=N_FOLDS, embargo_days=1):
    dates = pd.to_datetime(dates_series).reset_index(drop=True)
    u_days = pd.Series(dates.dt.floor("D")).dropna().unique()
    u_days = pd.to_datetime(u_days)
    day_folds = np.array_split(np.arange(len(u_days)), n_splits)
    folds = []
    for k in range(n_splits):
        va_days_idx = day_folds[k]
        va_days = set(u_days[va_days_idx])
        if len(va_days):
            min_va = min(va_days)
            embargo_mask = (dates.dt.floor("D") >= (min_va - pd.Timedelta(days=embargo_days))) & (dates.dt.floor("D") < min_va)
        else:
            embargo_mask = pd.Series(False, index=dates.index)
        va_mask = dates.dt.floor("D").isin(va_days)
        tr_mask = ~va_mask & ~embargo_mask
        tr_idx = np.where(tr_mask.values)[0]
        va_idx = np.where(va_mask.values)[0]
        if len(tr_idx) and len(va_idx):
            folds.append((tr_idx, va_idx))
    return folds

def tune_temperature_for_topk(p_oof, y, K=TOPK, T_grid=np.linspace(0.8, 1.6, 17)):
    y = np.asarray(y).astype(int)
    best_T, best_hits = 1.0, -1
    logits = logit(np.clip(p_oof, 1e-6, 1-1e-6))
    for T in T_grid:
        p_adj = expit(logits * T)
        order = np.argsort(-p_adj)
        hits = int(y[order][:K].sum())
        if hits > best_hits:
            best_hits, best_T = hits, float(T)
    return best_T

def _rank_desc(x):
    x = np.asarray(x)
    return pd.Series(-x).rank(method="min").astype(int).values

# ============= SIDEBAR — UPLOADS & KEYS =============
with st.sidebar:
    st.header("📤 Upload Data")
    ev_file  = st.file_uploader("Event-level (CSV/Parquet)", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Batter Profile (CSV)", type=["csv"], key="bat")
    pit_file = st.file_uploader("Pitcher Profile (CSV)", type=["csv"], key="pit")

if ev_file is None or bat_file is None or pit_file is None:
    st.info("Upload event-level + batter profile + pitcher profile to begin.")
    st.stop()

# ============= LOAD & LIGHT CLEANING =============
with st.spinner("Loading files..."):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file)
    pit = _read_any(pit_file)

ev  = dedup_columns(_safe_fix_types(ev))
bat = dedup_columns(_safe_fix_types(bat))
pit = dedup_columns(_safe_fix_types(pit))

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ============= KEYS (default to batter_id/pitcher_id) =============
def _default_key(options, prefer):
    if prefer in options: return prefer
    # fallback: first column
    return options[0] if len(options) else None

ev_batter_key = _default_key(list(ev.columns), "batter_id") if "batter_id" in ev.columns else _default_key(list(ev.columns), "batter")
ev_pitcher_key = _default_key(list(ev.columns), "pitcher_id") if "pitcher_id" in ev.columns else _default_key(list(ev.columns), "pitcher")
bat_key = _default_key(list(bat.columns), "batter_id")
pit_key = _default_key(list(pit.columns), "pitcher_id")

with st.expander("🔗 Join Keys (adjust only if needed)"):
    ev_batter_key = st.selectbox("Event → Batter key", options=sorted(ev.columns), index=sorted(ev.columns).index(ev_batter_key))
    bat_key       = st.selectbox("Batter profile key", options=sorted(bat.columns), index=sorted(bat.columns).index(bat_key))
    ev_pitcher_key= st.selectbox("Event → Pitcher key", options=sorted(ev.columns), index=sorted(ev.columns).index(ev_pitcher_key))
    pit_key       = st.selectbox("Pitcher profile key", options=sorted(pit.columns), index=sorted(pit.columns).index(pit_key))

# ============= MERGE (robust like prediction app) =============
def _as_str(s):
    return s.astype(str).str.strip().fillna("")

def _safe_merge_profiles(ev, bat, pit, ev_batter_key, bat_key, ev_pitcher_key, pit_key):
    ev = ev.copy()
    bat_pref = bat.add_prefix("batprof_").copy()
    pit_pref = pit.add_prefix("pitprof_").copy()

    # Restore the key names post-prefix so we can merge
    bat_pref = bat_pref.rename(columns={f"batprof_{bat_key}": "bat_key_merge"})
    pit_pref = pit_pref.rename(columns={f"pitprof_{pit_key}": "pit_key_merge"})

    # Create merge keys as string to avoid int/object mismatch
    ev["bat_key_merge"] = _as_str(ev[ev_batter_key]) if ev_batter_key in ev.columns else _as_str(ev.iloc[:,0])
    ev["pit_key_merge"] = _as_str(ev[ev_pitcher_key]) if ev_pitcher_key in ev.columns else _as_str(ev.iloc[:,0])
    bat_pref["bat_key_merge"] = _as_str(bat_pref["bat_key_merge"])
    pit_pref["pit_key_merge"] = _as_str(pit_pref["pit_key_merge"])

    # Merge left
    ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
    ev = ev.merge(pit_pref, on="pit_key_merge", how="left")
    return ev

with st.spinner("Merging profiles into event-level…"):
    ev = _safe_merge_profiles(ev, bat, pit, ev_batter_key, bat_key, ev_pitcher_key, pit_key)
st.success("✅ Profiles merged.")

# ============= DROP OBVIOUS LEAKS (same as main app family) =============
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

# ============= DERIVE TB≥2 & RBI≥2 IF MISSING =============
def _derive_tb_rbi_labels(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()

    # Choose event text col
    evt = ev["events_clean"] if "events_clean" in ev.columns else ev.get("events", pd.Series("", index=ev.index))
    evt = evt.astype(str).str.lower().fillna("")

    # Group per (game, batter)
    grp_cols = []
    if "game_pk" in ev.columns: grp_cols.append("game_pk")
    if "batter_id" in ev.columns: grp_cols.append("batter_id")
    if not grp_cols:
        if "game_date" in ev.columns and "batter_id" in ev.columns:
            grp_cols = ["game_date","batter_id"]
        else:
            grp_cols = ["batter_id"]

    # TB≥2
    if "tb_ge_2" not in ev.columns:
        tb_map = {"home_run":4, "triple":3, "double":2, "single":1}
        tb_pa = np.zeros(len(ev), dtype=np.int16)
        for k, v in tb_map.items():
            tb_pa = np.where(evt.str.contains(k), v, tb_pa)
        ev["__tb_pa"] = tb_pa
        tb_game = ev.groupby(grp_cols, dropna=False)["__tb_pa"].sum().rename("__tb_game")
        ev = ev.merge(tb_game, on=grp_cols, how="left")
        ev["tb_ge_2"] = (ev["__tb_game"] >= 2).astype(int)
        ev.drop(columns=["__tb_pa","__tb_game"], inplace=True, errors="ignore")

    # RBI≥2
    if "rbi_ge_2" not in ev.columns:
        if "rbi" in ev.columns:
            rbi_pa = pd.to_numeric(ev["rbi"], errors="coerce").fillna(0).astype(int)
        else:
            # Conservative: RBIs from HR only using runners_on
            if set(["on_1b","on_2b","on_3b"]).issubset(ev.columns):
                runners_on = ev[["on_1b","on_2b","on_3b"]].notna().sum(axis=1).astype(int)
            else:
                runners_on = pd.Series(0, index=ev.index, dtype=int)
            is_hr = evt.str.contains("home_run")
            rbi_pa = (is_hr.astype(int) * (1 + runners_on)).astype(int)

        ev["__rbi_pa"] = rbi_pa
        rbi_game = ev.groupby(grp_cols, dropna=False)["__rbi_pa"].sum().rename("__rbi_game")
        ev = ev.merge(rbi_game, on=grp_cols, how="left")
        ev["rbi_ge_2"] = (ev["__rbi_game"] >= 2).astype(int)
        ev.drop(columns=["__rbi_pa","__rbi_game"], inplace=True, errors="ignore")

    return ev

with st.spinner("Deriving TB≥2 and RBI≥2 labels (if missing)…"):
    ev = _derive_tb_rbi_labels(ev)
st.success("✅ Derived TB≥2 / RBI≥2 (if needed).")

# ============= TARGETS & DATES =============
if "hr_outcome" not in ev.columns:
    st.error("Your event-level file must include hr_outcome (0/1) for the HR target.")
    st.stop()

y_hr   = ev["hr_outcome"].fillna(0).astype(int)
y_tb2  = ev["tb_ge_2"].fillna(0).astype(int)
y_rbi2 = ev["rbi_ge_2"].fillna(0).astype(int)

dates_col = "game_date" if "game_date" in ev.columns else None
if dates_col:
    dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# ============= BASIC FEATURES (numeric only, weather-free) =============
num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

# ============= TRAIN BASE MODELS (OOF) ON HR (used for all targets) =============
st.subheader("⚙️ Training Base Models (with early stopping)")
folds = embargo_time_splits(dates, n_splits=N_FOLDS, embargo_days=1)

P_xgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_lgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_cat_oof = np.zeros(len(y_hr), dtype=np.float32)
ranker_oof = np.zeros(len(y_hr), dtype=np.float32)

fold_times = []
for fi, (tr_idx, va_idx) in enumerate(folds):
    t0 = time.time()
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr, y_va = y_hr.iloc[tr_idx], y_hr.iloc[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for sd in SEEDS:
        spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=900, max_depth=6, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1600, learning_rate=0.035, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=sd
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=1800, depth=7, learning_rate=0.035, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, spw], od_type="Iter", od_wait=50,
            verbose=0, thread_count=1, random_seed=sd
        )

        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds_xgb.append(xgb_clf.predict_proba(X_va)[:,1])
        preds_lgb.append(lgb_clf.predict_proba(X_va)[:,1])
        preds_cat.append(cat_clf.predict_proba(X_va)[:,1])

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

    # Day-wise ranker on HR labels (optional but helpful)
    days_tr = pd.to_datetime(dates.iloc[tr_idx]).dt.floor("D")
    days_va = pd.to_datetime(dates.iloc[va_idx]).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    try:
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=700, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=_groups_from_days(days_tr), eval_set=[(X_va, y_va)], eval_group=[_groups_from_days(days_va)],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
    except Exception:
        # if ranks can’t be computed reliably (single day etc.), leave zeros
        pass

    dt = time.time() - t0
    fold_times.append(dt)
    st.write(f"Fold {fi+1}/{len(folds)} finished in {timedelta(seconds=int(dt))}")

# ============= META STACKER + CALIBRATION (on HR OOF) =============
st.subheader("🧮 Meta Stacker + Calibration (trained on HR OOF)")
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y_hr.values)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"OOF AUC (HR meta): {roc_auc_score(y_hr, oof_meta):.4f} | OOF LogLoss: {log_loss(y_hr, oof_meta):.4f}")

ir = IsotonicRegression(out_of_bounds="clip")
y_oof_iso = ir.fit_transform(oof_meta, y_hr.values)
Tbest = tune_temperature_for_topk(y_oof_iso, y_hr.values, K=TOPK, T_grid=np.linspace(0.8, 1.6, 17))
logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
p_base_hr = expit(logits_oof * Tbest)  # calibrated, temp-tuned HR OOF

# We’ll reuse p_base_hr as the calibrated base signal for TB/RBI too (shared model capacity).
# (If you want, we can add simple target-specific reweighting later.)

# ============= WEATHER-FREE OVERLAY FROM PROFILES =============
# Column knobs (adjust names to your profile schema)
bat_cols = dict(
    barrel="batprof_barrel_rate",
    fb="batprof_fb_rate",
    pull="batprof_pull_rate",
    hr_pa="batprof_hr_per_pa"
)
pit_cols = dict(
    brl_allowed="pitprof_barrel_rate_allowed",
    fb="pitprof_fb_rate",
    bb="pitprof_bb_rate",
    xwoba_con="pitprof_xwoba_con"
)
platoon_cols = dict(
    batter_vsL="batprof_hr_pa_vsL",
    batter_vsR="batprof_hr_pa_vsR",
    pitcher_vsL="pitprof_hr_pa_vsL",
    pitcher_vsR="pitprof_hr_pa_vsR",
    stand="stand",
    batter_hand="batter_hand",
    p_throws="pitcher_hand"
)
park_cols = dict(
    park_factor="park_hr_rate",
)

def _platoon_factor_row(row):
    bhand = str(row.get(platoon_cols["stand"], row.get(platoon_cols["batter_hand"], "R"))).upper()
    phand = str(row.get(platoon_cols["p_throws"], "R")).upper()
    if bhand == "L":
        b_rate = row.get(platoon_cols["batter_vsR"], np.nan)
        p_rate = row.get(platoon_cols["pitcher_vsL"], np.nan)
    else:
        b_rate = row.get(platoon_cols["batter_vsL"], np.nan)
        p_rate = row.get(platoon_cols["pitcher_vsR"], np.nan)
    f = 1.0
    if pd.notnull(b_rate):
        if b_rate >= 0.05: f *= 1.05
        elif b_rate <= 0.02: f *= 0.98
    if pd.notnull(p_rate):
        if p_rate >= 0.05: f *= 1.04
        elif p_rate <= 0.02: f *= 0.99
    if (bhand == "L" and phand == "R") or (bhand == "R" and phand == "L"):
        f *= 1.01
    return float(np.clip(f, 0.94, 1.10))

def _batter_factor_row(row):
    f = 1.0
    brl = row.get(bat_cols["barrel"], np.nan)
    fb  = row.get(bat_cols["fb"], np.nan)
    pull= row.get(bat_cols["pull"], np.nan)
    hrp = row.get(bat_cols["hr_pa"], np.nan)
    if pd.notnull(brl):
        if brl >= 0.12: f *= 1.06
        elif brl >= 0.09: f *= 1.03
    if pd.notnull(fb) and fb >= 0.22: f *= 1.02
    if pd.notnull(pull) and pull >= 0.35: f *= 1.02
    if pd.notnull(hrp) and hrp >= 0.06: f *= 1.03
    return float(np.clip(f, 0.95, 1.10))

def _pitcher_factor_row(row):
    f = 1.0
    brlA = row.get(pit_cols["brl_allowed"], np.nan)
    fb   = row.get(pit_cols["fb"], np.nan)
    bb   = row.get(pit_cols["bb"], np.nan)
    xcon = row.get(pit_cols["xwoba_con"], np.nan)
    if pd.notnull(brlA):
        if brlA >= 0.11: f *= 1.05
        elif brlA >= 0.09: f *= 1.03
    if pd.notnull(fb) and fb >= 0.40: f *= 1.03
    if pd.notnull(bb) and bb >= 0.10: f *= 1.02
    if pd.notnull(xcon):
        if xcon >= 0.40: f *= 1.04
        elif xcon >= 0.36: f *= 1.02
    return float(np.clip(f, 0.94, 1.12))

def _park_factor_row(row):
    pf = row.get(park_cols["park_factor"], np.nan)
    try:
        pf = float(pf)
        return float(np.clip(pf, 0.85, 1.20))
    except Exception:
        return 1.0

with st.spinner("Computing overlay components (profiles only)…"):
    bf = ev.apply(_batter_factor_row, axis=1).astype(np.float32)
    pf = ev.apply(_pitcher_factor_row, axis=1).astype(np.float32)
    pltf = ev.apply(_platoon_factor_row, axis=1).astype(np.float32)
    pkf = ev.apply(_park_factor_row, axis=1).astype(np.float32)

# ============= BUILD RRF + DISAGREEMENT OFF OOF (once) =============
disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty = np.clip(zscore(disagree_std), 0, 3)

r_prob    = _rank_desc(p_base_hr)
r_ranker  = _rank_desc(zscore(ranker_oof))
# overlay will be recomputed per-target after exponent tuning
k_rrf = 60.0

# ============= TUNERS (per target) =============
def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

def _hits_at_k(y_true, s, K=TOPK):
    ord_idx = np.argsort(-s)
    return int(np.sum(y_true[ord_idx][:K]))

def _dcg_at_k(rels, K):
    rels = np.asarray(rels)[:K]
    if rels.size == 0: return 0.0
    discounts = 1.0/np.log2(np.arange(2, 2+len(rels)))
    return float(np.sum(rels*discounts))

def _ndcg_at_k(y_true, s, K):
    ord_idx = np.argsort(-s)
    rel_sorted = y_true[ord_idx]
    dcg = _dcg_at_k(rel_sorted, K)
    ideal = np.sort(y_true)[::-1]
    idcg = _dcg_at_k(ideal, K)
    return (dcg/idcg) if idcg > 0 else 0.0

def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

def run_all_tuners_for_target(y_target: pd.Series, target_name: str):
    # Start from currently saved defaults for this target
    use_mult = st.session_state.best_by_target[target_name]["mult"].copy()
    use_blend= st.session_state.best_by_target[target_name]["blend"].copy()

    # (A) Multiplier Tuner (overlay exponents)
    rng = np.random.default_rng(123)
    best_key = None; best_res = None
    for _ in range(SAMPLES_MULT):
        a_b = float(rng.uniform(0.2, 1.6))
        b_p = float(rng.uniform(0.2, 1.6))
        c_pl= float(rng.uniform(0.0, 1.2))
        d_pk= float(rng.uniform(0.0, 1.2))
        ov = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
        # Evaluate via calibrated base prob + overlay (as a proxy)
        eval_score = expit(logit(np.clip(p_base_hr,1e-6,1-1e-6)) + np.log(ov+1e-9))
        hK = _hits_at_k(y_target.values, eval_score, TOPK)
        nd = _ndcg_at_k(y_target.values, eval_score, 30)
        key = (hK, nd)
        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk, HitsAtK=hK, NDCG30=nd)
    if best_res:
        use_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}

    # Final overlay & logs for this target
    overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
    log_overlay = np.log(overlay + 1e-9)

    # (B) Build RRF (needs overlay ranks)
    r_overlay = _rank_desc(overlay)
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)
    ranker_z = zscore(ranker_oof)
    logit_p = logit(np.clip(p_base_hr, 1e-6, 1-1e-6))

    # (C) Blended Tuner
    rng = np.random.default_rng(777)
    best_tuple = None; best_row = None
    for _ in range(SAMPLES_BLEND):
        w = rng.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
        hK = _hits_at_k(y_target.values, s, TOPK)
        h30 = _hits_at_k(y_target.values, s, 30)
        nd = _ndcg_at_k(y_target.values, s, 30)
        tup = (hK, nd, h30)
        if (best_tuple is None) or (tup > best_tuple):
            best_tuple = tup
            best_row = dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]),
                            w_rrf=float(w[3]), w_penalty=float(w[4]),
                            HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd))
    if best_row:
        use_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}

    # Final blended OOF (diagnostic)
    final_oof = blend_with_weights(
        use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"], use_blend["w_rrf"], use_blend["w_penalty"],
        logit_p, log_overlay, ranker_z, rrf_z, dis_penalty
    )

    # Persist
    st.session_state.best_by_target[target_name]["mult"]  = use_mult.copy()
    st.session_state.best_by_target[target_name]["blend"] = use_blend.copy()

    # Report
    out = {
        "target": target_name,
        "AUC_final_blend": float(roc_auc_score(y_target, final_oof)) if len(np.unique(y_target))>1 else None,
        "Hits@30": int(_hits_at_k(y_target.values, final_oof, TOPK)),
        "NDCG@30": float(_ndcg_at_k(y_target.values, final_oof, 30)),
        "overlay_exponents": use_mult.copy(),
        "blend_weights": use_blend.copy(),
    }
    return out

# ============= RUN TUNERS (HR, TB≥2, RBI≥2) =============
st.subheader("🔧 Running Tuners (HR / TB≥2 / RBI≥2)")
with st.spinner("Searching overlay exponents and blend weights (cloud-safe)…"):
    report_hr   = run_all_tuners_for_target(y_hr,   "HR")
    report_tb2  = run_all_tuners_for_target(y_tb2,  "TB2")
    report_rbi2 = run_all_tuners_for_target(y_rbi2, "RBI2")
st.success("✅ Tuners complete.")

# ============= REPORTS (concise) =============
st.markdown("### 📋 Results Summary")
st.json(report_hr, expanded=False)
st.json(report_tb2, expanded=False)
st.json(report_rbi2, expanded=False)

# ============= EXPORT =============
export_payload = {
    "notes": "Per-target overlay exponents and blend weights (weather-free), tuned on event-level OOF using HR-trained meta ensemble.",
    "K": TOPK,
    "folds": N_FOLDS,
    "targets": {
        "HR":   st.session_state.best_by_target["HR"],
        "TB2":  st.session_state.best_by_target["TB2"],
        "RBI2": st.session_state.best_by_target["RBI2"],
    }
}
st.download_button(
    "⬇️ Download Best Weights (JSON)",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json",
)

st.caption("Exported weights can be plugged into your main prediction app: use the per-target overlay exponents and blend weights to compute final ranked probabilities.")
