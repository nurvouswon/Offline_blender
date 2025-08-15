# offline_tuner_cloud.py
# ============================================================
# 🧪 MLB HR Offline Tuner (Cloud-Optimized, No TODAY CSV)
# - Inputs: Event-level data (+ hr_outcome), Season-long Batter Profile, Season-long Pitcher Profile
# - Base meta-ensemble (XGB/LGB/CB → LR) with early stopping
# - Isotonic calibration + Top-30 temperature tuning
# - Weather-free overlay from profiles (batter/pitcher/platoon/park)
# - Multiplier Tuner (overlay exponents) → ALWAYS ON
# - Blended Tuner (weights: prob, overlay, ranker, rrf, penalty) → ALWAYS ON
# - Day-wise LambdaRank (honors time budget; soft-skip if needed)
# - Cloud speed fixes: feature cap, negative downsampling, lean models, global time budget
# - Progress bars + ETA; no plots
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import time, json, gc
from datetime import timedelta
from collections import defaultdict

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.special import logit, expit

# -------------------- Fixed settings for Streamlit Cloud --------------------
N_FOLDS = 5
TOPK = 30
SEEDS = [42]                 # single-seed bag for speed
N_JOBS = 1
TIME_BUDGET_SEC = 9 * 60     # ~9 minutes soft budget for base+ranker

# Model sizes (lean, fast early-stopping)
XGB_N_EST, XGB_LR, XGB_ES = 180, 0.07, 15
LGB_N_EST, LGB_LR, LGB_ES = 320, 0.07, 15
CAT_ITERS, CAT_LR, CAT_ES  = 450, 0.07, 20
RANK_N_EST, RANK_LR, RANK_ES = 200, 0.07, 15

# Speed helpers
FEATURE_CAP = 300      # keep top-N numeric by variance
NEG_DOWNSAMPLE = 15    # max negative:positive ratio per fold

# Defaults (used if we can’t improve)
DEFAULT_BLEND = dict(w_prob=0.30, w_overlay=0.20, w_ranker=0.20, w_rrf=0.10, w_penalty=0.20)
DEFAULT_MULT  = dict(a_batter=0.80, b_pitcher=0.80, c_platoon=0.60, d_park=0.40)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🧪 MLB HR Offline Tuner (Cloud-Optimized)", layout="wide")
st.title("🧪 MLB HR Offline Tuner (Cloud-Optimized)")

with st.sidebar:
    st.header("📤 Upload Data (3 files)")
    ev_file  = st.file_uploader("Event-level CSV/Parquet (with hr_outcome)", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Season-long Batter Profile CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV", type=["csv"], key="pit")

if not (ev_file and bat_file and pit_file):
    st.info("Upload event-level + batter profile + pitcher profile to begin.")
    st.stop()

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False, max_entries=4)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Gently coerce numeric-like columns; leave IDs/strings intact."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            # try numeric; if mostly numbers it will convert
            try:
                conv = pd.to_numeric(out[c], errors="coerce")
                # adopt if >80% became numeric (others can be IDs)
                if conv.notna().mean() >= 0.8:
                    out[c] = conv
            except Exception:
                pass
    return out

def _to_str_series(s, name):
    """Make a robust string key series; useful for merges with mixed dtypes."""
    return s.astype(str).str.strip().fillna("")

def zscore(a):
    a = np.asarray(a, dtype=np.float64)
    mu = np.nanmean(a); sd = np.nanstd(a) + 1e-9
    return (a - mu) / sd

def embargo_time_splits(dates_series, n_splits=5, embargo_days=1):
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

def _hits_at_k(y_true, s, K=TOPK):
    ord_idx = np.argsort(-s)
    return int(np.sum(np.asarray(y_true)[ord_idx][:K]))

def _dcg_at_k(rels, K):
    rels = np.asarray(rels)[:K]
    if rels.size == 0: return 0.0
    discounts = 1.0/np.log2(np.arange(2, 2+len(rels)))
    return float(np.sum(rels*discounts))

def _ndcg_at_k(y_true, s, K=TOPK):
    ord_idx = np.argsort(-s)
    rel_sorted = np.asarray(y_true)[ord_idx]
    dcg = _dcg_at_k(rel_sorted, K)
    ideal = np.sort(np.asarray(y_true))[::-1]
    idcg = _dcg_at_k(ideal, K)
    return (dcg/idcg) if idcg > 0 else 0.0

# -------------------- Load files --------------------
with st.spinner("Loading files..."):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file)
    pit = _read_any(pit_file)

# Gentle numeric coercion
ev  = _safe_num_cols(ev)
bat = _safe_num_cols(bat)
pit = _safe_num_cols(pit)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# -------------------- Join keys (prediction-app style, but simple) --------------------
st.subheader("🔗 Join Keys (auto-detected sensible defaults)")
# Try to pick sane defaults from known column patterns; user can override if needed later
ev_batter_candidates = [c for c in ev.columns if c.lower() in ("batter_id","batter","batter_key","player_id","playerid")]
ev_pitcher_candidates= [c for c in ev.columns if c.lower() in ("pitcher_id","pitcher","pitcher_key")]
bat_key_candidates   = [c for c in bat.columns if "id" in c.lower() or "player" in c.lower()]
pit_key_candidates   = [c for c in pit.columns if "id" in c.lower() or "player" in c.lower()]

ev_batter_key = ev_batter_candidates[0] if ev_batter_candidates else st.selectbox("Event → Batter key", options=ev.columns, index=0)
bat_key       = bat_key_candidates[0]   if bat_key_candidates else st.selectbox("Batter profile key", options=bat.columns, index=0)
ev_pitcher_key= ev_pitcher_candidates[0] if ev_pitcher_candidates else st.selectbox("Event → Pitcher key", options=ev.columns, index=0)
pit_key       = pit_key_candidates[0]   if pit_key_candidates else st.selectbox("Pitcher profile key", options=pit.columns, index=0)

# Robust string keys to avoid object/int mismatches
ev = ev.copy()
ev["bat_key_merge"] = _to_str_series(ev[ev_batter_key], "bat_key")
ev["pit_key_merge"] = _to_str_series(ev[ev_pitcher_key], "pit_key")

bat_pref = bat.add_prefix("batprof_").copy()
pit_pref = pit.add_prefix("pitprof_").copy()
bat_pref = bat_pref.rename(columns={f"batprof_{bat_key}": "bat_key_merge"})
pit_pref = pit_pref.rename(columns={f"pitprof_{pit_key}": "pit_key_merge"})

with st.spinner("Merging profiles into event-level…"):
    ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
    ev = ev.merge(pit_pref, on="pit_key_merge", how="left")
st.success("✅ Profiles merged.")

# -------------------- Targets & dates --------------------
target_col = "hr_outcome"
if target_col not in ev.columns:
    st.error("Event-level file must contain hr_outcome (0/1).")
    st.stop()
y = ev[target_col].fillna(0).astype(int)

dates_col = "game_date" if "game_date" in ev.columns else None
dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01")) if dates_col else pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# (Optional) derive RBI>=2 and TB>=2 indicators if possible (safe; diagnostic only)
def _derive_rbi_tb(ev_df: pd.DataFrame):
    out = {}
    # RBI: try typical pair (post_bat_score - bat_score) if present
    if {"post_bat_score","bat_score","game_pk","batter"}.issubset(ev_df.columns):
        try:
            tmp = ev_df[["game_pk","batter","post_bat_score","bat_score"]].copy()
            tmp["rbi_evt"] = (pd.to_numeric(tmp["post_bat_score"], errors="coerce") - pd.to_numeric(tmp["bat_score"], errors="coerce")).clip(lower=0).fillna(0).astype(float)
            rbi_game = tmp.groupby(["game_pk","batter"], as_index=False)["rbi_evt"].sum()
            rbi_game["rbi_ge2"] = (rbi_game["rbi_evt"] >= 2).astype(int)
            out["rbi_by_game"] = rbi_game
        except Exception:
            pass
    # Total bases: if events_clean or events available, approximate bases per PA
    # Single=1, Double=2, Triple=3, Home Run=4
    if {"events_clean","events","game_pk","batter"}.intersection(ev_df.columns):
        try:
            ec = "events_clean" if "events_clean" in ev_df.columns else "events"
            tmp = ev_df[["game_pk","batter",ec]].copy()
            m = tmp[ec].astype(str).str.lower()
            tb = np.zeros(len(tmp), dtype=np.int16)
            tb[m.str.contains("home_run")] = 4
            tb[(m.str.contains("double")) & ~m.str.contains("grounded_into_double_play")] = 2
            tb[m.str.contains("triple")] = 3
            # single: if contains 'single' but not double/triple/hr
            is_single = (m.str.contains("single")) & ~(m.str.contains("double") | m.str.contains("triple") | m.str.contains("home_run"))
            tb[is_single.values] = 1
            tmp["tb_evt"] = tb
            tb_game = tmp.groupby(["game_pk","batter"], as_index=False)["tb_evt"].sum()
            tb_game["tb_ge2"] = (tb_game["tb_evt"] >= 2).astype(int)
            out["tb_by_game"] = tb_game
        except Exception:
            pass
    return out

extra_targets = _derive_rbi_tb(ev)
if extra_targets:
    st.info("Derived extra diagnostics (if keys were present): " +
            " • ".join([k for k in extra_targets.keys()]))

# -------------------- Leak-aware column drop (like your prediction app) --------------------
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

# -------------------- Numeric feature matrix --------------------
num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].copy()
X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

# Variance cap for speed
if X_base.shape[1] > FEATURE_CAP:
    var = X_base.var().sort_values(ascending=False)
    keep = var.head(FEATURE_CAP).index.tolist()
    X_base = X_base[keep]

# -------------------- Embargoed folds --------------------
folds = embargo_time_splits(dates, n_splits=N_FOLDS, embargo_days=1)

# -------------------- Train base models (progress + ETA + budget) --------------------
st.subheader("⚙️ Training Base Models (with early stopping)")
P_xgb_oof = np.zeros(len(y), dtype=np.float32)
P_lgb_oof = np.zeros(len(y), dtype=np.float32)
P_cat_oof = np.zeros(len(y), dtype=np.float32)

total_steps = len(folds) * len(SEEDS)
step = 0
pbar = st.progress(0)
status = st.empty()
t0_all = time.time()
hard_stop = False

for fi, (tr_idx, va_idx) in enumerate(folds):
    if hard_stop:
        break

    X_tr_full, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr_full, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # Negative downsampling for speed (and often better ranking on rare HRs)
    pos_idx = np.where(y_tr_full.values == 1)[0]
    neg_idx = np.where(y_tr_full.values == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > NEG_DOWNSAMPLE * len(pos_idx):
        rng = np.random.default_rng(123 + fi)
        keep_neg = rng.choice(neg_idx, size=NEG_DOWNSAMPLE * len(pos_idx), replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        keep_idx.sort()
        X_tr = X_tr_full.iloc[keep_idx]
        y_tr = y_tr_full.iloc[keep_idx]
    else:
        X_tr, y_tr = X_tr_full, y_tr_full

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for sd in SEEDS:
        # Soft time budget
        if (time.time() - t0_all) > TIME_BUDGET_SEC:
            hard_stop = True
            break

        spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=XGB_N_EST, max_depth=6, learning_rate=XGB_LR,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=XGB_ES,
            n_jobs=N_JOBS, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=LGB_N_EST, learning_rate=LGB_LR, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=N_JOBS, random_state=sd
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=CAT_ITERS, depth=6, learning_rate=CAT_LR, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, spw], od_type="Iter", od_wait=CAT_ES,
            verbose=0, thread_count=N_JOBS, random_seed=sd
        )

        t_fold_seed = time.time()
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(LGB_ES), lgb.log_evaluation(0)])
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds_xgb.append(xgb_clf.predict_proba(X_va)[:,1])
        preds_lgb.append(lgb_clf.predict_proba(X_va)[:,1])
        preds_cat.append(cat_clf.predict_proba(X_va)[:,1])

        step += 1
        elapsed = time.time() - t0_all
        pct = int(100 * step / max(1, total_steps))
        avg_step = elapsed / step
        eta = avg_step * (total_steps - step)
        pbar.progress(min(pct, 100))
        status.write(
            f"Training base models: {step}/{total_steps} • "
            f"Elapsed: {timedelta(seconds=int(elapsed))} • "
            f"ETA: {timedelta(seconds=int(eta))} • fold {fi+1}/{len(folds)}, seed {sd}"
        )

    if hard_stop and (len(preds_xgb) == 0):
        break

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

status.write("✅ Base models complete (soft budget respected).")

# -------------------- Optional day-wise ranker (obeys budget) --------------------
st.subheader("📈 Day-wise Ranker (auto-skips if budget exceeded)")
days = pd.to_datetime(dates).dt.floor("D")
ranker_oof = np.zeros(len(y), dtype=np.float32)
rank_parts = []
for fi, (tr_idx, va_idx) in enumerate(folds):
    if (time.time() - t0_all) > TIME_BUDGET_SEC:
        st.info("⏱️ Skipping remaining ranker folds due to time budget.")
        break
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)

    rk = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=RANK_N_EST, learning_rate=RANK_LR, num_leaves=63,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        random_state=fi, n_jobs=N_JOBS
    )
    rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
           callbacks=[lgb.early_stopping(RANK_ES), lgb.log_evaluation(0)])
    ranker_oof[va_idx] = rk.predict(X_va)
    rank_parts.append(rk.predict(X_base))

ranker_full = np.mean(rank_parts, axis=0) if rank_parts else np.zeros(len(y), dtype=np.float32)

# -------------------- Meta stacker + calibration --------------------
st.subheader("🧮 Meta Stacker + Calibration")
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y.values)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"OOF AUC (meta): {roc_auc_score(y, oof_meta):.4f} | OOF LogLoss (meta): {log_loss(y, oof_meta):.4f}")

# isotonic on OOF and temp-tune (Top-30)
ir = IsotonicRegression(out_of_bounds="clip")
y_oof_iso = ir.fit_transform(oof_meta, y.values)
Tbest = tune_temperature_for_topk(y_oof_iso, y.values, K=TOPK, T_grid=np.linspace(0.8, 1.6, 17))
p_base = expit(logit(np.clip(y_oof_iso, 1e-6, 1-1e-6)) * Tbest)

# -------------------- Weather-free Overlay from Profiles --------------------
st.subheader("🧩 Weather-free Overlay (profiles only)")

# Column knobs (adjust to your profile schema as needed)
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

def _batter_factor(row):
    f = 1.0
    brl = row.get(bat_cols["barrel"], np.nan)
    fb  = row.get(bat_cols["fb"], np.nan)
    pull= row.get(bat_cols["pull"], np.nan)
    hrp = row.get(bat_cols["hr_pa"], np.nan)
    try:
        if pd.notnull(brl):
            if brl >= 0.12: f *= 1.06
            elif brl >= 0.09: f *= 1.03
        if pd.notnull(fb) and fb >= 0.22: f *= 1.02
        if pd.notnull(pull) and pull >= 0.35: f *= 1.02
        if pd.notnull(hrp) and hrp >= 0.06: f *= 1.03
    except Exception:
        pass
    return float(np.clip(f, 0.95, 1.10))

def _pitcher_factor(row):
    f = 1.0
    brlA = row.get(pit_cols["brl_allowed"], np.nan)
    fb   = row.get(pit_cols["fb"], np.nan)
    bb   = row.get(pit_cols["bb"], np.nan)
    xcon = row.get(pit_cols["xwoba_con"], np.nan)
    try:
        if pd.notnull(brlA):
            if brlA >= 0.11: f *= 1.05
            elif brlA >= 0.09: f *= 1.03
        if pd.notnull(fb) and fb >= 0.40: f *= 1.03
        if pd.notnull(bb) and bb >= 0.10: f *= 1.02
        if pd.notnull(xcon):
            if xcon >= 0.40: f *= 1.04
            elif xcon >= 0.36: f *= 1.02
    except Exception:
        pass
    return float(np.clip(f, 0.94, 1.12))

def _platoon_factor(row):
    bhand = str(row.get(platoon_cols["stand"], row.get(platoon_cols["batter_hand"], "R"))).upper()
    phand = str(row.get(platoon_cols["p_throws"], "R")).upper()
    if bhand == "L":
        b_rate = row.get(platoon_cols["batter_vsR"], np.nan)
        p_rate = row.get(platoon_cols["pitcher_vsL"], np.nan)
    else:
        b_rate = row.get(platoon_cols["batter_vsL"], np.nan)
        p_rate = row.get(platoon_cols["pitcher_vsR"], np.nan)
    f = 1.0
    try:
        if pd.notnull(b_rate):
            if b_rate >= 0.05: f *= 1.05
            elif b_rate <= 0.02: f *= 0.98
        if pd.notnull(p_rate):
            if p_rate >= 0.05: f *= 1.04
            elif p_rate <= 0.02: f *= 0.99
        if (bhand == "L" and phand == "R") or (bhand == "R" and phand == "L"):
            f *= 1.01
    except Exception:
        pass
    return float(np.clip(f, 0.94, 1.10))

def _park_factor(row):
    pf = row.get(park_cols["park_factor"], np.nan)
    try:
        pf = float(pf)
        return float(np.clip(pf, 0.85, 1.20))
    except Exception:
        return 1.0

with st.spinner("Computing profile-only overlay components…"):
    bf = ev.apply(_batter_factor, axis=1)
    pf = ev.apply(_pitcher_factor, axis=1)
    pltf = ev.apply(_platoon_factor, axis=1)
    pkf = ev.apply(_park_factor, axis=1)

# -------------------- Multiplier Tuner (overlay exponents) — ALWAYS ON --------------------
st.subheader("🎛️ Multiplier Tuner (Overlay Exponents)")
def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

use_mult = DEFAULT_MULT.copy()
rng = np.random.default_rng(123)
samples_mult = 5000  # cloud-safe
best_key = None; best_res = None
logit_meta = logit(np.clip(p_base, 1e-6, 1-1e-6))

for _ in range(samples_mult):
    a_b = float(rng.uniform(0.2, 1.6))
    b_p = float(rng.uniform(0.2, 1.6))
    c_pl= float(rng.uniform(0.0, 1.2))
    d_pk= float(rng.uniform(0.0, 1.2))
    ov = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
    eval_score = expit(logit_meta + np.log(ov + 1e-9))
    hK = _hits_at_k(y.values, eval_score, TOPK)
    nd = _ndcg_at_k(y.values, eval_score, 30)
    key = (hK, nd)
    if (best_key is None) or (key > best_key):
        best_key = key
        best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk, HitsAtK=hK, NDCG30=nd)

if best_res:
    use_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
    st.success(f"New overlay exponents: {json.dumps(use_mult, indent=2)} | Hits@{TOPK}={best_res['HitsAtK']} NDCG@30={best_res['NDCG30']:.4f}")
else:
    st.info(f"Using default overlay exponents: {json.dumps(use_mult, indent=2)}")

overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay = np.log(overlay + 1e-9)

# -------------------- RRF + disagreement penalty (on OOF) --------------------
def _rank_desc(x):
    x = np.asarray(x)
    return pd.Series(-x).rank(method="min").astype(int).values

disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty = np.clip(zscore(disagree_std), 0, 3)

r_prob    = _rank_desc(p_base)
r_ranker  = _rank_desc(zscore(ranker_oof))
r_overlay = _rank_desc(overlay)
k_rrf = 60.0
rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
rrf_z = zscore(rrf)
ranker_z = zscore(ranker_oof)
logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))

# -------------------- Blended Tuner (final weights) — ALWAYS ON --------------------
st.subheader("🧪 Blended Tuner (Final Weights)")
def blend_with_weights(w, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(w["w_prob"]*logit_p + w["w_overlay"]*log_overlay + w["w_ranker"]*ranker_z + w["w_rrf"]*rrf_z - w["w_penalty"]*dis_penalty)

use_blend = DEFAULT_BLEND.copy()
rng = np.random.default_rng(777)
samples_blend = 8000  # cloud-safe
best_tuple = None; best_row = None

for _ in range(samples_blend):
    wv = rng.dirichlet(np.ones(5))
    cand = dict(w_prob=float(wv[0]), w_overlay=float(wv[1]), w_ranker=float(wv[2]), w_rrf=float(wv[3]), w_penalty=float(wv[4]))
    s = blend_with_weights(cand, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
    hK = _hits_at_k(y.values, s, TOPK)
    h30 = _hits_at_k(y.values, s, 30)
    nd = _ndcg_at_k(y.values, s, 30)
    tup = (hK, nd, h30)
    if (best_tuple is None) or (tup > best_tuple):
        best_tuple = tup
        best_row = {**cand, "HitsAtK":int(hK), "HitsAt30":int(h30), "NDCG30":float(nd)}

if best_row:
    use_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
    st.success(f"New blend weights: {json.dumps(use_blend, indent=2)} | Hits@{TOPK}={best_row['HitsAtK']} NDCG@30={best_row['NDCG30']:.4f}")
else:
    st.info(f"Using default blend weights: {json.dumps(use_blend, indent=2)}")

final_oof = blend_with_weights(use_blend, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)

# -------------------- Diagnostics (no plots) --------------------
st.subheader("📊 Diagnostics")
try:
    st.write(f"AUC (final blend): {roc_auc_score(y, final_oof):.4f}")
except Exception:
    pass
st.write(f"Hits@{TOPK}: {_hits_at_k(y.values, final_oof, TOPK)}")
st.write(f"NDCG@30: {_ndcg_at_k(y.values, final_oof, 30):.4f}")

# Optional RBIs/TB diagnostics (if derived)
if "rbi_by_game" in extra_targets:
    st.write("RBI≥2 sample (first 10 rows):")
    st.dataframe(extra_targets["rbi_by_game"].head(10), use_container_width=True)
if "tb_by_game" in extra_targets:
    st.write("TB≥2 sample (first 10 rows):")
    st.dataframe(extra_targets["tb_by_game"].head(10), use_container_width=True)

# -------------------- Export weights --------------------
st.subheader("💾 Export Best Weights (use in prediction app)")
export_payload = {
    "multiplier_exponents": use_mult,
    "blend_weights": use_blend,
    "notes": "Cloud-optimized offline tuner: weather-free overlay exponents + final blend weights.",
    "meta": {
        "folds": N_FOLDS, "topK": TOPK,
        "feature_cap": FEATURE_CAP, "neg_downsample": NEG_DOWNSAMPLE
    }
}
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

gc.collect()
st.success("✅ Finished. Use the exported JSON to update your main prediction app.")
