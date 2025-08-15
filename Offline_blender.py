# offline_tuner.py
# ============================================================
# 🧪 MLB HR Offline Tuner (Streamlit Cloud — Fast + Progress)
# - Inputs: Event-level data (+ hr_outcome if available), Season-long Batter Profile, Season-long Pitcher Profile
# - Robust key merges (string-cast like prediction app)
# - Base meta-ensemble (XGB/LGB/CB → LR) with early stopping
# - Isotonic calibration + TopK temperature tuning (K=30)
# - Weather-free overlay from profiles (batter/pitcher/platoon/park)
# - Multiplier Tuner (overlay exponents) + progress + ETA
# - Blended Tuner (prob, overlay, ranker, rrf, penalty) + progress + ETA
# - Optional LambdaRank day head (if dates vary)
# - Derives extra targets from event data: TB>=2 and RBI>=2
# - Cloud-safe speed tweaks (2 seeds, leaner iterations, adaptive tuner sizes)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, json
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

# ===================== UI =====================
st.set_page_config(page_title="🧪 MLB Offline Tuner (Cloud Fast)", layout="wide")
st.title("🧪 MLB Offline Tuner (Cloud Fast)")

# ============= CONSTANTS (Cloud-friendly) =============
TOPK = 30
N_FOLDS = 5
SEEDS = [42, 101]  # fewer seeds → ~2x faster, still stable

def _adaptive_samples(n_rows: int):
    # Scale by data size but cap for cloud safety
    m = min(5000, max(1500, int(n_rows * 0.004)))   # Multiplier tuner
    b = min(8000, max(2500, int(n_rows * 0.006)))   # Blended tuner
    return m, b

# ---------- Progress & ETA helpers ----------
def _fmt_eta(seconds):
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

class PhaseProgress:
    def __init__(self, label, total_steps: int):
        self.label = label
        self.total = max(1, int(total_steps))
        self.done = 0
        self.t0 = time.time()
        self.prog = st.progress(0, text=f"{label}… 0%")
        self.info = st.empty()

    def tick(self, step_inc: int = 1, extra_text: str = ""):
        self.done += step_inc
        self.done = min(self.done, self.total)
        pct = int(100 * self.done / self.total)
        elapsed = time.time() - self.t0
        rate = elapsed / max(1, self.done)
        eta = rate * (self.total - self.done)
        self.prog.progress(self.done / self.total, text=f"{self.label}… {pct}%")
        txt = f"{self.label}: {self.done}/{self.total} • Elapsed: {_fmt_eta(elapsed)} • ETA: {_fmt_eta(eta)}"
        if extra_text:
            txt += f" • {extra_text}"
        self.info.write(txt)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False, max_entries=3)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_num_df(df: pd.DataFrame):
    df = df.copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
    return df

def _to_str_key(s):
    # robust string key casting (like prediction app)
    try:
        s = s.astype(str)
    except Exception:
        s = s.map(lambda x: "" if pd.isna(x) else str(x))
    return s.str.strip()

def _winsor(df, cols, lo=0.01, hi=0.99):
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        ql, qh = v.quantile(lo), v.quantile(hi)
        df[c] = v.clip(lower=ql, upper=qh)
    return df

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

# ---------- Uploads ----------
with st.sidebar:
    st.header("📤 Upload Data")
    ev_file = st.file_uploader("Event-level CSV/Parquet", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Season-long Batter Profile CSV (optional)", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV (optional)", type=["csv"], key="pit")

if ev_file is None:
    st.info("Upload event-level data to begin.")
    st.stop()

# ---------- Load ----------
with st.spinner("Loading files..."):
    ev = _read_any(ev_file)
    bat = _read_any(bat_file) if bat_file is not None else pd.DataFrame()
    pit = _read_any(pit_file) if pit_file is not None else pd.DataFrame()

# Light cleaning (numeric coercion where safe)
ev = _safe_num_df(ev)
if not bat.empty: bat = _safe_num_df(bat)
if not pit.empty: pit = _safe_num_df(pit)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ---------- Key detection & merge (robust like prediction app) ----------
def _guess_batter_key(df):
    for k in ["batter_id","batter","player_id","batter_key","bat_id"]:
        if k in df.columns: return k
    # fallback: exact 'batter' commonly exists
    return "batter" if "batter" in df.columns else df.columns[0]

def _guess_pitcher_key(df):
    for k in ["pitcher_id","pitcher","p_pitcher","pitcher_key","pit_id"]:
        if k in df.columns: return k
    return "pitcher" if "pitcher" in df.columns else df.columns[0]

ev_batter_key = _guess_batter_key(ev)
ev_pitcher_key = _guess_pitcher_key(ev)

# Cast keys to string
ev["bat_key_merge"] = _to_str_key(ev[ev_batter_key]) if ev_batter_key in ev.columns else _to_str_key(ev.iloc[:,0])
ev["pit_key_merge"] = _to_str_key(ev[ev_pitcher_key]) if ev_pitcher_key in ev.columns else _to_str_key(ev.iloc[:,0])

if not bat.empty:
    # Try to pick the first id-like column as key
    bat_key = None
    for c in bat.columns:
        if "id" in c.lower() or "batter" in c.lower() or "player" in c.lower():
            bat_key = c; break
    if bat_key is None:
        bat_key = bat.columns[0]
    bat_pref = bat.add_prefix("batprof_")
    bat_pref = bat_pref.rename(columns={f"batprof_{bat_key}": "bat_key_merge"})
    bat_pref["bat_key_merge"] = _to_str_key(bat_pref["bat_key_merge"])
    ev = ev.merge(bat_pref, on="bat_key_merge", how="left")

if not pit.empty:
    pit_key = None
    for c in pit.columns:
        if "id" in c.lower() or "pitch" in c.lower() or "player" in c.lower():
            pit_key = c; break
    if pit_key is None:
        pit_key = pit.columns[0]
    pit_pref = pit.add_prefix("pitprof_")
    pit_pref = pit_pref.rename(columns={f"pitprof_{pit_key}": "pit_key_merge"})
    pit_pref["pit_key_merge"] = _to_str_key(pit_pref["pit_key_merge"])
    ev = ev.merge(pit_pref, on="pit_key_merge", how="left")

st.success("✅ Merged profiles (if provided).")

# ---------- Targets: HR, TB>=2, RBI>=2 ----------
y_targets = {}

# HR (if available)
if "hr_outcome" in ev.columns:
    y_targets["HR"] = ev["hr_outcome"].fillna(0).astype(int).values

# TB>=2 derived from events_clean/events
def _tb_from_event(s: pd.Series):
    s = s.astype(str).str.lower()
    m = {
        "single": 1, "double": 2, "triple": 3, "home_run": 4,
        "home run": 4, "homerun": 4, "hr": 4
    }
    return s.map(lambda x: m.get(x, 0)).fillna(0).astype(int)

tb_source = None
for c in ["events_clean","events","des"]:
    if c in ev.columns:
        tb_source = c; break

if tb_source is not None:
    tb_val = _tb_from_event(ev[tb_source])
    y_targets["TB_2+"] = (tb_val >= 2).astype(int).values

# RBI>=2 derived from score delta if present
if {"bat_score","post_bat_score"}.issubset(ev.columns):
    rbi_play = (pd.to_numeric(ev["post_bat_score"], errors="coerce")
                - pd.to_numeric(ev["bat_score"], errors="coerce")).fillna(0)
    rbi_play = rbi_play.clip(lower=0).astype(int)
    y_targets["RBI_2+"] = (rbi_play >= 2).astype(int).values

if not y_targets:
    st.error("No usable targets found. Provide hr_outcome OR columns to derive TB/RBI (events_clean/events and bat_score/post_bat_score).")
    st.stop()

# ---------- Features ----------
dates_col = "game_date" if "game_date" in ev.columns else None
if dates_col:
    dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# Avoid obvious leakage features for TRAINING (keep labels derivation separate)
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events_clean","events","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
num_cols_all = ev.select_dtypes(include=[np.number]).columns.tolist()
# Drop leakage numeric columns if present
num_cols = [c for c in num_cols_all if c not in LEAK]
X_base = ev[num_cols].copy()
# Clean matrix
X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

# Adaptive tuner sizes
SAMPLES_MULT, SAMPLES_BLEND = _adaptive_samples(len(ev))
st.write(f"Adaptive tuner sizes → Mult: {SAMPLES_MULT:,} • Blend: {SAMPLES_BLEND:,}")

# ---------- Train base models (OOF) with early stopping + progress ----------
folds = embargo_time_splits(dates, n_splits=N_FOLDS, embargo_days=1)
total_steps = len(folds) * len(SEEDS)
pbar_models = PhaseProgress("Training base models", total_steps)

P_xgb_oof = np.zeros(len(ev), dtype=np.float32)
P_lgb_oof = np.zeros(len(ev), dtype=np.float32)
P_cat_oof = np.zeros(len(ev), dtype=np.float32)

fold_times = []
for fi, (tr_idx, va_idx) in enumerate(folds):
    t0 = time.time()
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for sd in SEEDS:
        # balance
        # Auto-choose any target present to compute spw (use HR if available else first)
        y_for_spw = next(iter(y_targets.values()))
        y_tr_spw = y_for_spw[tr_idx]
        spw = max(1.0, (len(y_tr_spw) - y_tr_spw.sum()) / max(1.0, y_tr_spw.sum()))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=700, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.04, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=sd
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=1300, depth=7, learning_rate=0.04, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, spw], od_type="Iter", od_wait=50,
            verbose=0, thread_count=1, random_seed=sd
        )

        # Fit/predict per target? We need single OOF base for meta — use HR if present else first target.
        # Train on the same y_for_spw (proxy base learner); meta will calibrate per target later.
        y_tr = y_for_spw[tr_idx]
        y_va = y_for_spw[va_idx]

        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds_xgb.append(xgb_clf.predict_proba(X_va)[:,1])
        preds_lgb.append(lgb_clf.predict_proba(X_va)[:,1])
        preds_cat.append(cat_clf.predict_proba(X_va)[:,1])

        pbar_models.tick(extra_text=f"fold {fi+1}/{len(folds)}, seed {sd}")

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

    dt = time.time() - t0
    fold_times.append(dt)

# ---------- Optional day-wise ranker ----------
has_real_days = dates.nunique() > 1
if has_real_days:
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    ranker_oof = np.zeros(len(ev), dtype=np.float32)
    parts = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
        # Use the same proxy target as base learners (HR if present else first)
        y_proxy = next(iter(y_targets.values()))
        y_tr, y_va = y_proxy[tr_idx], y_proxy[va_idx]
        d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
        g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=550, learning_rate=0.06, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
        parts.append(rk.predict(X_base))
    ranker_full = np.mean(parts, axis=0)
    st.success("✅ LambdaRank head trained.")
else:
    ranker_oof = np.zeros(len(ev), dtype=np.float32)
    ranker_full = ranker_oof
    st.info("Only one unique day found — skipping LambdaRank head.")

# ---------- RRF + Disagreement (OOF) ----------
def _rank_desc(x):
    x = np.asarray(x)
    return pd.Series(-x).rank(method="min").astype(int).values

disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty = np.clip(zscore(disagree_std), 0, 3)

# ---------- Weather-free Overlay (profiles) ----------
# Column knobs (adjust names to your profile schema if needed)
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

def _park_factor_row(row):
    pf = row.get(park_cols["park_factor"], np.nan)
    try:
        pf = float(pf)
        return float(np.clip(pf, 0.85, 1.20))
    except Exception:
        return 1.0

with st.spinner("Computing overlay components (profiles only)…"):
    bf = ev.apply(_batter_factor_row, axis=1)
    pf_ = ev.apply(_pitcher_factor_row, axis=1)
    pltf = ev.apply(_platoon_factor_row, axis=1)
    pkf = ev.apply(_park_factor_row, axis=1)

# ---------- Multiplier tuner + Blended tuner per target ----------
if "saved_best_mult" not in st.session_state:
    st.session_state.saved_best_mult = dict(a_batter=0.80, b_pitcher=0.80, c_platoon=0.60, d_park=0.40)
if "saved_best_blend" not in st.session_state:
    st.session_state.saved_best_blend = dict(w_prob=0.30, w_overlay=0.20, w_ranker=0.20, w_rrf=0.10, w_penalty=0.20)

def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf_**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

def _hits_at_k(y_true, s, K=TOPK):
    ord_idx = np.argsort(-s)
    return int(np.sum(y_true[ord_idx][:K]))

def _dcg_at_k(rels, K):
    rels = np.asarray(rels)[:K]
    if rels.size == 0: return 0.0
    discounts = 1.0/np.log2(np.arange(2, 2+len(rels)))
    return float(np.sum(rels*discounts))

def _ndcg_at_k(y_true, s, K=TOPK):
    ord_idx = np.argsort(-s)
    rel_sorted = y_true[ord_idx]
    dcg = _dcg_at_k(rel_sorted, K)
    ideal = np.sort(y_true)[::-1]
    idcg = _dcg_at_k(ideal, K)
    return (dcg/idcg) if idcg > 0 else 0.0

def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

results_rows = []

for target_name, y in y_targets.items():
    st.markdown(f"### 🎯 Target: **{target_name}**")

    # Meta stacker + calibration for THIS target
    X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
    scaler_meta = StandardScaler()
    X_meta_s = scaler_meta.fit_transform(X_meta)
    meta = LogisticRegression(max_iter=1000, solver="lbfgs")
    meta.fit(X_meta_s, y)
    oof_meta = meta.predict_proba(X_meta_s)[:,1]

    try:
        auc = roc_auc_score(y, oof_meta)
        ll  = log_loss(y, oof_meta)
        st.write(f"Base meta OOF → AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    except Exception:
        pass

    ir = IsotonicRegression(out_of_bounds="clip")
    y_oof_iso = ir.fit_transform(oof_meta, y)
    Tbest = tune_temperature_for_topk(y_oof_iso, y, K=TOPK, T_grid=np.linspace(0.8, 1.6, 17))
    logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
    p_base = expit(logits_oof * Tbest)
    logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))

    # RRF pieces (OOF)
    r_prob   = _rank_desc(p_base)
    r_ranker = _rank_desc(zscore(ranker_oof))
    # temporary overlay placeholder; will recalc inside tuner
    r_overlay_dummy = _rank_desc(np.ones_like(p_base))
    k_rrf = 60.0
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay_dummy)
    rrf_z_base = zscore(rrf)
    ranker_z = zscore(ranker_oof)

    # ===== Multiplier Tuner (overlay exponents) with progress =====
    use_mult = st.session_state.saved_best_mult.copy()
    pbar_mult = PhaseProgress(f"[{target_name}] Multiplier tuner", SAMPLES_MULT)

    best_key = None; best_res = None
    rng = np.random.default_rng(123)
    # Chunked ticks to reduce Streamlit overhead
    CHUNK = 50; progressed = 0

    for i in range(SAMPLES_MULT):
        a_b = float(rng.uniform(0.2, 1.6))
        b_p = float(rng.uniform(0.2, 1.6))
        c_pl= float(rng.uniform(0.0, 1.2))
        d_pk= float(rng.uniform(0.0, 1.2))
        overlay = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
        log_overlay = np.log(overlay + 1e-9)

        # Update RRF with true overlay rank
        r_overlay = _rank_desc(overlay)
        rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
        rrf_z = zscore(rrf)

        eval_score = expit(logit_p + log_overlay + 0.10*rrf_z)  # light proxy
        hK = _hits_at_k(y, eval_score, TOPK)
        nd = _ndcg_at_k(y, eval_score, 30)
        key = (hK, nd)

        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk,
                            HitsAtK=int(hK), NDCG30=float(nd))

        progressed += 1
        if progressed % CHUNK == 0 or progressed == SAMPLES_MULT:
            pbar_mult.tick(step_inc=(CHUNK if progressed % CHUNK == 0 else SAMPLES_MULT % CHUNK))

    if best_res:
        st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
        use_mult = st.session_state.saved_best_mult.copy()
        st.success(f"Overlay exponents → {json.dumps(use_mult)} | Hits@{TOPK}={best_res['HitsAtK']} • NDCG@30={best_res['NDCG30']:.4f}")

    # Final overlay for this target
    overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
    log_overlay = np.log(overlay + 1e-9)

    # ===== Blended tuner with progress =====
    use_blend = st.session_state.saved_best_blend.copy()
    pbar_blend = PhaseProgress(f"[{target_name}] Blended tuner", SAMPLES_BLEND)

    best_tuple = None; best_row = None
    rng2 = np.random.default_rng(777)
    CHUNK2 = 50; progressed2 = 0

    # Recompute RRF with final overlay ranks
    r_overlay = _rank_desc(overlay)
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)

    for j in range(SAMPLES_BLEND):
        w = rng2.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
        hK = _hits_at_k(y, s, TOPK)
        h30 = _hits_at_k(y, s, 30)
        nd = _ndcg_at_k(y, s, 30)
        tup = (hK, nd, h30)
        if (best_tuple is None) or (tup > best_tuple):
            best_tuple = tup
            best_row = dict(
                w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]),
                w_rrf=float(w[3]), w_penalty=float(w[4]),
                HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd)
            )
        progressed2 += 1
        if progressed2 % CHUNK2 == 0 or progressed2 == SAMPLES_BLEND:
            pbar_blend.tick(step_inc=(CHUNK2 if progressed2 % CHUNK2 == 0 else SAMPLES_BLEND % CHUNK2))

    if best_row:
        st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
        use_blend = st.session_state.saved_best_blend.copy()
        st.success(f"Blend weights → {json.dumps(use_blend)} | Hits@{TOPK}={best_row['HitsAtK']} • NDCG@30={best_row['NDCG30']:.4f}")

    # Final OOF score + diagnostics for this target
    final_oof = blend_with_weights(
        use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
        use_blend["w_rrf"], use_blend["w_penalty"],
        logit_p, log_overlay, ranker_z, rrf_z, dis_penalty
    )

    try:
        auc_final = roc_auc_score(y, final_oof)
    except Exception:
        auc_final = None

    res = dict(
        Target=target_name,
        AUC=f"{auc_final:.4f}" if auc_final is not None else "n/a",
        HitsAtK=_hits_at_k(y, final_oof, TOPK),
        NDCG30=f"{_ndcg_at_k(y, final_oof, 30):.4f}",
        OverlayExponents=st.session_state.saved_best_mult.copy(),
        BlendWeights=st.session_state.saved_best_blend.copy()
    )
    results_rows.append(res)

# ---------- Summary table + export ----------
st.markdown("## 📋 Summary (per target)")
summary_df = pd.DataFrame(results_rows)
st.dataframe(summary_df, use_container_width=True)

export_payload = {
    "notes": "Offline tuner (cloud fast) — learned weather-free overlay exponents + final blend weights per target.",
    "targets": results_rows
}
st.download_button(
    "⬇️ Download Best Weights (JSON)",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

gc.collect()
st.success("✅ Tuning complete.")
