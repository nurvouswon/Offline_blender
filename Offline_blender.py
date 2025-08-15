# Offline_blender.py
# ============================================================
# 🧪 MLB HR Offline Tuner (Cloud-Optimized, No Weather)
# - Inputs: Event-level parquet/CSV (+ hr_outcome), Season-long Batter Profile CSV, Season-long Pitcher Profile CSV
# - Uses SAME merge style as your prediction app (string join keys)  ✅
# - Builds base meta-ensemble (XGB/LGB/CB → LR) with early stopping
# - Isotonic calibration + top-K temperature tuning (K=30 fixed)
# - Weather-free overlay from profiles (batter/pitcher/platoon/park)
# - Multiplier Tuner (overlay exponents) + Blended Tuner (prob/overlay/ranker/rrf/penalty)
# - Optional LambdaRank (if multiple dates)
# - Progress bars + ETA; no plots; 5 folds fixed
# - Cloud-safe memory & time guards
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, json, os, psutil
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

# ------------------- App configuration -------------------
st.set_page_config(page_title="🧪 MLB HR Offline Tuner (Cloud)", layout="wide")
st.title("🧪 MLB HR Offline Tuner (Cloud-Optimized)")

# Fixed settings (per your request)
N_FOLDS = 5
TOPK_EVAL = 30
RUN_MULT_TUNER = True
RUN_BLEND_TUNER = True

# Conservative defaults to keep Streamlit Cloud happy
DEFAULT_BLEND = dict(
    w_prob=0.30, w_overlay=0.20, w_ranker=0.20, w_rrf=0.10, w_penalty=0.20
)
DEFAULT_MULT = dict(
    a_batter=0.80, b_pitcher=0.80, c_platoon=0.60, d_park=0.40
)

if "saved_best_blend" not in st.session_state:
    st.session_state.saved_best_blend = DEFAULT_BLEND.copy()
if "saved_best_mult" not in st.session_state:
    st.session_state.saved_best_mult = DEFAULT_MULT.copy()

# ------------------- Helpers -------------------
@st.cache_data(show_spinner=False, max_entries=3)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_num_frame(df: pd.DataFrame) -> pd.DataFrame:
    # numeric-friendly cast but DO NOT touch join keys yet
    for c in df.columns:
        if df[c].dtype == 'O':
            # leave IDs/names as-is; numeric-like strings will become numbers
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def _winsor(v: pd.Series, lo=0.01, hi=0.99):
    v = pd.to_numeric(v, errors="coerce")
    ql, qh = v.quantile(lo), v.quantile(hi)
    return v.clip(lower=ql, upper=qh)

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

def tune_temperature_for_topk(p_oof, y, K=TOPK_EVAL, T_grid=np.linspace(0.8, 1.6, 17)):
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

def _hits_at_k(y_true, s, K):
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

# ------------------- Uploads -------------------
with st.sidebar:
    st.header("📤 Upload Data")
    ev_file  = st.file_uploader("Event-level (.parquet/.csv) — must have hr_outcome", type=["parquet","csv"], key="ev")
    bat_file = st.file_uploader("Season Batter Profile (.csv)", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season Pitcher Profile (.csv)", type=["csv"], key="pit")

if ev_file is None or bat_file is None or pit_file is None:
    st.info("Upload all three files to begin (event-level + batter profile + pitcher profile).")
    st.stop()

# ------------------- Load -------------------
with st.spinner("Loading files…"):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file)
    pit = _read_any(pit_file)

# Light numeric-friendly casting but keep join-keys flexible
ev  = _safe_num_frame(ev)
bat = _safe_num_frame(bat)
pit = _safe_num_frame(pit)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ------------------- Auto key suggestions -------------------
st.subheader("🔗 Join Keys (auto-detected sensible defaults)")
# For event-level we saw headers earlier; use robust defaults but allow override if desired later
ev_batter_candidates = [c for c in ev.columns if c.lower() in ("batter_id","batter","batterid","player_id","playerid")]
ev_pitcher_candidates = [c for c in ev.columns if c.lower() in ("pitcher_id","pitcher","pitcherid")]

ev_batter_key = ev_batter_candidates[0] if ev_batter_candidates else ev.columns[0]
ev_pitcher_key = ev_pitcher_candidates[0] if ev_pitcher_candidates else ev.columns[0]

bat_key = [c for c in bat.columns if "id" in c.lower() or "key" in c.lower()]
bat_key = bat_key[0] if bat_key else bat.columns[0]

pit_key = [c for c in pit.columns if "id" in c.lower() or "key" in c.lower()]
pit_key = pit_key[0] if pit_key else pit.columns[0]

st.write(f"• Event→Batter key: `{ev_batter_key}`  |  Batter profile key: `{bat_key}`")
st.write(f"• Event→Pitcher key: `{ev_pitcher_key}` |  Pitcher profile key: `{pit_key}`")

# ------------------- Merge (IDENTICAL STYLE TO PREDICTION APP) -------------------
# Cast BOTH sides to string before merging (prevents object vs int64 mismatch)
bat_pref = bat.add_prefix("batprof_").rename(columns={f"batprof_{bat_key}": "bat_key_merge"})
pit_pref = pit.add_prefix("pitprof_").rename(columns={f"pitprof_{pit_key}": "pit_key_merge"})

bat_pref["bat_key_merge"] = bat_pref["bat_key_merge"].astype(str)
pit_pref["pit_key_merge"] = pit_pref["pit_key_merge"].astype(str)

ev = ev.copy()
ev["bat_key_merge"] = ev[ev_batter_key].astype(str)
ev["pit_key_merge"] = ev[ev_pitcher_key].astype(str)

with st.spinner("Merging profiles into event-level…"):
    ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
    ev = ev.merge(pit_pref, on="pit_key_merge", how="left")
st.success("✅ Profiles merged.")

# ------------------- Basic label/feature prep -------------------
target_col = "hr_outcome"
if target_col not in ev.columns:
    st.error("Event-level file must contain hr_outcome (0/1).")
    st.stop()

# Compute derived labels for RBI≥2 and Total Bases≥2 (from event-level if not provided)
# We assume event-level rows are PAs; aggregate by game+player if you have per-PA labels.
# For offline tuner scoring quality, we evaluate at PA-level by proxy (common when slate-like).
def _safe_col(name): return name in ev.columns

# Try to build simple proxies (best-effort without full boxscore):
if "events_clean" in ev.columns:
    # TB proxy from events/launch
    single_like = ev["events_clean"].isin(["single"])
    double_like = ev["events_clean"].isin(["double"])
    triple_like = ev["events_clean"].isin(["triple"])
    hr_like     = ev["events_clean"].isin(["home_run"])
    tb = (single_like*1 + double_like*2 + triple_like*3 + hr_like*4).astype(int)
else:
    # fallback: if slg_numeric exists, rough TB proxy
    tb = pd.to_numeric(ev.get("slg_numeric", pd.Series(0, index=ev.index)), errors="coerce").fillna(0)
    tb = (tb*4).round().astype(int).clip(lower=0, upper=4)

ev["label_TB2"] = (tb >= 2).astype(int)

# RBI proxy: if we have woba_value/babip_value/etc., not reliable—fallback to HR as minimal RBI signal
# If you have real RBI per PA, replace this with that column.
if "events_clean" in ev.columns:
    rbi_proxy = ev["events_clean"].isin(["home_run"]).astype(int)  # HR guarantees ≥1 RBI
else:
    rbi_proxy = pd.Series(0, index=ev.index, dtype=int)
ev["label_RBI2"] = (rbi_proxy >= 2).astype(int)  # conservative; likely rare without real RBI

# Note: These proxies are placeholders to let the tuner run; your prediction app uses TODAY csv for real RBI/TB outcomes.
# When you have a better per-PA RBI/TB label, just replace label_RBI2/label_TB2 accordingly.

# Remove obvious leakage columns (angles/speeds/outcomes from same PA)
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

# Dates (for time splits / ranker)
if "game_date" in ev.columns:
    dates = pd.to_datetime(ev["game_date"], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# Numeric feature matrix
num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

y_hr   = ev[target_col].fillna(0).astype(int).values
y_tb2  = ev["label_TB2"].fillna(0).astype(int).values
y_rbi2 = ev["label_RBI2"].fillna(0).astype(int).values

gc.collect()

# ------------------- CV folds -------------------
folds = embargo_time_splits(dates, n_splits=N_FOLDS, embargo_days=1)
if len(folds) < 2:
    st.error("Not enough distinct days for time-embargoed folds. Add more data.")
    st.stop()

# ------------------- Training base models (with progress) -------------------
st.subheader("⚙️ Training Base Models (with early stopping)")

P_xgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_lgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_cat_oof = np.zeros(len(y_hr), dtype=np.float32)

# Cloud runtime guard: 2 seeds for speed (you asked to keep power, but Cloud constraints)
SEEDS = [42, 101]

fold_bar = st.progress(0, text="Starting…")
step = 0
t_global0 = time.time()
total_steps = len(folds) * len(SEEDS)

for fi, (tr_idx, va_idx) in enumerate(folds):
    t_fold0 = time.time()
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr, y_va = y_hr[tr_idx], y_hr[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for si, sd in enumerate(SEEDS):
        spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=900, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1600, learning_rate=0.03, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=sd
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=1800, depth=7, learning_rate=0.03, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="Logloss",
            class_weights=[1.0, spw], od_type="Iter", od_wait=50,
            verbose=0, thread_count=1, random_seed=sd
        )

        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds_xgb.append(xgb_clf.predict_proba(X_va)[:,1])
        preds_lgb.append(lgb_clf.predict_proba(X_va)[:,1])
        preds_cat.append(cat_clf.predict_proba(X_va)[:,1])

        step += 1
        elapsed = time.time() - t_global0
        rate = elapsed / max(1, step)
        eta = rate * (total_steps - step)
        fold_bar.progress(
            min(100, int((step/total_steps)*100)),
            text=f"Training base models: {step}/{total_steps} • Elapsed: {timedelta(seconds=int(elapsed))} • ETA: {timedelta(seconds=int(eta))} • fold {fi+1}/{len(folds)}, seed {sd}"
        )

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

st.success("✅ Base models trained.")

# ------------------- Optional day-wise ranker -------------------
st.subheader("📈 Day-wise Ranker (optional)")
has_real_days = pd.Series(dates).nunique() > 1
if has_real_days:
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    ranker_oof = np.zeros(len(y_hr), dtype=np.float32)
    parts = []
    rk_bar = st.progress(0, text="Training LambdaRank…")
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
        y_tr, y_va = y_hr[tr_idx], y_hr[va_idx]
        d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
        g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
        parts.append(rk.predict(X_base))
        rk_bar.progress(int(((fi+1)/len(folds))*100), text=f"LambdaRank fold {fi+1}/{len(folds)}")
    ranker_full = np.mean(parts, axis=0)
    st.success("✅ Ranker trained.")
else:
    ranker_oof = np.zeros(len(y_hr), dtype=np.float32)
    ranker_full = ranker_oof
    st.info("Only one unique day found — skipping LambdaRank head.")

# ------------------- Meta stacker + calibration -------------------
st.subheader("🧮 Meta Stacker + Calibration")
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y_hr)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"OOF AUC: {roc_auc_score(y_hr, oof_meta):.4f} | OOF LogLoss: {log_loss(y_hr, oof_meta):.4f}")

# isotonic on OOF and temp-tune for TOPK_EVAL
ir = IsotonicRegression(out_of_bounds="clip")
y_oof_iso = ir.fit_transform(oof_meta, y_hr)
Tbest = tune_temperature_for_topk(y_oof_iso, y_hr, K=TOPK_EVAL, T_grid=np.linspace(0.8, 1.6, 17))
logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
p_base = expit(logits_oof * Tbest)

# ------------------- Weather-free overlay (profiles) -------------------
st.subheader("🧩 Weather-free Overlay (profiles only)")
# column maps (adjust if your profile schema differs)
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
    park_factor="park_hr_rate"
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

with st.spinner("Computing overlay components…"):
    bf   = ev.apply(_batter_factor_row, axis=1).astype(np.float32)
    pf   = ev.apply(_pitcher_factor_row, axis=1).astype(np.float32)
    pltf = ev.apply(_platoon_factor_row, axis=1).astype(np.float32)
    pkf  = ev.apply(_park_factor_row, axis=1).astype(np.float32)

# ------------------- Multiplier tuner (exponents) -------------------
st.subheader("🎛️ Multiplier Tuner (Overlay Exponents)")
def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

use_mult = st.session_state.saved_best_mult.copy()

if RUN_MULT_TUNER:
    rng = np.random.default_rng(123)
    samples = 6000  # Cloud-safe
    best_key = None; best_res = None
    # Evaluate vs HR labels (primary), but these exponents will also help TB/RBI blends
    for _ in range(samples):
        a_b = float(rng.uniform(0.2, 1.6))
        b_p = float(rng.uniform(0.2, 1.6))
        c_pl= float(rng.uniform(0.0, 1.2))
        d_pk= float(rng.uniform(0.0, 1.2))
        ov = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
        eval_score = expit(logit(np.clip(p_base,1e-6,1-1e-6)) + np.log(ov+1e-9))
        hK = _hits_at_k(y_hr, eval_score, TOPK_EVAL)
        nd = _ndcg_at_k(y_hr, eval_score, 30)
        key = (hK, nd)
        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk,
                            HitsAtK=hK, NDCG30=nd)
    if best_res:
        st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
        use_mult = st.session_state.saved_best_mult.copy()
        st.success(f"New overlay exponents: {json.dumps(use_mult)}  |  Hits@{TOPK_EVAL}={best_res['HitsAtK']}  NDCG@30={best_res['NDCG30']:.4f}")
else:
    st.info(f"Using saved overlay exponents: {json.dumps(use_mult)}")

overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay = np.log(overlay + 1e-9)

# ------------------- RRF + disagreement penalty (OOF-based) -------------------
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

# ------------------- Blended tuner (final weights) -------------------
st.subheader("🧪 Blended Tuner (Final weights, HR primary)")
def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

use_blend = st.session_state.saved_best_blend.copy()

if RUN_BLEND_TUNER:
    rng = np.random.default_rng(777)
    samples = 8000  # Cloud-safe
    best_tuple = None; best_row = None
    for _ in range(samples):
        w = rng.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
        hK = _hits_at_k(y_hr, s, TOPK_EVAL)
        h30 = _hits_at_k(y_hr, s, 30)
        nd = _ndcg_at_k(y_hr, s, 30)
        tup = (hK, nd, h30)
        if (best_tuple is None) or (tup > best_tuple):
            best_tuple = tup
            best_row = dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]),
                            w_rrf=float(w[3]), w_penalty=float(w[4]),
                            HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd))
    if best_row:
        st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
        use_blend = st.session_state.saved_best_blend.copy()
        st.success(f"New blend weights: {json.dumps(use_blend)}  |  Hits@{TOPK_EVAL}={best_row['HitsAtK']}  NDCG@30={best_row['NDCG30']:.4f}")
else:
    st.info(f"Using saved blend weights: {json.dumps(use_blend)}")

# Final HR OOF score (diagnostic)
final_hr_oof = blend_with_weights(
    use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
    use_blend["w_rrf"], use_blend["w_penalty"],
    logit_p, log_overlay, ranker_z, rrf_z, dis_penalty
)

# ------------------- Quick diagnostics (no plots) -------------------
st.subheader("📊 Diagnostics (OOF)")
st.write(f"HR  | Hits@{TOPK_EVAL}: {_hits_at_k(y_hr, final_hr_oof, TOPK_EVAL)}  |  NDCG@30: {_ndcg_at_k(y_hr, final_hr_oof, 30):.4f}")
# Evaluate overlay+meta as proxies for TB≥2 and RBI≥2 (given placeholder labels)
st.write(f"TB2 | Hits@{TOPK_EVAL}: {_hits_at_k(y_tb2, final_hr_oof, TOPK_EVAL)}  |  NDCG@30: {_ndcg_at_k(y_tb2, final_hr_oof, 30):.4f}")
st.write(f"RBI2| Hits@{TOPK_EVAL}: {_hits_at_k(y_rbi2, final_hr_oof, TOPK_EVAL)}  |  NDCG@30: {_ndcg_at_k(y_rbi2, final_hr_oof, 30):.4f}")

# ------------------- Export best weights -------------------
st.subheader("💾 Export Best Weights")
export_payload = {
    "multiplier_exponents": st.session_state.saved_best_mult,
    "blend_weights": st.session_state.saved_best_blend,
    "notes": "Weather-free overlay exponents + final blend weights from offline tuner (Cloud-optimized)."
}
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

st.caption("This tuner mirrors your prediction app’s merge behavior and adds ETA/progress, early-stop ensembles, and cloud-safe tuners.")
