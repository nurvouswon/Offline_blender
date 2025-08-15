# offline_tuner.py
# ============================================================
# 🧪 MLB HR Offline Tuner (No-TODAY, No-Weather) — Cloud Speed (stability-patched)
# - Same modeling & tuning logic you had
# - Only changes: dtype fixes, robust merge (string keys), and aggressive memory cleanup per seed/fold
# - Progress bars + ETA; fixed 5 folds; no plots
# ============================================================

import os
os.environ["OMP_NUM_THREADS"] = "2"  # avoid oversubscription on Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, json
from datetime import timedelta

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from scipy.special import logit, expit

# -------------------- Cloud-speed constants (no toggles) --------------------
SEEDS = [42]               # single bag for speed (unchanged from your last version)
N_FOLDS = 5                # fixed
TOPK = 30                  # evaluate Hits@30, NDCG@30
N_JOBS = 1                 # play nice with Streamlit Cloud

# Base models (your lean setup with early stop)
XGB_N_EST, XGB_LR, XGB_ES = 320, 0.06, 25
LGB_N_EST, LGB_LR, LGB_ES = 650, 0.06, 25
CAT_ITERS, CAT_LR, CAT_ES  = 900, 0.06, 30
RANK_N_EST, RANK_LR, RANK_ES = 320, 0.06, 25

# ---------- Defaults for tuners (unchanged) ----------
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

# ---------- UI ----------
st.set_page_config(page_title="🧪 MLB HR Offline Tuner (Cloud)", layout="wide")
st.title("🧪 MLB HR Offline Tuner (Cloud)")

with st.sidebar:
    st.header("📤 Upload Data")
    ev_file  = st.file_uploader("Event-level CSV/Parquet (must include hr_outcome)", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Season-long Batter Profile CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV", type=["csv"], key="pit")
    st.caption("Folds fixed at 5 • Hits@K fixed at 30 • No plots (fast)")

if ev_file is None or bat_file is None or pit_file is None:
    st.info("Upload event-level + batter profile + pitcher profile to begin.")
    st.stop()

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

def _safe_num_df(df: pd.DataFrame) -> pd.DataFrame:
    # match prediction app: try numeric, keep strings where needed
    for c in df.columns:
        if df[c].isnull().all():
            continue
        if df[c].dtype == "O":
            try:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            except Exception:
                pass
        if pd.api.types.is_float_dtype(df[c]):
            try:
                s = df[c].dropna()
                if len(s) and (s % 1 == 0).all():
                    df[c] = df[c].astype(pd.Int64Dtype())
            except Exception:
                pass
    return df

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

def _to_key_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# ---------- Load ----------
with st.spinner("Loading files..."):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file)
    pit = _read_any(pit_file)

# IMPORTANT: actually apply the dtype cleaner to each DF (bug fix)
ev  = _safe_num_df(ev)
bat = _safe_num_df(bat)
pit = _safe_num_df(pit)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ---------- Join Keys (UI, unchanged UX) ----------
st.subheader("🔗 Join Keys")
ev_batter_key = st.selectbox(
    "Event → Batter key",
    options=[c for c in ev.columns if "bat" in c.lower() or "player" in c.lower() or "id" in c.lower()],
    index=0
)
bat_key = st.selectbox("Batter profile key", options=bat.columns, index=0)

ev_pitcher_key = st.selectbox(
    "Event → Pitcher key",
    options=[c for c in ev.columns if "pit" in c.lower() or "pitch" in c.lower() or "id" in c.lower()],
    index=0
)
pit_key = st.selectbox("Pitcher profile key", options=pit.columns, index=0)

# ---------- Robust merge like the prediction app (string keys on BOTH sides) ----------
with st.spinner("Merging profiles into event-level…"):
    ev = ev.copy()
    ev["bat_key_merge"] = _to_key_str(ev[ev_batter_key])
    ev["pit_key_merge"] = _to_key_str(ev[ev_pitcher_key])

    bat_pref = bat.add_prefix("batprof_").copy()
    if f"batprof_{bat_key}" not in bat_pref.columns:
        st.error("Selected batter profile key not found after prefixing.")
        st.stop()
    bat_pref["bat_key_merge"] = _to_key_str(bat_pref[f"batprof_{bat_key}"])

    pit_pref = pit.add_prefix("pitprof_").copy()
    if f"pitprof_{pit_key}" not in pit_pref.columns:
        st.error("Selected pitcher profile key not found after prefixing.")
        st.stop()
    pit_pref["pit_key_merge"] = _to_key_str(pit_pref[f"pitprof_{pit_key}"])

    ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
    ev = ev.merge(pit_pref, on="pit_key_merge", how="left")

st.success("✅ Merged profiles.")

# ---------- Basic feature prep ----------
target_col = "hr_outcome"
if target_col not in ev.columns:
    st.error("Event-level file must contain hr_outcome (0/1).")
    st.stop()

# Avoid obvious leakage (unchanged list)
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

y = ev[target_col].fillna(0).astype(int)
dates_col = "game_date" if "game_date" in ev.columns else None
if dates_col:
    dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# select numeric features
num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].copy()
X_base = X_base.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

# ---------- Folds ----------
st.subheader("⚙️ Training (5 folds, cloud-lean)")
folds = embargo_time_splits(dates, n_splits=N_FOLDS, embargo_days=1)

# ---------- Train base models (progress + ETA) ----------
P_xgb_oof = np.zeros(len(y), dtype=np.float32)
P_lgb_oof = np.zeros(len(y), dtype=np.float32)
P_cat_oof = np.zeros(len(y), dtype=np.float32)

total_steps = len(folds) * len(SEEDS)
step = 0
pbar = st.progress(0)
status = st.empty()
t0_all = time.time()

for fi, (tr_idx, va_idx) in enumerate(folds):
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for sd in SEEDS:
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
            verbose=0, thread_count=N_JOBS, random_seed=sd,
            allow_writing_files=False   # <<< stability on Streamlit Cloud
        )

        # Train
        xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(LGB_ES), lgb.log_evaluation(0)])
        cat_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # Predict
        preds_xgb.append(xgb_clf.predict_proba(X_va)[:,1])
        preds_lgb.append(lgb_clf.predict_proba(X_va)[:,1])
        preds_cat.append(cat_clf.predict_proba(X_va)[:,1])

        # ---- aggressive cleanup per seed ----
        try:
            del xgb_clf
        except: pass
        try:
            del lgb_clf
        except: pass
        try:
            del cat_clf
        except: pass
        gc.collect()

        # progress
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

    # assign OOF
    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

    # ---- cleanup fold-level lists to avoid memory creep ----
    del preds_xgb, preds_lgb, preds_cat, X_tr, X_va, y_tr, y_va
    gc.collect()

status.write("✅ Base models complete.")

# ---------- Optional day-wise ranker (if dates vary) ----------
st.subheader("📈 LambdaRank Head (cloud-lean)")
has_real_days = dates.nunique() > 1
if has_real_days:
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()

    ranker_oof = np.zeros(len(y), dtype=np.float32)

    pbar_rk = st.progress(0)
    status_rk = st.empty()
    total_steps_rk = len(folds)
    step_rk = 0
    t0_rk = time.time()

    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
        g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)

        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=RANK_N_EST, learning_rate=RANK_LR, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(RANK_ES), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)

        # fold cleanup
        try:
            del rk
        except: pass
        del X_tr, X_va, y_tr, y_va, d_tr, d_va, g_tr, g_va
        gc.collect()

        step_rk += 1
        elapsed = time.time() - t0_rk
        pct = int(100 * step_rk / max(1, total_steps_rk))
        avg_step = elapsed / step_rk
        eta = avg_step * (total_steps_rk - step_rk)
        pbar_rk.progress(min(pct, 100))
        status_rk.write(
            f"Training ranker: {step_rk}/{total_steps_rk} • "
            f"Elapsed: {timedelta(seconds=int(elapsed))} • "
            f"ETA: {timedelta(seconds=int(eta))}"
        )

    st.success("✅ Ranker trained.")
else:
    ranker_oof = np.zeros(len(y), dtype=np.float32)
    st.info("Only one unique day found — skipping LambdaRank head.")

# ---------- Meta stacker + calibration (unchanged) ----------
st.subheader("🧮 Meta Stacker + Calibration")
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)

# release base OOF arrays not needed anymore? (keep for disagreement calc later)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y.values)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"OOF AUC: {roc_auc_score(y, oof_meta):.4f} | OOF LogLoss: {log_loss(y, oof_meta):.4f}")

ir = IsotonicRegression(out_of_bounds="clip")
y_oof_iso = ir.fit_transform(oof_meta, y.values)
Tbest = tune_temperature_for_topk(y_oof_iso, y.values, K=TOPK, T_grid=np.linspace(0.8, 1.6, 17))
logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
p_base = expit(logits_oof * Tbest)

# ---------- Weather-free Overlay (profiles) ----------
st.subheader("🧩 Weather-free Overlay from Profiles (season-long)")

bat_cols = dict(
    barrel="batprof_barrel_rate", fb="batprof_fb_rate",
    pull="batprof_pull_rate", hr_pa="batprof_hr_per_pa"
)
pit_cols = dict(
    brl_allowed="pitprof_barrel_rate_allowed", fb="pitprof_fb_rate",
    bb="pitprof_bb_rate", xwoba_con="pitprof_xwoba_con"
)
platoon_cols = dict(
    batter_vsL="batprof_hr_pa_vsL", batter_vsR="batprof_hr_pa_vsR",
    pitcher_vsL="pitprof_hr_pa_vsL", pitcher_vsR="pitprof_hr_pa_vsR",
    stand="stand", batter_hand="batter_hand", p_throws="pitcher_hand"
)
park_cols = dict(park_factor="park_hr_rate")

def _batter_factor(row):
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

def _pitcher_factor(row):
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
    if pd.notnull(b_rate):
        if b_rate >= 0.05: f *= 1.05
        elif b_rate <= 0.02: f *= 0.98
    if pd.notnull(p_rate):
        if p_rate >= 0.05: f *= 1.04
        elif p_rate <= 0.02: f *= 0.99
    if (bhand == "L" and phand == "R") or (bhand == "R" and phand == "L"):
        f *= 1.01
    return float(np.clip(f, 0.94, 1.10))

def _park_factor(row):
    pf = row.get(park_cols["park_factor"], np.nan)
    try:
        pf = float(pf)
        return float(np.clip(pf, 0.85, 1.20))
    except Exception:
        return 1.0

with st.spinner("Computing overlay components…"):
    bf = ev.apply(_batter_factor, axis=1)
    pf_ = ev.apply(_pitcher_factor, axis=1)
    pltf = ev.apply(_platoon_factor, axis=1)
    pkf = ev.apply(_park_factor, axis=1)

# ---------- Multiplier Tuner (exponents) ----------
st.subheader("🎛️ Multiplier Tuner (Overlay Exponents)")
def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf_**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

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

use_mult = st.session_state.saved_best_mult.copy()

rng = np.random.default_rng(123)
samples = 3000  # cloud-lean
best_key = None; best_res = None

pbar_m = st.progress(0)
t0_m = time.time()
for i in range(samples):
    a_b = float(rng.uniform(0.2, 1.6))
    b_p = float(rng.uniform(0.2, 1.6))
    c_pl= float(rng.uniform(0.0, 1.2))
    d_pk= float(rng.uniform(0.0, 1.2))
    ov = overlay_from_exponents(a_b, b_p, c_pl, d_pk)

    eval_score = expit(logit(np.clip(p_base,1e-6,1-1e-6)) + np.log(ov+1e-9))
    hK = _hits_at_k(y.values, eval_score, TOPK)
    nd = _ndcg_at_k(y.values, eval_score, 30)
    key = (hK, nd)

    if (best_key is None) or (key > best_key):
        best_key = key
        best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk, HitsAtK=hK, NDCG30=nd)

    if (i+1) % 50 == 0:
        pbar_m.progress(int(100*(i+1)/samples))

if best_res:
    st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
    use_mult = st.session_state.saved_best_mult.copy()
    st.success(f"New overlay exponents found: {json.dumps(use_mult)} | Hits@{TOPK}={best_res['HitsAtK']} NDCG@30={best_res['NDCG30']:.4f}")

overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay = np.log(overlay + 1e-9)

# ---------- RRF + Disagreement penalty on OOF ----------
def _rank_desc(x):
    x = np.asarray(x)
    return pd.Series(-x).rank(method="min").astype(int).values

disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty = np.clip(zscore(disagree_std), 0, 3)

r_prob   = _rank_desc(p_base)
r_ranker = _rank_desc(zscore(ranker_oof))
r_overlay= _rank_desc(overlay)
k_rrf = 60.0
rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
rrf_z = zscore(rrf)
ranker_z = zscore(ranker_oof)
logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))

# ---------- Blended Tuner ----------
st.subheader("🧪 Blended Tuner (Final weights)")
def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

use_blend = st.session_state.saved_best_blend.copy()

rng = np.random.default_rng(777)
samples = 6000  # cloud-lean
best_tuple = None; best_row = None
pbar_b = st.progress(0)
for i in range(samples):
    w = rng.dirichlet(np.ones(5))
    s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
    hK = _hits_at_k(y.values, s, TOPK)
    h30 = _hits_at_k(y.values, s, 30)
    nd = _ndcg_at_k(y.values, s, 30)
    tup = (hK, nd, h30)
    if (best_tuple is None) or (tup > best_tuple):
        best_tuple = tup
        best_row = dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]), w_rrf=float(w[3]), w_penalty=float(w[4]),
                        HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd))
    if (i+1) % 50 == 0:
        pbar_b.progress(int(100*(i+1)/samples))

if best_row:
    st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
    use_blend = st.session_state.saved_best_blend.copy()
    st.success(f"New blend weights: {json.dumps(use_blend)} | Hits@{TOPK}={best_row['HitsAtK']} NDCG@30={best_row['NDCG30']:.4f}")

# ---------- Final OOF score (diagnostic) ----------
final_oof = blend_with_weights(
    use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
    use_blend["w_rrf"], use_blend["w_penalty"],
    logit_p, log_overlay, ranker_z, rrf_z, dis_penalty
)

st.subheader("📊 Summary (OOF)")
try:
    auc_final = roc_auc_score(y, final_oof)
    st.write(f"AUC (final blend): {auc_final:.4f}")
except Exception:
    pass
st.write(f"Hits@{TOPK}: {_hits_at_k(y.values, final_oof, TOPK)}")
st.write(f"NDCG@30: {_ndcg_at_k(y.values, final_oof, 30):.4f}")

# ---------- Export best weights ----------
st.subheader("💾 Export Best Weights")
col1, col2 = st.columns(2)
with col1:
    st.json(st.session_state.saved_best_mult, expanded=False)
with col2:
    st.json(st.session_state.saved_best_blend, expanded=False)

export_payload = {
    "multiplier_exponents": st.session_state.saved_best_mult,
    "blend_weights": st.session_state.saved_best_blend,
    "notes": "Weather-free overlay exponents + final blend weights from offline tuner (cloud-lean, stability patched)."
}
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

gc.collect()
st.success("✅ Offline tuning complete.")
