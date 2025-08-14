# offline_tuner.py
# ============================================================
# 🧪 MLB HR Offline Tuner (No TODAY.csv, No weather dependency)
# - Inputs: Event-level data (+ hr_outcome), Season-long Batter Profile, Season-long Pitcher Profile
# - Base models: XGB / LGB / CatBoost -> Logistic stacker (with early stopping)
# - Isotonic calibration + top-K temperature tuning
# - Weather-FREE overlay built from profiles (batter/pitcher/platoon/park)
# - Multiplier Tuner (learn overlay exponents)
# - Blended Tuner (weights: prob, overlay, ranker, rrf, penalty)
# - Optional day-wise LambdaRank (if dates exist)
# - Streamlit Cloud friendly (modest n_estimators, 1 CPU threads, caching)
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

import matplotlib.pyplot as plt
from scipy.special import logit, expit

# ---------------- UI ----------------
st.set_page_config(page_title="🧪 MLB HR Offline Tuner (No Weather)", layout="wide")
st.title("🧪 MLB HR Offline Tuner (No Weather)")

st.caption("Upload your event-level parquet/CSV (with hr_outcome) plus season-long Batter & Pitcher profile CSVs. "
           "This tuner learns overlay exponents and final blend weights — so you can paste the weights into the main prediction app.")

# ---------------- Defaults (safe starting points) ----------------
DEFAULT_BLEND = dict(
    w_prob=0.30,     # base prob (calibrated + temp-tuned)
    w_overlay=0.20,  # overlay (profiles only)
    w_ranker=0.20,   # LambdaRank head
    w_rrf=0.10,      # reciprocal rank fusion aux
    w_penalty=0.20   # disagreement penalty (subtracted)
)

DEFAULT_MULT = dict(
    a_batter=0.80,   # exponent for batter factor
    b_pitcher=0.80,  # exponent for pitcher factor
    c_platoon=0.60,  # exponent for platoon factor
    d_park=0.40      # exponent for park factor
)

if "saved_best_blend" not in st.session_state:
    st.session_state.saved_best_blend = DEFAULT_BLEND.copy()
if "saved_best_mult" not in st.session_state:
    st.session_state.saved_best_mult = DEFAULT_MULT.copy()

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, max_entries=3)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_numeric(df):
    # light converter: try numeric, leave others as-is
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
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

def tune_temperature_for_topk(p_oof, y, K=20, T_grid=np.linspace(0.8, 1.6, 17)):
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

def _rank_desc(x):
    x = np.asarray(x)
    return pd.Series(-x).rank(method="min").astype(int).values

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("📤 Upload Data")
    ev_file  = st.file_uploader("Event-level CSV/Parquet (must include hr_outcome)", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Season-long Batter Profile CSV", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV", type=["csv"], key="pit")

    st.header("⚙️ Settings")
    n_splits = st.slider("CV Splits (time-embargoed)", 4, 7, 5, 1)
    topK_eval = st.selectbox("Hits@K metric", options=[10, 20, 30], index=1)
    run_mult_tuner  = st.toggle("Run Multiplier Tuner", value=True)
    run_blend_tuner = st.toggle("Run Blended Tuner", value=True)

if ev_file is None or bat_file is None or pit_file is None:
    st.info("Upload event-level + batter profile + pitcher profile to begin.")
    st.stop()

# ---------------- Load ----------------
with st.spinner("Loading files..."):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file)
    pit = _read_any(pit_file)

for df in (ev, bat, pit):
    _safe_numeric(df)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ---------------- Join Keys (Prediction-app–style merge) ----------------
st.subheader("🔗 Join Keys")
ev_batter_key = st.selectbox("Event → Batter key", options=[c for c in ev.columns if "bat" in c or "player" in c or "id" in c], index=0)
bat_key       = st.selectbox("Batter profile key", options=bat.columns, index=0)

ev_pitcher_key = st.selectbox("Event → Pitcher key", options=[c for c in ev.columns if "pit" in c or "pitcher" in c or "id" in c], index=0)
pit_key        = st.selectbox("Pitcher profile key", options=pit.columns, index=0)

with st.spinner("Merging profiles into event-level…"):
    ev = ev.copy()

    # ---- EXACT approach we use in prediction app: cast to string + strip both sides ----
    ev["bat_key_merge"] = ev[ev_batter_key].astype(str).str.strip()
    bat_local = bat.copy()
    bat_local["bat_key_merge"] = bat_local[bat_key].astype(str).str.strip()

    ev["pit_key_merge"] = ev[ev_pitcher_key].astype(str).str.strip()
    pit_local = pit.copy()
    pit_local["pit_key_merge"] = pit_local[pit_key].astype(str).str.strip()

    # Prefix to avoid collisions
    bat_local = bat_local.add_prefix("batprof_")
    pit_local = pit_local.add_prefix("pitprof_")

    # restore merge key names post-prefix
    bat_local = bat_local.rename(columns={f"batprof_{bat_key}":"_drop_bat_key"})
    pit_local = pit_local.rename(columns={f"pitprof_{pit_key}":"_drop_pit_key"})

    # Keep explicit merge keys
    bat_local = bat_local.rename(columns={"batprof_bat_key_merge":"bat_key_merge"})
    pit_local = pit_local.rename(columns={"pitprof_pit_key_merge":"pit_key_merge"})

    # If the above two renames didn't hit (depends on pandas add_prefix behavior with new col),
    # ensure the merge cols exist:
    if "bat_key_merge" not in bat_local.columns and "batprof_bat_key_merge" in bat_local.columns:
        bat_local = bat_local.rename(columns={"batprof_bat_key_merge":"bat_key_merge"})
    if "pit_key_merge" not in pit_local.columns and "pitprof_pit_key_merge" in pit_local.columns:
        pit_local = pit_local.rename(columns={"pitprof_pit_key_merge":"pit_key_merge"})

    # Left-joins to keep all event rows
    ev = ev.merge(bat_local.drop(columns=["_drop_bat_key"], errors="ignore"), on="bat_key_merge", how="left")
    ev = ev.merge(pit_local.drop(columns=["_drop_pit_key"], errors="ignore"), on="pit_key_merge", how="left")

st.success("✅ Merged profiles.")

# ---------------- Target & Dates ----------------
target_col = "hr_outcome"
if target_col not in ev.columns:
    st.error("Event-level file must contain hr_outcome (0/1).")
    st.stop()

y = ev[target_col].fillna(0).astype(int)

dates_col = "game_date" if "game_date" in ev.columns else None
if dates_col:
    dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

# ---------------- Feature matrix ----------------
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

# ---------------- CV & Base models (early stopping) ----------------
st.subheader("⚙️ Training Base Models (with early stopping)")
folds = embargo_time_splits(dates, n_splits=n_splits, embargo_days=1)

P_xgb_oof = np.zeros(len(y), dtype=np.float32)
P_lgb_oof = np.zeros(len(y), dtype=np.float32)
P_cat_oof = np.zeros(len(y), dtype=np.float32)

seeds = [42, 101]  # lighter for Streamlit Cloud; bump to [42,101,202,404] later
fold_times = []

for fi, (tr_idx, va_idx) in enumerate(folds):
    t0 = time.time()
    X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []

    for sd in seeds:
        spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))

        xgb_clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, max_depth=-1, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=sd
        )
        cat_clf = cb.CatBoostClassifier(
            iterations=1500, depth=7, learning_rate=0.03, l2_leaf_reg=6.0,
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

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)

    dt = time.time() - t0
    fold_times.append(dt)
    st.write(f"Fold {fi+1}/{len(folds)} in {timedelta(seconds=int(dt))}")

# ---------------- Optional day-wise ranker ----------------
st.subheader("📈 Day-wise Ranker (optional)")
has_real_days = dates.nunique() > 1
if has_real_days:
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    ranker_oof = np.zeros(len(y), dtype=np.float32)
    parts = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X_base.iloc[tr_idx], X_base.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
        g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
        parts.append(rk.predict(X_base))
    ranker_full = np.mean(parts, axis=0)
    st.success("✅ Ranker trained.")
else:
    ranker_oof = np.zeros(len(y), dtype=np.float32)
    ranker_full = ranker_oof
    st.info("Only one unique day found — skipping LambdaRank head.")

# ---------------- Meta stacker + Isotonic + temp-tune ----------------
st.subheader("🧮 Meta Stacker + Calibration")
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y.values)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"OOF AUC: {roc_auc_score(y, oof_meta):.4f} | OOF LogLoss: {log_loss(y, oof_meta):.4f}")

ir = IsotonicRegression(out_of_bounds="clip")
y_oof_iso = ir.fit_transform(oof_meta, y.values)
Tbest = tune_temperature_for_topk(y_oof_iso, y.values, K=topK_eval, T_grid=np.linspace(0.8, 1.6, 17))
logits_oof = logit(np.clip(y_oof_iso, 1e-6, 1-1e-6))
p_base = expit(logits_oof * Tbest)
logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))

# ---------------- Weather-FREE Overlay (profiles only) ----------------
st.subheader("🧩 Weather-free Overlay from Profiles")

# You can remap these to your exact profile headers as needed
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

with st.spinner("Computing overlay components (profiles only)…"):
    bf = ev.apply(_batter_factor, axis=1)
    pf = ev.apply(_pitcher_factor, axis=1)
    pltf = ev.apply(_platoon_factor, axis=1)
    pkf = ev.apply(_park_factor, axis=1)

# ---------------- Multiplier Tuner (overlay exponents) ----------------
st.subheader("🎛️ Multiplier Tuner (Overlay Exponents)")
def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pf**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

use_mult = st.session_state.saved_best_mult.copy()

if run_mult_tuner:
    rng = np.random.default_rng(123)
    samples = 5000  # safer for Streamlit Cloud; bump later if you want
    best_key = None; best_res = None
    for _ in range(samples):
        a_b = float(rng.uniform(0.2, 1.6))
        b_p = float(rng.uniform(0.2, 1.6))
        c_pl= float(rng.uniform(0.0, 1.2))
        d_pk= float(rng.uniform(0.0, 1.2))
        ov = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
        eval_score = expit(logit(np.clip(p_base,1e-6,1-1e-6)) + np.log(ov+1e-9))
        hK = _hits_at_k(y.values, eval_score, topK_eval)
        nd = _ndcg_at_k(y.values, eval_score, 30)
        key = (hK, nd)
        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk, HitsAtK=hK, NDCG30=nd)
    if best_res:
        st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
        use_mult = st.session_state.saved_best_mult.copy()
        st.success(f"New overlay exponents: {json.dumps(use_mult, indent=2)} | "
                   f"Hits@{topK_eval}={best_res['HitsAtK']} NDCG@30={best_res['NDCG30']:.4f}")
else:
    st.info(f"Using saved overlay exponents: {json.dumps(use_mult, indent=2)}")

overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay = np.log(overlay + 1e-9)

# ---------------- RRF + Disagreement penalty (OOF) ----------------
disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty = np.clip(zscore(disagree_std), 0, 3)

r_prob    = _rank_desc(p_base)
r_ranker  = _rank_desc(zscore(ranker_oof))
r_overlay = _rank_desc(overlay)
k_rrf = 60.0
rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
rrf_z = zscore(rrf)
ranker_z = zscore(ranker_oof)

# ---------------- Blended Tuner (final weights) ----------------
st.subheader("🧪 Blended Tuner (Final weights)")
def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

use_blend = st.session_state.saved_best_blend.copy()

if run_blend_tuner:
    rng = np.random.default_rng(777)
    samples = 8000  # modest for Streamlit Cloud
    best_tuple = None; best_row = None
    for _ in range(samples):
        w = rng.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
        hK = _hits_at_k(y.values, s, topK_eval)
        h30 = _hits_at_k(y.values, s, 30)
        nd = _ndcg_at_k(y.values, s, 30)
        tup = (hK, nd, h30)
        if (best_tuple is None) or (tup > best_tuple):
            best_tuple = tup
            best_row = dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]),
                            w_rrf=float(w[3]), w_penalty=float(w[4]),
                            HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd))
    if best_row:
        st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
        use_blend = st.session_state.saved_best_blend.copy()
        st.success(f"New blend weights: {json.dumps(use_blend, indent=2)} | "
                   f"Hits@{topK_eval}={best_row['HitsAtK']} NDCG@30={best_row['NDCG30']:.4f}")
else:
    st.info(f"Using saved blend weights: {json.dumps(use_blend, indent=2)}")

# ---------------- Final OOF score (diagnostics only in this offline app) ----------------
final_oof = blend_with_weights(
    use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
    use_blend["w_rrf"], use_blend["w_penalty"],
    logit_p, log_overlay, ranker_z, rrf_z, dis_penalty
)

st.subheader("📊 Diagnostics on Event-Level OOF")
try:
    st.write(f"AUC (final blend): {roc_auc_score(y, final_oof):.4f}")
except Exception:
    pass
st.write(f"Hits@{topK_eval}: {_hits_at_k(y.values, final_oof, topK_eval)}")
st.write(f"NDCG@30: {_ndcg_at_k(y.values, final_oof, 30):.4f}")

# ---------------- Plots ----------------
with st.expander("Distribution Plots"):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(final_oof, bins=40, alpha=0.8)
    ax.set_title("Final Blended OOF Score Distribution")
    st.pyplot(fig)
    plt.close(fig)

# ---------------- Export best weights ----------------
st.subheader("💾 Export Best Weights")
col1, col2 = st.columns(2)
with col1:
    st.json(st.session_state.saved_best_mult, expanded=False)
with col2:
    st.json(st.session_state.saved_best_blend, expanded=False)

export_payload = {
    "multiplier_exponents": st.session_state.saved_best_mult,
    "blend_weights": st.session_state.saved_best_blend,
    "notes": "Weather-free overlay exponents + final blend weights from offline tuner."
}
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

gc.collect()
st.success("✅ Done. Paste these weights into the prediction app when ready.")
