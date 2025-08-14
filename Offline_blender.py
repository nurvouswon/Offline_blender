# tuner_offline_cloud.py
# ============================================================
# 🧪 MLB Offline Tuner (Streamlit Cloud Safe) — HR + TB(>1.5) + RBI(>1.5)*
# *RBI runs only if an 'rbi' column exists; TB is built from slg_numeric per batter-game.
#
# Key upgrades in this version:
#   • HR pipeline unchanged (early-stop meta-ensemble + isotonic + temp tune + overlay + blend tuner)
#   • TB/RBI now use richer game-level, pre-game features from your event headers:
#       - Batter rolling quality: avg EV, barrel%, FB%, hard-hit%, SLG (windows 7/14/30)
#       - Pitcher vulnerability: barrel%, FB%, hard-hit% (7/14/30)
#       - Platoon: stand vs pitcher_hand
#       - Park: park_hr_rate (+ hand splits if present)
#       - Small synergies: batter_barrel × pitcher_barrel, batter_FB × pitcher_FB, batter_barrel × park
#   • Prevents same-day leakage by taking the FIRST appearance row per batter-date to form features.
#   • Streamlit Cloud friendly: bounded estimators, single seed for TB/RBI, early stopping everywhere.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import gc, time, json, re
from datetime import timedelta

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
st.set_page_config(page_title="🧪 MLB Offline Tuner — HR + TB + RBI (Cloud)", layout="wide")
st.title("🧪 MLB Offline Tuner — HR + TB + RBI (Cloud-safe)")

# ---------------- Defaults ----------------
DEFAULT_BLEND = dict(w_prob=0.30, w_overlay=0.20, w_ranker=0.20, w_rrf=0.10, w_penalty=0.20)
DEFAULT_MULT  = dict(a_batter=0.80, b_pitcher=0.80, c_platoon=0.60, d_park=0.40)

if "saved_best_blend" not in st.session_state:
    st.session_state.saved_best_blend = DEFAULT_BLEND.copy()
if "saved_best_mult" not in st.session_state:
    st.session_state.saved_best_mult = DEFAULT_MULT.copy()

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, max_entries=5)
def _read_any(path):
    fn = str(getattr(path, 'name', path)).lower()
    if fn.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)

def _safe_num_df(df):
    for c in df.columns:
        if df[c].dtype == "O":
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

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📤 Uploads & Settings")
    ev_file  = st.file_uploader("Event-level (parquet/csv) — must include hr_outcome", type=["parquet","csv"])
    bat_file = st.file_uploader("Season-long Batter Profile CSV (optional)", type=["csv"])
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV (optional)", type=["csv"])

    n_splits = st.slider("CV Splits (time-embargoed)", 4, 7, 5, 1)
    topK_eval = st.selectbox("Hits@K metric", options=[10, 20, 30], index=1)

    run_mult_tuner  = st.toggle("Run Multiplier Tuner (HR)", value=True)
    run_blend_tuner = st.toggle("Run Blended Tuner (HR)", value=True)

if ev_file is None:
    st.info("Upload your event-level file to start.")
    st.stop()

# ---------------- Load ----------------
with st.spinner("Loading files…"):
    ev = _read_any(ev_file)
    if bat_file is not None: bat = _read_any(bat_file)
    else: bat = pd.DataFrame()
    if pit_file is not None: pit = _read_any(pit_file)
    else: pit = pd.DataFrame()

ev = _safe_num_df(ev)
bat = _safe_num_df(bat) if not bat.empty else bat
pit = _safe_num_df(pit) if not pit.empty else pit

st.write(f"Event rows: {len(ev):,}")
if not bat.empty: st.write(f"Batter profile rows: {len(bat):,}")
if not pit.empty: st.write(f"Pitcher profile rows: {len(pit):,}")

# ---------------- Optional: merge profiles (for HR overlay only) ----------------
if not bat.empty or not pit.empty:
    st.subheader("🔗 Join Keys (Profiles → Event)")
    ev_batter_key = st.selectbox("Event → Batter key", options=[c for c in ev.columns if "bat" in c or "player" in c or "id" in c], index=0)
    if not bat.empty:
        bat_key = st.selectbox("Batter profile key", options=bat.columns, index=0)
    if not pit.empty:
        ev_pitcher_key = st.selectbox("Event → Pitcher key", options=[c for c in ev.columns if "pit" in c or "pitcher" in c or "id" in c], index=0)
        pit_key = st.selectbox("Pitcher profile key", options=pit.columns, index=0)

    with st.spinner("Merging profiles into event-level…"):
        ev = ev.copy()
        if not bat.empty:
            bat_pref = bat.add_prefix("batprof_")
            bat_pref = bat_pref.rename(columns={f"batprof_{bat_key}": "bat_key_merge"})
            ev["bat_key_merge"] = ev[ev_batter_key]
            ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
        if not pit.empty:
            pit_pref = pit.add_prefix("pitprof_")
            pit_pref = pit_pref.rename(columns={f"pitprof_{pit_key}": "pit_key_merge"})
            ev["pit_key_merge"] = ev[ev_pitcher_key]
            ev = ev.merge(pit_pref, on="pit_key_merge", how="left")
    st.success("✅ Profiles merged.")

gc.collect()

# ---------------- HR (event-level) model ----------------
st.subheader("⚙️ HR Model (event-level)")
target_hr = "hr_outcome"
if target_hr not in ev.columns:
    st.error("Event file must include hr_outcome (0/1).")
    st.stop()

# Anti-leak drops (keep TB/RBI raw file separate later)
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

dates_col = "game_date" if "game_date" in ev.columns else None
if dates_col:
    dates = pd.to_datetime(ev[dates_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
else:
    dates = pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)

y_hr = ev[target_hr].fillna(0).astype(int)
X_hr = ev.select_dtypes(include=[np.number]).copy().replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

folds = embargo_time_splits(dates, n_splits=n_splits, embargo_days=1)
P_xgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_lgb_oof = np.zeros(len(y_hr), dtype=np.float32)
P_cat_oof = np.zeros(len(y_hr), dtype=np.float32)
seeds = [42, 101]  # cloud-safe

times = []
for fi, (tr_idx, va_idx) in enumerate(folds):
    t0 = time.time()
    X_tr, X_va = X_hr.iloc[tr_idx], X_hr.iloc[va_idx]
    y_tr, y_va = y_hr.iloc[tr_idx], y_hr.iloc[va_idx]

    preds_xgb, preds_lgb, preds_cat = [], [], []
    for sd in seeds:
        spw = max(1.0, (len(y_tr) - y_tr.sum()) / max(1.0, y_tr.sum()))
        xgb_clf = xgb.XGBClassifier(
            n_estimators=900, max_depth=6, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            eval_metric="logloss", tree_method="hist",
            scale_pos_weight=spw, early_stopping_rounds=50,
            n_jobs=1, verbosity=0, random_state=sd
        )
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.03, max_depth=-1, num_leaves=63,
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

    P_xgb_oof[va_idx] = np.mean(preds_xgb, axis=0)
    P_lgb_oof[va_idx] = np.mean(preds_lgb, axis=0)
    P_cat_oof[va_idx] = np.mean(preds_cat, axis=0)
    dt = time.time() - t0
    times.append(dt)
    st.write(f"HR fold {fi+1}/{len(folds)} — {timedelta(seconds=int(dt))}")

# Optional LambdaRank (if multiple days)
st.caption("Training day-wise ranker for HR…")
has_real_days = pd.to_datetime(dates).dt.floor("D").nunique() > 1
if has_real_days:
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d):
        return d.groupby(d.values).size().values.tolist()
    ranker_oof = np.zeros(len(y_hr), dtype=np.float32)
    parts = []
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X_hr.iloc[tr_idx], X_hr.iloc[va_idx]
        y_tr, y_va = y_hr.iloc[tr_idx], y_hr.iloc[va_idx]
        d_tr, d_va = days.iloc[tr_idx], days.iloc[va_idx]
        g_tr, g_va = _groups_from_days(d_tr), _groups_from_days(d_va)
        rk = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg",
            n_estimators=700, learning_rate=0.05, num_leaves=63,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            random_state=fi
        )
        rk.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        ranker_oof[va_idx] = rk.predict(X_va)
        parts.append(rk.predict(X_hr))
    ranker_full_hr = np.mean(parts, axis=0)
else:
    ranker_oof = np.zeros(len(y_hr), dtype=np.float32)
    ranker_full_hr = ranker_oof

# Stacker + calibration for HR
X_meta = np.column_stack([P_xgb_oof, P_lgb_oof, P_cat_oof]).astype(np.float32)
scaler_meta = StandardScaler()
X_meta_s = scaler_meta.fit_transform(X_meta)
meta = LogisticRegression(max_iter=1000, solver="lbfgs")
meta.fit(X_meta_s, y_hr.values)
oof_meta = meta.predict_proba(X_meta_s)[:,1]
st.write(f"HR OOF AUC: {roc_auc_score(y_hr, oof_meta):.4f} | LogLoss: {log_loss(y_hr, oof_meta):.4f}")

ir = IsotonicRegression(out_of_bounds="clip")
oof_iso = ir.fit_transform(oof_meta, y_hr.values)
Tbest_hr = tune_temperature_for_topk(oof_iso, y_hr.values, K=topK_eval, T_grid=np.linspace(0.8, 1.6, 17))
p_hr_base = expit(logit(np.clip(oof_iso, 1e-6, 1-1e-6)) * Tbest_hr)

# RRF & disagreement for HR (OOF)
disagree_std = np.std(np.vstack([P_xgb_oof, P_lgb_oof, P_cat_oof]), axis=0)
dis_penalty_hr = np.clip(zscore(disagree_std), 0, 3)
r_prob   = _rank_desc(p_hr_base)
r_ranker = _rank_desc(zscore(ranker_oof))
rrf_tmp  = 1.0/(60.0 + r_prob) + 1.0/(60.0 + r_ranker)
rrf_z_hr = zscore(rrf_tmp)
ranker_z_hr = zscore(ranker_oof)

# ---------------- Weather-free overlay (profiles) for HR ----------------
st.subheader("🧩 Profiles Overlay (weather-free) for HR")
bat_cols = dict(barrel="batprof_barrel_rate", fb="batprof_fb_rate", pull="batprof_pull_rate", hr_pa="batprof_hr_per_pa")
pit_cols = dict(brl_allowed="pitprof_barrel_rate_allowed", fb="pitprof_fb_rate", bb="pitprof_bb_rate", xwoba_con="pitprof_xwoba_con")
platoon_cols = dict(batter_vsL="batprof_hr_pa_vsL", batter_vsR="batprof_hr_pa_vsR",
                    pitcher_vsL="pitprof_hr_pa_vsL", pitcher_vsR="pitprof_hr_pa_vsR",
                    stand="stand", batter_hand="batter_hand", p_throws="pitcher_hand")
park_cols = dict(park_factor="park_hr_rate")

def _batter_factor(row):
    f=1.0; brl=row.get(bat_cols["barrel"],np.nan); fb=row.get(bat_cols["fb"],np.nan); pull=row.get(bat_cols["pull"],np.nan); hrp=row.get(bat_cols["hr_pa"],np.nan)
    if pd.notnull(brl): f*=1.06 if brl>=0.12 else 1.03 if brl>=0.09 else 1.0
    if pd.notnull(fb) and fb>=0.22: f*=1.02
    if pd.notnull(pull) and pull>=0.35: f*=1.02
    if pd.notnull(hrp) and hrp>=0.06: f*=1.03
    return float(np.clip(f,0.95,1.10))
def _pitcher_factor(row):
    f=1.0; ba=row.get(pit_cols["brl_allowed"],np.nan); fb=row.get(pit_cols["fb"],np.nan); bb=row.get(pit_cols["bb"],np.nan); x=row.get(pit_cols["xwoba_con"],np.nan)
    if pd.notnull(ba): f*=1.05 if ba>=0.11 else 1.03 if ba>=0.09 else 1.0
    if pd.notnull(fb) and fb>=0.40: f*=1.03
    if pd.notnull(bb) and bb>=0.10: f*=1.02
    if pd.notnull(x): f*=1.04 if x>=0.40 else 1.02 if x>=0.36 else 1.0
    return float(np.clip(f,0.94,1.12))
def _platoon_factor(row):
    bhand = str(row.get(platoon_cols["stand"], row.get(platoon_cols["batter_hand"], "R"))).upper()
    phand = str(row.get(platoon_cols["p_throws"], "R")).upper()
    if bhand=="L":
        b_rate=row.get(platoon_cols["batter_vsR"],np.nan); p_rate=row.get(platoon_cols["pitcher_vsL"],np.nan)
    else:
        b_rate=row.get(platoon_cols["batter_vsL"],np.nan); p_rate=row.get(platoon_cols["pitcher_vsR"],np.nan)
    f=1.0
    if pd.notnull(b_rate): f*=1.05 if b_rate>=0.05 else 0.98 if b_rate<=0.02 else 1.0
    if pd.notnull(p_rate): f*=1.04 if p_rate>=0.05 else 0.99 if p_rate<=0.02 else 1.0
    if (bhand=="L" and phand=="R") or (bhand=="R" and phand=="L"): f*=1.01
    return float(np.clip(f,0.94,1.10))
def _park_factor(row):
    pf=row.get(park_cols["park_factor"],np.nan)
    try: pf=float(pf); return float(np.clip(pf,0.85,1.20))
    except: return 1.0

with st.spinner("Computing overlay components…"):
    bf = ev.apply(_batter_factor, axis=1) if any(k in ev.columns for k in bat_cols.values()) else pd.Series(1.0, index=ev.index)
    pfv= ev.apply(_pitcher_factor, axis=1) if any(k in ev.columns for k in pit_cols.values()) else pd.Series(1.0, index=ev.index)
    pltf= ev.apply(_platoon_factor, axis=1) if any(k in ev.columns for k in ["stand","batter_hand","pitcher_hand",
                                                                               "batprof_hr_pa_vsL","batprof_hr_pa_vsR",
                                                                               "pitprof_hr_pa_vsL","pitprof_hr_pa_vsR"]) \
         else pd.Series(1.0, index=ev.index)
    pkf = ev.apply(_park_factor, axis=1) if "park_hr_rate" in ev.columns else pd.Series(1.0, index=ev.index)

def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pfv**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

use_mult = st.session_state.saved_best_mult.copy()

# Multiplier tuner (HR)
if run_mult_tuner:
    rng = np.random.default_rng(123)
    samples = 4000
    best_key = None; best_res = None
    for _ in range(samples):
        ab = float(rng.uniform(0.2, 1.6))
        bp = float(rng.uniform(0.2, 1.6))
        cp = float(rng.uniform(0.0, 1.2))
        dp = float(rng.uniform(0.0, 1.2))
        ov = overlay_from_exponents(ab, bp, cp, dp)
        eval_score = expit(logit(np.clip(p_hr_base,1e-6,1-1e-6)) + np.log(ov+1e-9))
        hK = _hits_at_k(y_hr.values, eval_score, topK_eval)
        nd = _ndcg_at_k(y_hr.values, eval_score, 30)
        key = (hK, nd)
        if (best_key is None) or (key > best_key):
            best_key = key
            best_res = dict(a_batter=ab, b_pitcher=bp, c_platoon=cp, d_park=dp, HitsAtK=hK, NDCG30=nd)
    if best_res:
        st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
        use_mult = st.session_state.saved_best_mult.copy()
        st.success(f"Overlay exponents (HR-tuned): {json.dumps(use_mult, indent=2)} | Hits@{topK_eval}={best_res['HitsAtK']} NDCG@30={best_res['NDCG30']:.4f}")
else:
    st.info(f"Using saved overlay exponents: {json.dumps(use_mult, indent=2)}")

overlay_ev = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay_ev = np.log(overlay_ev + 1e-9)

# RRF for HR with overlay
r_overlay = _rank_desc(overlay_ev)
rrf = 1.0/(60.0 + r_prob) + 1.0/(60.0 + r_ranker) + 1.0/(60.0 + r_overlay)
rrf_z_hr = zscore(rrf)

# Blend tuner (HR)
def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

use_blend = st.session_state.saved_best_blend.copy()
logit_hr = logit(np.clip(p_hr_base, 1e-6, 1-1e-6))

if run_blend_tuner:
    rng = np.random.default_rng(777)
    samples = 6000
    best_tuple=None; best_row=None
    for _ in range(samples):
        w = rng.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_hr, log_overlay_ev, zscore(ranker_oof), rrf_z_hr, dis_penalty_hr)
        hK = _hits_at_k(y_hr.values, s, topK_eval)
        h30= _hits_at_k(y_hr.values, s, 30)
        nd = _ndcg_at_k(y_hr.values, s, 30)
        tup = (hK, nd, h30)
        if (best_tuple is None) or (tup > best_tuple):
            best_tuple=tup
            best_row=dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]), w_rrf=float(w[3]), w_penalty=float(w[4]),
                          HitsAtK=int(hK), HitsAt30=int(h30), NDCG30=float(nd))
    if best_row:
        st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
        use_blend = st.session_state.saved_best_blend.copy()
        st.success(f"Blend weights (HR): {json.dumps(use_blend, indent=2)} | Hits@{topK_eval}={best_row['HitsAtK']} NDCG@30={best_row['NDCG30']:.4f}")
else:
    st.info(f"Using saved blend weights: {json.dumps(use_blend, indent=2)}")

final_oof_hr = blend_with_weights(use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
                                  use_blend["w_rrf"], use_blend["w_penalty"],
                                  logit_hr, log_overlay_ev, zscore(ranker_oof), rrf_z_hr, dis_penalty_hr)

st.subheader("📊 HR Diagnostics")
st.write(f"HR AUC (final blend): {roc_auc_score(y_hr, final_oof_hr):.4f}")
st.write(f"HR Hits@{topK_eval}: {_hits_at_k(y_hr.values, final_oof_hr, topK_eval)} | NDCG@30: {_ndcg_at_k(y_hr.values, final_oof_hr, 30):.4f}")

with st.expander("HR Score Distribution"):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(final_oof_hr, bins=40, alpha=0.8)
    ax.set_title("HR — Final Blended OOF Scores")
    st.pyplot(fig); plt.close(fig)

# ======================================================================
#                     TB / RBI  — FEATURE BUILDER
# ======================================================================

# Helper: pick first available col among variants like b_fb_rate_7, b_fb_rate_14, b_fb_rate_7_x, etc.
def pick_first(df, roots, windows=(7,14,30,60), allow_base=True):
    # roots can be a string (base root) or list of explicit column names
    if isinstance(roots, (list, tuple)):
        for r in roots:
            if r in df.columns: return r
        return None
    # tolerate suffix variants (_x) and window suffixes
    base = roots
    cands = []
    if allow_base and base in df.columns: cands.append(base)
    for w in windows:
        cands += [f"{base}_{w}", f"{base}_{w}_x"]
    for c in cands:
        if c in df.columns:
            return c
    # fallback by regex startswith
    pat = re.compile(rf"^{re.escape(base)}_(\d+)(_x)?$")
    for c in df.columns:
        if pat.match(str(c)): return c
    return None

# Build compact, stable game-level features from FIRST PA row per batter-date (pre-game snapshot)
def build_game_features(raw_ev, id_col):
    df = raw_ev.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date

    # Keep one row (first appearance) per batter-date
    df = df.sort_values(["game_date", id_col, "at_bat_number"], kind="mergesort") if "at_bat_number" in df.columns else \
         df.sort_values(["game_date", id_col], kind="mergesort")
    df = df.drop_duplicates([id_col, "game_date"], keep="first").reset_index(drop=True)

    # Pick columns using your header patterns
    cols = {}

    # Batter quality (use 7→14→30 preference)
    cols["b_ev"]    = pick_first(df, "b_avg_exit_velo")
    cols["b_brl"]   = pick_first(df, "b_barrel_rate")
    cols["b_fb"]    = pick_first(df, "b_fb_rate")
    cols["b_hh"]    = pick_first(df, "b_hard_hit_rate")
    cols["b_slg"]   = pick_first(df, "b_slg")
    cols["b_pull"]  = pick_first(df, "b_pull_rate")

    # Pitcher allowed quality
    cols["p_brl"]   = pick_first(df, "p_barrel_rate")
    cols["p_fb"]    = pick_first(df, "p_fb_rate")
    cols["p_hh"]    = pick_first(df, "p_hard_hit_rate")

    # Park & platoon
    cols["park"]    = "park_hr_rate" if "park_hr_rate" in df.columns else None
    cols["park_rhb"]= "park_hr_pct_rhb" if "park_hr_pct_rhb" in df.columns else None
    cols["park_lhb"]= "park_hr_pct_lhb" if "park_hr_pct_lhb" in df.columns else None
    hand_col = "stand" if "stand" in df.columns else ("batter_hand" if "batter_hand" in df.columns else None)
    pthrow_col = "pitcher_hand" if "pitcher_hand" in df.columns else None

    # Assemble feature frame
    feats = pd.DataFrame(index=df.index)
    def addf(name, col, clip=None):
        if col and col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            if clip: v = v.clip(*clip)
            feats[name] = v
        else:
            feats[name] = np.nan

    addf("b_ev", cols["b_ev"])
    addf("b_brl", cols["b_brl"], clip=(0, 0.30))
    addf("b_fb", cols["b_fb"], clip=(0, 0.80))
    addf("b_hh", cols["b_hh"], clip=(0, 1.00))
    addf("b_slg", cols["b_slg"], clip=(0, 2.50))
    addf("b_pull", cols["b_pull"], clip=(0, 1.00))

    addf("p_brl", cols["p_brl"], clip=(0, 0.30))
    addf("p_fb",  cols["p_fb"],  clip=(0, 0.80))
    addf("p_hh",  cols["p_hh"],  clip=(0, 1.00))

    # Park (hand-aware if possible)
    if cols["park_rhb"] and cols["park_lhb"] and hand_col:
        hand = df[hand_col].astype(str).str.upper().fillna("R")
        pf = np.where(hand=="R", pd.to_numeric(df[cols["park_rhb"]], errors="coerce"),
                               pd.to_numeric(df[cols["park_lhb"]], errors="coerce"))
        feats["park"] = pd.Series(pf, index=feats.index).clip(0.75, 1.30)
    else:
        addf("park", cols["park"], clip=(0.75, 1.30))

    # Platoon indicator (+ small generic bump for opposite hands)
    if hand_col and pthrow_col:
        hand = df[hand_col].astype(str).str.upper().fillna("R")
        pthr = df[pthrow_col].astype(str).str.upper().fillna("R")
        feats["platoon_opp"] = ((hand=="L") & (pthr=="R") | (hand=="R") & (pthr=="L")).astype(float)
    else:
        feats["platoon_opp"] = 0.0

    # Synergies (bounded)
    feats["syn_brl_pbrl"] = (feats["b_brl"] * feats["p_brl"]).clip(0, 0.12)
    feats["syn_fb_pfb"]   = (feats["b_fb"]  * feats["p_fb"]).clip(0, 0.48)
    feats["syn_brl_park"] = (feats["b_brl"] * feats["park"]).clip(0, 0.36)

    # Fill, cast
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(-1.0).astype(np.float32)

    meta = df[[id_col, "game_date"]].copy()
    return feats, meta

# ======================================================================
#                     TB > 1.5  — GAME-LEVEL MODEL
# ======================================================================
st.subheader("⚙️ TB > 1.5 (game-level)")
raw_ev_tb = _read_any(ev_file)
raw_ev_tb = _safe_num_df(raw_ev_tb)
tb_id_col = None
for cand in ["batter_id","batter","player_name"]:
    if cand in raw_ev_tb.columns:
        tb_id_col = cand; break

if tb_id_col and "game_date" in raw_ev_tb.columns and "slg_numeric" in raw_ev_tb.columns:
    raw_ev_tb["game_date"] = pd.to_datetime(raw_ev_tb["game_date"], errors="coerce").dt.date
    grp_tb = raw_ev_tb.groupby([tb_id_col, "game_date"], as_index=False)["slg_numeric"].sum()
    grp_tb["TB_over15"] = (grp_tb["slg_numeric"] > 1.5).astype(int)

    # Pre-game features (FIRST PA row per day)
    X_tb_feat, meta_tb = build_game_features(raw_ev_tb, tb_id_col)
    tb_df = grp_tb.merge(meta_tb, on=[tb_id_col, "game_date"], how="left").merge(
        X_tb_feat, left_index=True, right_index=True, how="left"
    )
    y_tb = tb_df["TB_over15"].astype(int)
    X_tb = tb_df.drop(columns=["slg_numeric","TB_over15"])

    # Keep only numeric features for model
    X_tb_num = X_tb.select_dtypes(include=[np.number]).copy().fillna(-1.0).astype(np.float32)

    dates_tb = pd.to_datetime(tb_df["game_date"])
    folds_tb = embargo_time_splits(dates_tb, n_splits=min(n_splits, max(2, dates_tb.nunique()//3)), embargo_days=1)

    P_tb = np.zeros(len(y_tb), dtype=np.float32)
    for fi,(tr_idx, va_idx) in enumerate(folds_tb):
        X_tr, X_va = X_tb_num.iloc[tr_idx], X_tb_num.iloc[va_idx]
        y_tr, y_va = y_tb.iloc[tr_idx], y_tb.iloc[va_idx]
        spw = max(1.0, (len(y_tr)-y_tr.sum())/max(1.0,y_tr.sum()))
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.035, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=fi
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        P_tb[va_idx] = clf.predict_proba(X_va)[:,1]

    ir_tb = IsotonicRegression(out_of_bounds="clip")
    p_tb_iso = ir_tb.fit_transform(P_tb, y_tb.values)
    Tbest_tb = tune_temperature_for_topk(p_tb_iso, y_tb.values, K=topK_eval, T_grid=np.linspace(0.9, 1.4, 11))
    p_tb = expit(logit(np.clip(p_tb_iso,1e-6,1-1e-6)) * Tbest_tb)

    st.write(f"TB>1.5 — AUC: {roc_auc_score(y_tb, p_tb):.4f} | Hits@{topK_eval}: {_hits_at_k(y_tb.values, p_tb, topK_eval)} | NDCG@30: {_ndcg_at_k(y_tb.values, p_tb, 30):.4f}")
else:
    p_tb = None
    st.info("Skipping TB: need columns game_date + slg_numeric + (batter_id or batter or player_name).")

# ======================================================================
#                     RBI > 1.5 — GAME-LEVEL MODEL (optional)
# ======================================================================
st.subheader("⚙️ RBI > 1.5 (game-level)")
raw_ev_rbi = _read_any(ev_file)
raw_ev_rbi = _safe_num_df(raw_ev_rbi)
rbi_id_col = tb_id_col  # use same ID if present

if rbi_id_col and "game_date" in raw_ev_rbi.columns and "rbi" in raw_ev_rbi.columns:
    raw_ev_rbi["game_date"] = pd.to_datetime(raw_ev_rbi["game_date"], errors="coerce").dt.date
    grp_rbi = raw_ev_rbi.groupby([rbi_id_col, "game_date"], as_index=False)["rbi"].sum()
    grp_rbi["RBI_over15"] = (grp_rbi["rbi"] > 1.5).astype(int)

    # Pre-game features (FIRST PA row per day)
    X_rbi_feat, meta_rbi = build_game_features(raw_ev_rbi, rbi_id_col)
    rbi_df = grp_rbi.merge(meta_rbi, on=[rbi_id_col, "game_date"], how="left").merge(
        X_rbi_feat, left_index=True, right_index=True, how="left"
    )
    y_rbi = rbi_df["RBI_over15"].astype(int)
    X_rbi = rbi_df.drop(columns=["rbi","RBI_over15"])

    X_rbi_num = X_rbi.select_dtypes(include=[np.number]).copy().fillna(-1.0).astype(np.float32)

    dates_rbi = pd.to_datetime(rbi_df["game_date"])
    folds_rbi = embargo_time_splits(dates_rbi, n_splits=min(n_splits, max(2, dates_rbi.nunique()//3)), embargo_days=1)

    P_rbi = np.zeros(len(y_rbi), dtype=np.float32)
    for fi,(tr_idx, va_idx) in enumerate(folds_rbi):
        X_tr, X_va = X_rbi_num.iloc[tr_idx], X_rbi_num.iloc[va_idx]
        y_tr, y_va = y_rbi.iloc[tr_idx], y_rbi.iloc[va_idx]
        spw = max(1.0, (len(y_tr)-y_tr.sum())/max(1.0,y_tr.sum()))
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.035, num_leaves=63,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
            reg_lambda=2.0, is_unbalance=True, n_jobs=1, random_state=fi
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
        P_rbi[va_idx] = clf.predict_proba(X_va)[:,1]

    ir_rbi = IsotonicRegression(out_of_bounds="clip")
    p_rbi_iso = ir_rbi.fit_transform(P_rbi, y_rbi.values)
    Tbest_rbi = tune_temperature_for_topk(p_rbi_iso, y_rbi.values, K=topK_eval, T_grid=np.linspace(0.9, 1.4, 11))
    p_rbi = expit(logit(np.clip(p_rbi_iso,1e-6,1-1e-6)) * Tbest_rbi)

    st.write(f"RBI>1.5 — AUC: {roc_auc_score(y_rbi, p_rbi):.4f} | Hits@{topK_eval}: {_hits_at_k(y_rbi.values, p_rbi, topK_eval)} | NDCG@30: {_ndcg_at_k(y_rbi.values, p_rbi, 30):.4f}")
else:
    p_rbi = None
    st.info("Skipping RBI: need columns game_date + rbi + (batter_id or batter or player_name).")

# ---------------- Export learned weights (for the main app) ----------------
st.subheader("💾 Export Best Weights (for your live app)")
col1, col2 = st.columns(2)
with col1:
    st.json(st.session_state.saved_best_mult, expanded=False)
with col2:
    st.json(st.session_state.saved_best_blend, expanded=False)

export_payload = {
    "multiplier_exponents": st.session_state.saved_best_mult,       # HR overlay exponents
    "blend_weights_hr":     st.session_state.saved_best_blend,      # HR blend weights
    "notes": "TB/RBI models trained at game-level with pre-game features; HR weights/exponents tuned from HR task."
}
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

st.caption("Cloud-safe offline tuner. Use JSON to seed your live prediction app. TB/RBI outputs here are true probabilities (calibrated).")
