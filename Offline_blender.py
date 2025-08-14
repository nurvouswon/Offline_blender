# offline_tuner.py
# ============================================================
# 🧪 MLB HR / TB≥2 / RBI≥2 Offline Tuner (Cloud-Safe, No Sliders/Plots)
# - Inputs: Event-level CSV/Parquet (must include hr_outcome)
# - Optional: Batter & Pitcher season profiles (CSV) for weather-free overlay
# - Fixed: 5 folds, Hits@30 as primary metric
# - Models: XGB/LGB/CB (early stop) → LR meta → Isotonic + temp tuning
# - Tuners: Overlay exponents + Final blend weights (auto-run; cloud-safe sizes)
# - Output: Metrics and best weights JSON (to port into prediction app)
# ============================================================

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

# ---------- Fixed knobs (per your request) ----------
TOPK_EVAL = 30
N_SPLITS = 5
MULT_SAMPLES  = 2000   # cloud-safe; raise locally later if desired
BLEND_SAMPLES = 6000   # cloud-safe; raise locally later if desired

st.set_page_config(page_title="🧪 MLB Offline Tuner (HR/TB/RBI)", layout="wide")
st.title("🧪 MLB Offline Tuner — HR, TB≥2, RBI≥2 (Cloud-safe)")

DEFAULT_BLEND = dict(w_prob=0.30, w_overlay=0.20, w_ranker=0.20, w_rrf=0.10, w_penalty=0.20)
DEFAULT_MULT  = dict(a_batter=0.80, b_pitcher=0.80, c_platoon=0.60, d_park=0.40)

if "saved_best_blend" not in st.session_state:
    st.session_state.saved_best_blend = DEFAULT_BLEND.copy()
if "saved_best_mult" not in st.session_state:
    st.session_state.saved_best_mult  = DEFAULT_MULT.copy()

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

def _safe_num_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "O":
            try:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().mean() > 0.5:
                    df[c] = s
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

# ---------- Uploads ----------
with st.sidebar:
    st.header("📤 Upload Data")
    ev_file  = st.file_uploader("Event-level CSV/Parquet (must include hr_outcome)", type=["csv","parquet"], key="ev")
    bat_file = st.file_uploader("Season-long Batter Profile CSV (optional)", type=["csv"], key="bat")
    pit_file = st.file_uploader("Season-long Pitcher Profile CSV (optional)", type=["csv"], key="pit")

if ev_file is None:
    st.info("Upload an event-level dataset to begin.")
    st.stop()

# ---------- Load ----------
with st.spinner("Loading files..."):
    ev  = _read_any(ev_file)
    bat = _read_any(bat_file) if bat_file else pd.DataFrame()
    pit = _read_any(pit_file) if pit_file else pd.DataFrame()

ev  = _safe_num_cols(ev)
if not bat.empty: bat = _safe_num_cols(bat)
if not pit.empty: pit = _safe_num_cols(pit)

st.write(f"Event rows: {len(ev):,} | Batter rows: {len(bat):,} | Pitcher rows: {len(pit):,}")

# ---------- Derive RBI and Total Bases BEFORE dropping leak columns ----------
def derive_rbi(df: pd.DataFrame) -> pd.Series:
    if "rbi" in df.columns:
        s = pd.to_numeric(df["rbi"], errors="coerce").fillna(0).astype(int)
        return s.clip(lower=0, upper=4)
    if {"bat_score","post_bat_score"}.issubset(df.columns):
        s = (pd.to_numeric(df["post_bat_score"], errors="coerce") 
             - pd.to_numeric(df["bat_score"], errors="coerce"))
        return s.fillna(0).clip(lower=0, upper=4).astype(int)
    needs = {"home_score","away_score","post_home_score","post_away_score","inning_topbot"}
    if needs.issubset(df.columns):
        bot = df["inning_topbot"].astype(str).str.upper().str.startswith("B")
        inc_home = (pd.to_numeric(df["post_home_score"], errors="coerce") 
                    - pd.to_numeric(df["home_score"], errors="coerce")).fillna(0)
        inc_away = (pd.to_numeric(df["post_away_score"], errors="coerce") 
                    - pd.to_numeric(df["away_score"], errors="coerce")).fillna(0)
        s = np.where(bot, inc_home, inc_away)
        return pd.Series(s).clip(lower=0, upper=4).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

TB_MAP = {
    "single": 1, "double": 2, "triple": 3, "home_run": 4, "grand_slam": 4
}
def derive_tb(df: pd.DataFrame) -> pd.Series:
    evs = df.get("events", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
    return evs.map(TB_MAP).fillna(0).astype(int)

ev = ev.copy()
ev["rbi_derived"] = derive_rbi(ev)
ev["tb_derived"]  = derive_tb(ev)
ev["rbi_ge2"]     = (ev["rbi_derived"] >= 2).astype(int)
ev["tb_ge2"]      = (ev["tb_derived"]  >= 2).astype(int)

# ---------- Targets ----------
targets = {"hr_outcome": "HR", "tb_ge2": "TB≥2", "rbi_ge2": "RBI≥2"}
missing = [t for t in targets if t not in ev.columns]
if missing:
    st.error(f"Missing required targets in event file: {missing}")
    st.stop()

# ---------- Optional: Join season profiles (type-safe) ----------
if not bat.empty or not pit.empty:
    st.subheader("🔗 Joining Season Profiles")
    # pick plausible keys automatically (first match)
    ev_batter_key = next((c for c in ev.columns if "bat" in c or "player" in c or "id" in c), None)
    ev_pitcher_key = next((c for c in ev.columns if "pit" in c or "pitcher" in c or "id" in c), None)
    bat_key = bat.columns[0] if not bat.empty else None
    pit_key = pit.columns[0] if not pit.empty else None

    with st.spinner("Merging profiles…"):
        if not bat.empty and ev_batter_key and bat_key:
            bat_pref = bat.add_prefix("batprof_").rename(columns={f"batprof_{bat_key}":"bat_key_merge"})
            ev["bat_key_merge"] = ev[ev_batter_key].astype(str)
            bat_pref["bat_key_merge"] = bat_pref["bat_key_merge"].astype(str)
            ev = ev.merge(bat_pref, on="bat_key_merge", how="left")
        if not pit.empty and ev_pitcher_key and pit_key:
            pit_pref = pit.add_prefix("pitprof_").rename(columns={f"pitprof_{pit_key}":"pit_key_merge"})
            ev["pit_key_merge"] = ev[ev_pitcher_key].astype(str)
            pit_pref["pit_key_merge"] = pit_pref["pit_key_merge"].astype(str)
            ev = ev.merge(pit_pref, on="pit_key_merge", how="left")
    st.success("✅ Profiles merged.")

# ---------- Drop leak-prone columns AFTER deriving labels ----------
LEAK = {
    "post_away_score","post_home_score","post_bat_score","post_fld_score",
    "delta_home_win_exp","delta_run_exp","delta_pitcher_run_exp",
    "home_win_exp","bat_win_exp","home_score_diff","bat_score_diff",
    "estimated_ba_using_speedangle","estimated_woba_using_speedangle","estimated_slg_using_speedangle",
    "woba_value","woba_denom","babip_value","events","events_clean","slg_numeric",
    "launch_speed","launch_angle","hit_distance_sc","at_bat_number","pitch_number","game_pk"
}
ev = ev.drop(columns=[c for c in ev.columns if c in LEAK], errors="ignore")

# ---------- Feature matrix ----------
y_hr   = ev["hr_outcome"].astype(int).reset_index(drop=True)
y_tb2  = ev["tb_ge2"].astype(int).reset_index(drop=True)
y_rbi2 = ev["rbi_ge2"].astype(int).reset_index(drop=True)

dates = (pd.to_datetime(ev["game_date"], errors="coerce")
         if "game_date" in ev.columns else pd.Series(pd.Timestamp("2000-01-01"), index=ev.index)).fillna(pd.Timestamp("2000-01-01"))

num_cols = ev.select_dtypes(include=[np.number]).columns.tolist()
X_base = ev[num_cols].copy().replace([np.inf,-np.inf], np.nan).fillna(-1.0).astype(np.float32)

# ---------- Train base models (once per target) ----------
st.subheader("⚙️ Training Base Models (early stopping, 5 folds)")
folds = embargo_time_splits(dates, n_splits=N_SPLITS, embargo_days=1)
seeds = [42, 101, 202, 404]

def train_oof(X, y):
    P_xgb = np.zeros(len(y), dtype=np.float32)
    P_lgb = np.zeros(len(y), dtype=np.float32)
    P_cat = np.zeros(len(y), dtype=np.float32)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        xgb_va, lgb_va, cat_va = [], [], []
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
            xgb_va.append(xgb_clf.predict_proba(X_va)[:,1])
            lgb_va.append(lgb_clf.predict_proba(X_va)[:,1])
            cat_va.append(cat_clf.predict_proba(X_va)[:,1])
        P_xgb[va_idx] = np.mean(xgb_va, axis=0)
        P_lgb[va_idx] = np.mean(lgb_va, axis=0)
        P_cat[va_idx] = np.mean(cat_va, axis=0)
        dt = time.time() - t0
        st.write(f"Fold {fi+1}/{len(folds)} in {timedelta(seconds=int(dt))}")
    return P_xgb, P_lgb, P_cat

with st.spinner("Training base learners (HR)…"):
    P_xgb_hr, P_lgb_hr, P_cat_hr = train_oof(X_base, y_hr)
with st.spinner("Training base learners (TB≥2)…"):
    P_xgb_tb, P_lgb_tb, P_cat_tb = train_oof(X_base, y_tb2)
with st.spinner("Training base learners (RBI≥2)…"):
    P_xgb_rbi, P_lgb_rbi, P_cat_rbi = train_oof(X_base, y_rbi2)

# ---------- Day-wise Ranker (optional if multiple days) ----------
st.subheader("📈 Day-wise Ranker")
def train_ranker_oof(X, y):
    if dates.nunique() <= 1:
        return np.zeros(len(y), dtype=np.float32)
    days = pd.to_datetime(dates).dt.floor("D")
    def _groups_from_days(d): return d.groupby(d.values).size().values.tolist()
    rk_oof = np.zeros(len(y), dtype=np.float32)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
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
        rk_oof[va_idx] = rk.predict(X_va)
    return rk_oof

with st.spinner("Training rankers…"):
    rk_hr  = train_ranker_oof(X_base, y_hr)
    rk_tb  = train_ranker_oof(X_base, y_tb2)
    rk_rbi = train_ranker_oof(X_base, y_rbi2)

# ---------- Meta + Calibration (per target) ----------
st.subheader("🧮 Meta + Isotonic + Top-30 Temp Tuning")
def stack_and_calibrate(Px, Pl, Pc, y, label):
    Xm = np.column_stack([Px, Pl, Pc]).astype(np.float32)
    sc = StandardScaler()
    Xm_s = sc.fit_transform(Xm)
    meta = LogisticRegression(max_iter=1000, solver="lbfgs")
    meta.fit(Xm_s, y.values)
    oof = meta.predict_proba(Xm_s)[:,1]
    auc = roc_auc_score(y, oof); ll = log_loss(y, oof)
    st.write(f"{label} — OOF AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    iso = IsotonicRegression(out_of_bounds="clip")
    y_iso = iso.fit_transform(oof, y.values)
    Tbest = tune_temperature_for_topk(y_iso, y.values, K=TOPK_EVAL, T_grid=np.linspace(0.8, 1.6, 17))
    logits = logit(np.clip(y_iso, 1e-6, 1-1e-6))
    p_base = expit(logits * Tbest)
    return p_base, (Px, Pl, Pc)

p_hr,  base_hr  = stack_and_calibrate(P_xgb_hr,  P_lgb_hr,  P_cat_hr,  y_hr,   "HR")
p_tb,  base_tb  = stack_and_calibrate(P_xgb_tb,  P_lgb_tb,  P_cat_tb,  y_tb2,  "TB≥2")
p_rbi, base_rbi = stack_and_calibrate(P_xgb_rbi, P_lgb_rbi, P_cat_rbi, y_rbi2, "RBI≥2")

# ---------- Weather-free Overlay (optional via profiles) ----------
st.subheader("🧩 Overlay from Season Profiles (weather-free)")
bat_cols = dict(barrel="batprof_barrel_rate", fb="batprof_fb_rate", pull="batprof_pull_rate", hr_pa="batprof_hr_per_pa")
pit_cols = dict(brl_allowed="pitprof_barrel_rate_allowed", fb="pitprof_fb_rate", bb="pitprof_bb_rate", xwoba_con="pitprof_xwoba_con")
platoon_cols = dict(
    batter_vsL="batprof_hr_pa_vsL", batter_vsR="batprof_hr_pa_vsR",
    pitcher_vsL="pitprof_hr_pa_vsL", pitcher_vsR="pitprof_hr_pa_vsR",
    stand="stand", batter_hand="batter_hand", p_throws="pitcher_hand"
)
park_cols = dict(park_factor="park_hr_rate")

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

def _park_factor(row):
    pf = row.get(park_cols["park_factor"], np.nan)
    try:
        return float(np.clip(float(pf), 0.85, 1.20))
    except Exception:
        return 1.0

if not bat.empty or not pit.empty:
    bf   = ev.apply(_batter_factor, axis=1)
    pfct = ev.apply(_pitcher_factor, axis=1)
    pltf = ev.apply(_platoon_factor, axis=1)
    pkf  = ev.apply(_park_factor, axis=1)
else:
    bf = pfct = pltf = pkf = pd.Series(1.0, index=ev.index, dtype=float)

def overlay_from_exponents(a_b, b_p, c_pl, d_pk):
    ov = (bf**a_b) * (pfct**b_p) * (pltf**c_pl) * (pkf**d_pk)
    return np.asarray(np.clip(ov, 0.80, 1.40), dtype=np.float32)

# ---------- Multiplier Tuner (auto-run; anchor on HR) ----------
use_mult = st.session_state.saved_best_mult.copy()
rng = np.random.default_rng(123)
best_key = None; best_res = None
for _ in range(MULT_SAMPLES):
    a_b = float(rng.uniform(0.2, 1.6))
    b_p = float(rng.uniform(0.2, 1.6))
    c_pl= float(rng.uniform(0.0, 1.2))
    d_pk= float(rng.uniform(0.0, 1.2))
    ov  = overlay_from_exponents(a_b, b_p, c_pl, d_pk)
    eval_score = expit(logit(np.clip(p_hr,1e-6,1-1e-6)) + np.log(ov+1e-9))
    order = np.argsort(-eval_score)
    hK = int(y_hr.values[order][:TOPK_EVAL].sum())
    rel_sorted = y_hr.values[order]
    discounts = 1.0/np.log2(np.arange(2, 2+min(30,len(rel_sorted))))
    dcg = float((rel_sorted[:len(discounts)]*discounts).sum())
    ideal = np.sort(y_hr.values)[::-1]
    idcg = float((ideal[:len(discounts)]*discounts).sum())
    nd = (dcg/idcg) if idcg>0 else 0.0
    key = (hK, nd)
    if (best_key is None) or (key > best_key):
        best_key = key
        best_res = dict(a_batter=a_b, b_pitcher=b_p, c_platoon=c_pl, d_park=d_pk, HitsAtK=hK, NDCG30=nd)

if best_res:
    st.session_state.saved_best_mult = {k:best_res[k] for k in ["a_batter","b_pitcher","c_platoon","d_park"]}
    use_mult = st.session_state.saved_best_mult.copy()
    st.success(f"Overlay exponents: {json.dumps(use_mult, indent=2)} | HR Hits@{TOPK_EVAL}={best_res['HitsAtK']} NDCG@30={best_res['NDCG30']:.4f}")

overlay = overlay_from_exponents(use_mult["a_batter"], use_mult["b_pitcher"], use_mult["c_platoon"], use_mult["d_park"])
log_overlay = np.log(overlay + 1e-9)

# ---------- RRF + disagreement + blend ----------
def blended_oof(p_base, Px, Pl, Pc, ranker_oof):
    disagree_std = np.std(np.vstack([Px, Pl, Pc]), axis=0)
    dis_penalty = np.clip(zscore(disagree_std), 0, 3)
    r_prob    = _rank_desc(p_base)
    r_ranker  = _rank_desc(zscore(ranker_oof))
    r_overlay = _rank_desc(overlay)
    k_rrf = 60.0
    rrf = 1.0/(k_rrf + r_prob) + 1.0/(k_rrf + r_ranker) + 1.0/(k_rrf + r_overlay)
    rrf_z = zscore(rrf)
    return dis_penalty, rrf_z, zscore(ranker_oof)

def blend_with_weights(wp, wo, wr, wrrf, wpen, logit_p, log_overlay, ranker_z, rrf_z, dis_penalty):
    return expit(wp*logit_p + wo*log_overlay + wr*ranker_z + wrrf*rrf_z - wpen*dis_penalty)

def tune_blend(label, p_base, base_triplet, y, ranker_oof):
    Px, Pl, Pc = base_triplet
    dis_penalty, rrf_z, ranker_z = blended_oof(p_base, Px, Pl, Pc, ranker_oof)
    use_blend = st.session_state.saved_best_blend.copy()
    logit_p = logit(np.clip(p_base, 1e-6, 1-1e-6))
    rng = np.random.default_rng(777)
    best_tup=None; best_row=None
    for _ in range(BLEND_SAMPLES):
        w = rng.dirichlet(np.ones(5))
        s = blend_with_weights(w[0], w[1], w[2], w[3], w[4], logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
        order = np.argsort(-s)
        hK = int(y.values[order][:TOPK_EVAL].sum())
        rel_sorted = y.values[order]
        discounts = 1.0/np.log2(np.arange(2, 2+min(30,len(rel_sorted))))
        dcg = float((rel_sorted[:len(discounts)]*discounts).sum())
        ideal = np.sort(y.values)[::-1]
        idcg = float((ideal[:len(discounts)]*discounts).sum())
        nd = (dcg/idcg) if idcg>0 else 0.0
        tup=(hK, nd)
        if (best_tup is None) or (tup > best_tup):
            best_tup=tup
            best_row=dict(w_prob=float(w[0]), w_overlay=float(w[1]), w_ranker=float(w[2]), w_rrf=float(w[3]), w_penalty=float(w[4]),
                          HitsAtK=int(hK), NDCG30=float(nd))
    if best_row:
        st.session_state.saved_best_blend = {k:best_row[k] for k in ["w_prob","w_overlay","w_ranker","w_rrf","w_penalty"]}
        use_blend = st.session_state.saved_best_blend.copy()
        st.success(f"{label} blend: {json.dumps(use_blend, indent=2)} | Hits@{TOPK_EVAL}={best_row['HitsAtK']} NDCG@30={best_row['NDCG30']:.4f}")

    final = blend_with_weights(use_blend["w_prob"], use_blend["w_overlay"], use_blend["w_ranker"],
                               use_blend["w_rrf"], use_blend["w_penalty"],
                               logit_p, log_overlay, ranker_z, rrf_z, dis_penalty)
    return final

st.subheader("🧪 Blended Tuning (auto, K=30, folds=5)")
final_hr  = tune_blend("HR",   p_hr,  base_hr,  y_hr,   rk_hr)
final_tb  = tune_blend("TB≥2", p_tb,  base_tb,  y_tb2,  rk_tb)
final_rbi = tune_blend("RBI≥2",p_rbi, base_rbi, y_rbi2, rk_rbi)

# ---------- Reports (no plots) ----------
def hits_at_k(y, s, K):
    order = np.argsort(-s)
    return int(y.values[order][:K].sum())

st.markdown("### 📊 Final Diagnostics")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**HR**")
    st.write(f"Hits@30: {hits_at_k(y_hr, final_hr, TOPK_EVAL)}")
    try: st.write(f"AUC: {roc_auc_score(y_hr, final_hr):.4f}")
    except: pass
with col2:
    st.write("**TB≥2**")
    st.write(f"Hits@30: {hits_at_k(y_tb2, final_tb, TOPK_EVAL)}")
    try: st.write(f"AUC: {roc_auc_score(y_tb2, final_tb):.4f}")
    except: pass
with col3:
    st.write("**RBI≥2**")
    st.write(f"Hits@30: {hits_at_k(y_rbi2, final_rbi, TOPK_EVAL)}")
    try: st.write(f"AUC: {roc_auc_score(y_rbi2, final_rbi):.4f}")
    except: pass

# ---------- Export best weights ----------
st.subheader("💾 Export Best Weights")
export_payload = {
    "multiplier_exponents": st.session_state.saved_best_mult,
    "blend_weights": st.session_state.saved_best_blend,
    "notes": f"Weather-free overlay exponents + final blend weights (K={TOPK_EVAL}, folds={N_SPLITS}). Targets: HR, TB≥2, RBI≥2."
}
st.json(export_payload, expanded=False)
st.download_button(
    "⬇️ Download Weights JSON",
    data=json.dumps(export_payload, indent=2),
    file_name="offline_tuner_best_weights.json",
    mime="application/json"
)

st.caption("Streamlit Cloud safe: fixed K=30, 5 folds, early stopping, automatic tuners, and no plots.")
