# Sui_index.py — SUI → Shipping Returns Forecaster (pretty & insightful)
# -----------------------------------------------------------------------------
# Upload SUI (date, sui[, direction]) → fetch prices → build SUI features →
# walk-forward OOS (Ridge & Logistic) → KPIs, heatmaps, ROC, correlations,
# per-ticker scores, strategy equity curve, and CSV exports.
# -----------------------------------------------------------------------------

from __future__ import annotations
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -------------------- Page & theme --------------------
st.set_page_config(page_title="SUI → Shipping Returns Forecaster", page_icon="🧭", layout="wide")
st.markdown("""
<style>
:root { --radius: 16px; }
.block-container { padding-top: 0.6rem; }
section[data-testid="stSidebar"] { width: 360px !important; }

/* Cards */
.card { border-radius: var(--radius); padding: 14px 16px; border: 1px solid rgba(0,0,0,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(247,248,255,.96));
        box-shadow: 0 6px 20px rgba(0,0,0,0.05); }
.kpi-label { font-size: .85rem; opacity: .72; }
.kpi-value { font-size: 1.55rem; font-weight: 700; margin-top: 2px; }
.kpi-sub { font-size: .82rem; opacity: .65; margin-top: 4px; }
.hdr { font-weight: 700; font-size: 1.20rem; margin: .75rem 0 .35rem; }
.sub { font-size: .92rem; opacity: .75; margin-bottom: .35rem; }
.small { font-size: .85rem; opacity: .75; }
.badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size:.78rem;
         background:#eef3ff; border:1px solid #dbe5ff; color:#233a94; }
</style>
""", unsafe_allow_html=True)

# -------------------- Utilities --------------------
PLOT_TEMPLATE = "plotly_white"

@st.cache_data(show_spinner=False)
def yf_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return (df if isinstance(df, pd.DataFrame) else df.to_frame()).dropna(how="all")

def eq_weight_returns(prices: pd.DataFrame) -> pd.Series:
    return prices.pct_change().mean(axis=1)

def forward_sum_return(prc: pd.Series, h: int) -> pd.Series:
    r = prc.pct_change()
    return r.rolling(h).sum().shift(-h)

def asof_align(index: pd.DatetimeIndex, dated_series: pd.Series) -> pd.Series:
    s = dated_series.sort_index()
    return s.reindex(index, method="ffill")

@st.cache_data(show_spinner=False)
def load_sui(file) -> pd.DataFrame:
    name = str(file.name).lower()
    df = pd.read_excel(file) if name.endswith((".xlsx",".xls")) else pd.read_csv(file)
    # date column guess
    date_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ["date","dato","month","period","yearmonth","yyyymmdd","time","tid","måned","maned"]):
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
    dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    if dt.isna().all():
        for fmt in ("%d.%m.%Y","%d/%m/%Y","%Y-%m-%d","%d.%m.%y","%d/%m/%y"):
            dt = pd.to_datetime(df[date_col].astype(str), format=fmt, errors="coerce");
            if dt.notna().any(): break
    # sui column
    sui_col = None
    for c in df.columns:
        if str(c).lower() in ("sui","index","level","value","score"):
            sui_col = c; break
    if sui_col is None:
        num = [c for c in df.columns if c!=date_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num: raise ValueError("Could not find a numeric SUI column.")
        sui_col = num[0]
    out = pd.DataFrame({"date": dt, "sui": pd.to_numeric(df[sui_col], errors="coerce")}).dropna()
    out = out.sort_values("date").set_index("date")
    # direction optional
    if "direction" in [c.lower() for c in df.columns]:
        dcol = [c for c in df.columns if c.lower()=="direction"][0]
        d = pd.to_numeric(df[dcol], errors="coerce").fillna(0).clip(-1,1).astype(int)
        out["direction"] = d.values[:len(out)]
    else:
        out["direction"] = np.sign(out["sui"].diff()).fillna(0).astype(int)
    return out

def kpi_card(label: str, value: str, sub: str = ""):
    st.markdown(f"<div class='card'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div>"
                f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

def df_download_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, df.to_csv(index=True).encode(), file_name=filename, mime="text/csv")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🧭 SUI → Returns Forecaster")
    st.caption("Pretty, insightful graphs & tables for your thesis")

    tickers_default = "SBLK, STNG, EURN, GNK, DAC, ZIM, KEX, MAERSK-B.CO"
    tickers = st.text_input("Shipping tickers (Yahoo Finance)", value=tickers_default)
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    horizon = st.select_slider("Prediction horizon (trading days)", options=[1,5,10,22], value=5)
    sui_lag_days = st.slider("Assumed publication lag (trading days)", 0, 22, 5)
    hist_years = st.slider("Market history (years)", 5, 25, 16)

    st.markdown("---")
    sui_file = st.file_uploader("Upload SUI (CSV or Excel)", type=["csv","xlsx","xls"])
    st.caption("Columns: **date**, **sui**. Optional: **direction** (−1/0/1).")

# -------------------- Data --------------------
if sui_file is None:
    st.info("⬆️ Upload your SUI to begin.")
    st.stop()

try:
    sui_df = load_sui(sui_file)
except Exception as e:
    st.error(f"Failed to read SUI: {e}")
    st.stop()

start = (datetime.today().date() - relativedelta(years=hist_years)).strftime("%Y-%m-%d")
end   = (datetime.today().date() + relativedelta(days=1)).strftime("%Y-%m-%d")

with st.spinner("Downloading shipping prices…"):
    prices = yf_prices(tickers, start, end)
if prices.empty:
    st.error("No price data. Check tickers or extend the history window.")
    st.stop()

daily_index = prices.index
sui_level = asof_align(daily_index, sui_df["sui"]).shift(sui_lag_days)

# SUI features
sui_z = (sui_level - sui_level.rolling(252).mean()) / sui_level.rolling(252).std()
sui_chg_1m = sui_level.pct_change(22)
sui_chg_3m = sui_level.pct_change(66)
sui_surprise = sui_level - sui_level.rolling(252).mean()

features = pd.DataFrame({
    "SUI_level": sui_level,
    "SUI_z": sui_z,
    "SUI_chg_1m": sui_chg_1m,
    "SUI_chg_3m": sui_chg_3m,
    "SUI_surprise": sui_surprise
}, index=daily_index).dropna()

# Targets
rets = prices.pct_change()
ew_ret = rets.mean(axis=1)
ew_fwd = ew_ret.rolling(horizon).sum().shift(-horizon)  # forward horizon return
y_dir = (ew_fwd > 0).astype(int)

# Align
df_all = features.join(ew_fwd.rename("EW_fwd_ret"), how="inner").dropna()
if df_all.empty:
    st.error("No overlap between SUI (after lag) and returns. Increase history or reduce lag.")
    st.stop()

# -------------------- KPIs (quick view) --------------------
c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("SUI latest (level)", f"{features['SUI_level'].iloc[-1]:.1f}", "After lag & alignment")
with c2: kpi_card("SUI z-score (1y)", f"{features['SUI_z'].iloc[-1]:.2f}")
with c3: kpi_card(f"EW {horizon}d fwd ret (last obs)", f"{df_all['EW_fwd_ret'].iloc[-1]:.2%}")
with c4: kpi_card("Sample (obs)", f"{len(df_all):,}")

# -------------------- Correlations --------------------
st.markdown("<div class='hdr'>Correlation insights</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Static and rolling correlations between SUI features and forward returns.</div>", unsafe_allow_html=True)

corr_cols = ["SUI_level","SUI_z","SUI_chg_1m","SUI_chg_3m","SUI_surprise","EW_fwd_ret"]
C = df_all[corr_cols].corr()
fig_heat = px.imshow(
    C.loc[["SUI_level","SUI_z","SUI_chg_1m","SUI_chg_3m","SUI_surprise"], ["EW_fwd_ret"]],
    text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", origin="lower", template=PLOT_TEMPLATE
)
fig_heat.update_layout(title=f"Correlation heatmap vs EW forward return (h={horizon}d)", height=320, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_heat, use_container_width=True)

roll_w = st.slider("Rolling window (days) for correlation", 60, 504, 252, step=21)
rc = df_all["SUI_z"].rolling(roll_w).corr(df_all["EW_fwd_ret"])
fig_rc = go.Figure()
fig_rc.add_trace(go.Scatter(x=rc.index, y=rc, mode="lines", name="Rolling corr (SUI_z vs EW fwd)", hovertemplate="%{x|%Y-%m-%d}<br>corr=%{y:.2f}<extra></extra>"))
fig_rc.add_hline(y=0, line_dash="dash", opacity=0.4)
fig_rc.update_layout(template=PLOT_TEMPLATE, title=f"Rolling correlation (window={roll_w}d)", height=300, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_rc, use_container_width=True)

# -------------------- Modeling (walk-forward) --------------------
st.markdown("<div class='hdr'>Walk-forward forecasting</div>", unsafe_allow_html=True)
split_date = st.date_input("Train end date", value=(daily_index.min() + relativedelta(years=10)))
split_dt = pd.Timestamp(split_date)

X = df_all[["SUI_z","SUI_chg_1m","SUI_chg_3m","SUI_surprise"]]
y_reg = df_all["EW_fwd_ret"]
y_clf = (y_reg > 0).astype(int)

Xtr, Xte = X[X.index <= split_dt], X[X.index > split_dt]
ytr_reg, yte_reg = y_reg.loc[Xtr.index], y_reg.loc[Xte.index]
ytr_clf, yte_clf = y_clf.loc[Xtr.index], y_clf.loc[Xte.index]

if len(Xte) < 50 or len(Xtr) < 200:
    st.warning("Small sample after split; consider moving the Train end date.")

scaler = StandardScaler().fit(Xtr)
Xs_tr = pd.DataFrame(scaler.transform(Xtr), index=Xtr.index, columns=Xtr.columns)
Xs_te = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)

reg = Ridge(alpha=1e-3).fit(Xs_tr, ytr_reg)
clf = LogisticRegression(max_iter=1000).fit(Xs_tr, ytr_clf)

y_reg_pred = pd.Series(reg.predict(Xs_te), index=Xs_te.index)
proba = pd.Series(clf.predict_proba(Xs_te)[:,1], index=Xs_te.index)
pred_dir = (proba >= 0.5).astype(int)

auc = roc_auc_score(yte_clf, proba) if yte_clf.nunique() > 1 else np.nan
acc = accuracy_score(yte_clf, pred_dir)

k1, k2, k3 = st.columns(3)
with k1: kpi_card("Directional accuracy (EW)", f"{acc:.2%}", f"h={horizon}d, threshold=0.5")
with k2: kpi_card("ROC AUC (EW)", f"{auc:.3f}", "0.5 = random")
with k3: kpi_card("Mean OOS return when LONG", f"{y_reg_pred[pred_dir==1].mean():.2%}")

# ROC curve
fpr, tpr, _ = roc_curve(yte_clf, proba) if yte_clf.nunique() > 1 else (np.array([0,1]), np.array([0,1]), None)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})", hovertemplate="FPR=%{x:.2f}<br>TPR=%{y:.2f}<extra></extra>"))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
fig_roc.update_layout(template=PLOT_TEMPLATE, title="ROC curve (EW)", height=300, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_roc, use_container_width=True)

# Confusion matrix
cm = confusion_matrix(yte_clf, pred_dir, labels=[0,1])
fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto", template=PLOT_TEMPLATE)
fig_cm.update_layout(title="Confusion matrix (EW direction)", xaxis_title="Predicted", yaxis_title="Actual", height=300, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_cm, use_container_width=True)

# Feature importances
imp = pd.Series(reg.coef_, index=X.columns).abs().sort_values(ascending=True)
fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(x=imp.values, y=imp.index, orientation="h", hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
fig_imp.update_layout(template=PLOT_TEMPLATE, title="Feature importance (|Ridge coef| on standardized features)", height=320, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_imp, use_container_width=True)

# -------------------- Threshold strategy backtest --------------------
st.markdown("<div class='hdr'>Threshold strategy backtest (EW)</div>", unsafe_allow_html=True)
p_hi = st.slider("Go LONG if P(up) ≥", 0.50, 0.80, 0.55, step=0.01)
p_lo = st.slider("Go SHORT if P(up) ≤", 0.20, 0.50, 0.45, step=0.01)

sig = pd.Series(0, index=proba.index)
sig[proba >= p_hi] = 1
sig[proba <= p_lo] = -1
strategy_ret = sig * yte_reg  # realized forward return over the OOS window
equity = (1 + strategy_ret).cumprod()

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines", name="Strategy", hovertemplate="%{x|%Y-%m-%d}<br>Equity=%{y:.3f}<extra></extra>"))
fig_eq.add_hline(y=1.0, line_dash="dash")
fig_eq.update_layout(template=PLOT_TEMPLATE, title=f"Equity curve (h={horizon}d)", height=320, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig_eq, use_container_width=True)

st.caption(f"Hit when LONG: {((sig==1) & (yte_reg>0)).mean():.2%} • Hit when SHORT: {((sig==-1) & (yte_reg<0)).mean():.2%} • Trades: {(sig!=0).sum():,}")

# Exports
exp1, exp2, exp3 = st.columns(3)
with exp1:
    out_preds = pd.DataFrame({"proba_up": proba, "pred_dir": pred_dir, "fwd_ret": yte_reg, "signal": sig})
    df_download_button(out_preds, "⬇️ Download OOS predictions (CSV)", "oos_predictions.csv")
with exp2:
    df_download_button(features, "⬇️ Download SUI features (CSV)", "sui_features.csv")
with exp3:
    df_download_button(prices, "⬇️ Download prices (CSV)", "prices.csv")

# -------------------- Per-ticker section --------------------
st.markdown("<div class='hdr'>Per-ticker directional accuracy</div>", unsafe_allow_html=True)
rows = []
for t in prices.columns:
    fwd = forward_sum_return(prices[t], horizon)
    df_t = features.join(fwd.rename("fwd"), how="inner").dropna()
    if df_t.empty:
        rows.append((t, np.nan, np.nan)); continue
    Xt = df_t[["SUI_z","SUI_chg_1m","SUI_chg_3m","SUI_surprise"]]
    yt = (df_t["fwd"] > 0).astype(int)
    Xtr_t, Xte_t = Xt[Xt.index <= split_dt], Xt[Xt.index > split_dt]
    ytr_t, yte_t = yt.loc[Xtr_t.index], yt.loc[Xte_t.index]
    if len(Xte_t) < 30 or yte_t.nunique() < 2:
        rows.append((t, np.nan, np.nan)); continue
    sc_t = StandardScaler().fit(Xtr_t)
    clf_t = LogisticRegression(max_iter=1000).fit(sc_t.transform(Xtr_t), ytr_t)
    proba_t = clf_t.predict_proba(sc_t.transform(Xte_t))[:,1]
    pred_t = (proba_t >= 0.5).astype(int)
    acc_t = accuracy_score(yte_t, pred_t)
    auc_t = roc_auc_score(yte_t, proba_t) if yte_t.nunique() > 1 else np.nan
    rows.append((t, acc_t, auc_t))

per_ticker = pd.DataFrame(rows, columns=["Ticker","Accuracy","AUC"]).set_index("Ticker")
colA, colB = st.columns([2,1])
with colA:
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=per_ticker.index, y=per_ticker["Accuracy"], hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig_bar.update_layout(template=PLOT_TEMPLATE, title="Per-ticker accuracy (OOS)", yaxis_tickformat=".0%", height=340, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
with colB:
    st.dataframe(per_ticker.style.format({"Accuracy": "{:.2%}","AUC": "{:.3f}"}))

df_download_button(per_ticker, "⬇️ Download per-ticker metrics (CSV)", "per_ticker_metrics.csv")

# -------------------- Today’s signal --------------------
st.markdown("<div class='hdr'>Today’s live signal</div>", unsafe_allow_html=True)
x_live = features[["SUI_z","SUI_chg_1m","SUI_chg_3m","SUI_surprise"]].iloc[[-1]]
prob_up_live = float(clf.predict_proba(scaler.transform(x_live))[:,1])
signal_live = "LONG" if prob_up_live >= p_hi else ("SHORT" if prob_up_live <= p_lo else "FLAT")
st.markdown(f"<span class='badge'>EW Prob(up in next {horizon}d): {prob_up_live:.2%} • Signal: {signal_live}</span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
**Methods.** SUI (2010–2025) is aligned to trading days via ASOF, with an assumed publication lag of **{sui_lag_days}** trading days.
We build features: **SUI z-score** (1y), **1m/3m % change**, and **surprise** (level − 1y mean).  
Targets are equal-weighted forward **{horizon}-day returns**. We split by date, standardize on train only, and fit Ridge (returns) & Logistic (direction).  
We report static & rolling correlations, ROC/AUC, confusion matrix, feature importances, per-ticker accuracy, and a thresholded long/short backtest with equity curve.
""")
