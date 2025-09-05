# app.py — SUI-driven Shipping Volatility Forecaster (Streamlit)
# -----------------------------------------------------------------------------
# How to run (Windows):
#   1) Open PowerShell and cd into your project folder:

#        cd "C:\Users\Henrik Gjærum\PycharmProjects\pythonProject1"
#   2) Activate venv:
#        .\.venv\Scripts\activate
#   3) Install deps (one time):
#        pip install streamlit yfinance plotly scikit-learn arch requests python-dateutil pandas numpy openpyxl
#      If you get SSL/cert errors from yfinance:
#        pip install --upgrade certifi yfinance
#        $bundle = python -c "import certifi, sys; print(certifi.where())"
#        setx SSL_CERT_FILE "$bundle"
#        <close/reopen terminal, then retry>
#   4) Run:
#        streamlit run ".venv\Scripts\app.py"
#
# Optional: secrets file for Guardian/NYT counts (won’t crash if missing):
#   Create C:\Users\<YourUser>\.streamlit\secrets.toml with:
#     guardian_key = "<YOUR_GUARDIAN_KEY>"
#     nyt_key      = "<YOUR_NYT_KEY>"
# -----------------------------------------------------------------------------

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import streamlit as st

# ===============================
# Page Config & Theming
# ===============================
st.set_page_config(
    page_title="SUI → Shipping Volatility Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root { --radius: 16px; }
.block-container { padding-top: 1rem; }
section[data-testid="stSidebar"] { width: 340px !important; }

/**** Metric Cards ****/
.metric-card {
  border-radius: var(--radius);
  padding: 16px 18px;
  border: 1px solid rgba(0,0,0,0.08);
  background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(250,250,255,0.9));
  box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}
.metric-label { font-size: 0.85rem; opacity: 0.7; }
.metric-value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.metric-sub { font-size: 0.8rem; opacity: 0.6; margin-top: 6px; }

/**** Section Headers ****/
.hdr { font-size: 1.35rem; font-weight: 700; margin: 0.6rem 0 0.2rem; }
.subhdr { font-size: 0.95rem; opacity: 0.75; margin-bottom: 0.6rem; }

/**** Badges ****/
.badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; }
.badge-blue  { background: #e8f0ff; color: #2443b8; border: 1px solid #cfdaff; }
.badge-amber { background: #fff6e6; color: #a15c00; border: 1px solid #ffe4b5; }

/**** Tables ****/
.dataframe td, .dataframe th { font-size: 0.9rem; }

/**** Footnote ****/
.foot { font-size: 0.82rem; opacity: 0.65; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# Helpers
# ===============================
@st.cache_data(show_spinner=False)
def yf_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download Adjusted Close for multiple tickers. Returns wide DataFrame."""
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):  # single ticker
        df = df.to_frame()
    df = df.dropna(how="all")
    return df

@st.cache_data(show_spinner=False)
def realized_vol_from_returns(ret: pd.Series, h: int = 10) -> pd.Series:
    """h-day realized volatility (sqrt sum r^2), shifted forward by h."""
    x = ret.rolling(h).apply(lambda r: np.sqrt(np.sum(np.square(r))), raw=True)
    return x.shift(-h)

@st.cache_data(show_spinner=False)
def rolling_std(ret: pd.Series, w: int) -> pd.Series:
    return ret.rolling(w).std()

@st.cache_data(show_spinner=False)
def to_annualized(vol_h: pd.Series, h: int) -> pd.Series:
    return vol_h * np.sqrt(252 / h)

@st.cache_data(show_spinner=False)
def eq_weighted_return(prices: pd.DataFrame) -> pd.Series:
    rets = prices.pct_change()
    return rets.mean(axis=1).dropna()

@st.cache_data(show_spinner=False)
def best_effort_download_single(tickers: list[str], start: str, end: str) -> pd.Series:
    """Try a list of alternative tickers; return the first series that works."""
    for t in tickers:
        try:
            s = yf_download([t], start, end)
            if not s.empty:
                return s.iloc[:, 0].rename(t)
        except Exception:
            pass
    return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Find a date-like column and use as index
    for c in ["date", "Date", "DATE", "yyyymmdd", "time", "Time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c).sort_index()
            break
    return df

@st.cache_data(show_spinner=False)
def load_uploaded_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    for c in ["date", "Date", "DATE", "yyyymmdd", "time", "Time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c).sort_index()
            break
    return df

# --- Safe news counts (optional, won’t crash if secrets missing) ---
@st.cache_data(show_spinner=False)
def guardian_count(query: str, days_back: int = 1) -> int:
    """Return Guardian article count, or -1 if not configured."""
    try:
        if hasattr(st, "secrets") and "guardian_key" in st.secrets:
            key = st.secrets["guardian_key"]
        else:
            return -1
    except Exception:
        return -1
    from_dt = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        f"https://content.guardianapis.com/search?q={requests.utils.quote(query)}"
        f"&from-date={from_dt}&api-key={key}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return -1
        data = r.json()
        return int(data.get("response", {}).get("total", -1))
    except Exception:
        return -1

@st.cache_data(show_spinner=False)
def nyt_count(query: str, days_back: int = 1) -> int:
    """Return NYT hits, or -1 if not configured."""
    try:
        if hasattr(st, "secrets") and "nyt_key" in st.secrets:
            key = st.secrets["nyt_key"]
        else:
            return -1
    except Exception:
        return -1
    url = (
        f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={requests.utils.quote(query)}"
        f"&api-key={key}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return -1
        data = r.json()
        return int(data.get("response", {}).get("meta", {}).get("hits", -1))
    except Exception:
        return -1

# ===============================
# Sidebar — Inputs
# ===============================
with st.sidebar:
    st.title("📈 SUI → Volatility Forecaster")
    st.caption("Live market data + your SUI signal")

    st.markdown("**1) Universe**")
    default_tickers = [
        "ZIM",       # ZIM Integrated Shipping Services
        "SBLK",      # Star Bulk Carriers
        "GNK",       # Genco Shipping
        "DAC",       # Danaos
        "EURN",      # Euronav
        "STNG",      # Scorpio Tankers
        "KEX",       # Kirby Corp
        "MAERSK-B.CO"  # A.P. Moller - Maersk B (Copenhagen)
    ]
    tickers = st.text_input(
        "Shipping tickers (comma-separated)",
        value=", ".join(default_tickers),
        help="Use Yahoo Finance tickers. Paste your own list if you like."
    )
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    st.markdown("**2) Prediction horizon**")
    horizon = st.select_slider(
        "Horizon (trading days)", options=[5, 10, 22], value=10,
        help="5≈week, 10≈two weeks, 22≈month"
    )

    st.markdown("**3) Sample window**")
    yrs = st.slider("History length (years)", 1, 10, 5)
    end_dt = datetime.now().date()
    start_dt = end_dt - relativedelta(years=yrs)

    st.markdown("**4) Macro controls (auto-fetched)**")
    use_vix = st.checkbox("Include VIX", value=True)
    use_oil = st.checkbox("Include Oil (WTI)", value=True)
    use_usd = st.checkbox("Include USD Index (DX)", value=False)

    st.markdown("**5) Optional: Upload historical SUI**")
    sui_historical_file = st.file_uploader(
        "CSV/Excel with columns: date, sui (and optionally direction ∈ {−1,0,1})",
        type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )

    st.divider()
    st.markdown("**Guardian/NYT helpers (optional)**")
    st.caption("Toggle below only if you have API keys in secrets.toml")
    enable_news_counts = st.toggle("Show Guardian/NYT counts", value=False)

# ===============================
# Fetch Live Data (robust)
# ===============================
start = start_dt.strftime("%Y-%m-%d")
end = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")  # include today

with st.spinner("Pulling live market data…"):
    try:
        prices = yf_download(tickers, start, end)
    except Exception as e:
        st.error(f"Price download failed: {e}")
        prices = pd.DataFrame()

    # Macro series (close)
    try:
        vix = best_effort_download_single(["^VIX"], start, end) if use_vix else pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"VIX download issue: {e}")
        vix = pd.Series(dtype=float)
    try:
        oil = best_effort_download_single(["CL=F", "BZ=F"], start, end) if use_oil else pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"Oil download issue: {e}")
        oil = pd.Series(dtype=float)
    try:
        usd = best_effort_download_single(["DX=F", "DX-Y.NYB"], start, end) if use_usd else pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"USD download issue: {e}")
        usd = pd.Series(dtype=float)

if prices.empty:
    st.error("No price data returned. Check your tickers **or upload a price CSV** below.")
    up = st.file_uploader("Upload fallback prices (CSV): first column=Date, other columns=tickers", type=["csv"], key="price_csv")
    if up is not None:
        try:
            p = pd.read_csv(up)
            p.iloc[:, 0] = pd.to_datetime(p.iloc[:, 0])
            prices = p.set_index(p.columns[0]).sort_index()
        except Exception as e:
            st.error(f"Failed to read uploaded prices: {e}")
    if prices.empty:
        st.stop()

# ===============================
# Build Baseline Features (HAR)
# ===============================
returns = prices.pct_change().dropna(how="all")
market_ret = eq_weighted_return(prices)

rv_h = realized_vol_from_returns(market_ret, h=horizon)
rv_d = rolling_std(market_ret, 1)
rv_w = rolling_std(market_ret, 5)
rv_m = rolling_std(market_ret, 22)

features = pd.DataFrame({
    f"RV_{horizon}d_fwd": rv_h,
    "RV_d": rv_d,
    "RV_w": rv_w,
    "RV_m": rv_m,
})

if not vix.empty:
    features["VIX_level"] = vix.reindex(features.index).ffill()
if not oil.empty:
    oil_ret = oil.pct_change()
    features["Oil_ret_22d_vol"] = oil_ret.rolling(22).std()
if not usd.empty:
    usd_ret = usd.pct_change()
    features["USD_ret_22d_vol"] = usd_ret.rolling(22).std()

features = features.dropna()

# ===============================
# SUI Inputs (Current + Optional History)
# ===============================
st.markdown("<div class='hdr'>SUI Input</div>", unsafe_allow_html=True)
st.markdown("<div class='subhdr'>Enter the current SUI level and whether it increased or decreased since the last reading.</div>", unsafe_allow_html=True)

col_sui_1, col_sui_2, col_sui_3 = st.columns([1.2, 0.9, 1.3])
with col_sui_1:
    sui_now = st.number_input("Current SUI level", value=100.0, step=1.0, help="Use your live SUI reading.")
with col_sui_2:
    sui_dir = st.radio("Direction vs last reading", ["Increased", "Decreased", "Unchanged"], horizontal=True)
with col_sui_3:
    if enable_news_counts:
        g_hits = guardian_count("shipping OR freight")
        n_hits = nyt_count("shipping OR freight")
    else:
        g_hits = n_hits = -1
    g_badge = f"<span class='badge badge-blue'>Guardian: {g_hits if g_hits>=0 else 'n/a'}</span>"
    n_badge = f"<span class='badge badge-amber'>NYT: {n_hits if n_hits>=0 else 'n/a'}</span>"
    st.markdown(g_badge + "\u00A0" + n_badge, unsafe_allow_html=True)

# Direction as signal
if sui_dir == "Increased":
    sui_sign = 1
elif sui_dir == "Decreased":
    sui_sign = -1
else:
    sui_sign = 0

# Historical SUI (optional, enables model fitting/backtest)
sui_hist = None
if sui_historical_file is not None:
    if str(sui_historical_file.name).lower().endswith((".xlsx", ".xls")):
        raw = load_uploaded_excel(sui_historical_file)
    else:
        raw = load_uploaded_csv(sui_historical_file)
    # Normalize column names
    colmap = {}
    for c in raw.columns:
        lc = c.lower()
        if lc in ("sui", "index", "level"):
            colmap[c] = "sui"
        if lc in ("dir", "direction", "sign"):
            colmap[c] = "direction"
    if colmap:
        raw = raw.rename(columns=colmap)
    # If no explicit direction, infer from diff
    if "direction" not in raw.columns and "sui" in raw.columns:
        d = raw["sui"].diff().fillna(0)
        raw["direction"] = np.sign(d).astype(int)
    sui_hist = raw[[c for c in ["sui", "direction"] if c in raw.columns]].copy()

# ===============================
# Model Specification (HARX-style)
# ===============================
st.markdown("<div class='hdr'>Model Setup</div>", unsafe_allow_html=True)
st.markdown("<div class='subhdr'>Interpretable HARX baseline with SUI and macro controls. Tune priors or estimate from history.</div>", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    beta_d = st.number_input("β_d (RV_d)", value=0.35, step=0.05)
    beta_w = st.number_input("β_w (RV_w)", value=0.25, step=0.05)
    beta_m = st.number_input("β_m (RV_m)", value=0.30, step=0.05)
with col_m2:
    gamma_sui = st.number_input("γ (SUI z-score)", value=0.15, step=0.05)
    delta_dir = st.number_input("δ (Direction sign)", value=0.08, step=0.02)
    alpha = st.number_input("α (intercept)", value=0.00, step=0.01)
with col_m3:
    theta_vix = st.number_input("θ_VIX (per 10 pts)", value=0.05, step=0.01)
    theta_oil = st.number_input("θ_OilVol", value=0.10, step=0.02)
    theta_usd = st.number_input("θ_USDVol", value=0.05, step=0.01)

st.caption("Coefficients are interpretable weights on standardized inputs. Fit them from data below, or keep as priors.")

# ===============================
# Fit coefficients if SUI history provided
# ===============================
fit_results = None
if sui_hist is not None and not sui_hist.empty:
    df_fit = features.join(sui_hist, how="inner")
    # 52-week rolling z-score; fallback to global if short sample
    if len(df_fit) >= 52 * 5:
        sui_z = (df_fit["sui"] - df_fit["sui"].rolling(52).mean()) / df_fit["sui"].rolling(52).std()
        sui_z = sui_z.fillna((df_fit["sui"] - df_fit["sui"].mean()) / df_fit["sui"].std())
    else:
        sui_z = (df_fit["sui"] - df_fit["sui"].mean()) / df_fit["sui"].std()

    X_cols = ["RV_d", "RV_w", "RV_m"]
    if "VIX_level" in df_fit.columns:
        df_fit["VIX_10"] = df_fit["VIX_level"] / 10.0
        X_cols.append("VIX_10")
    if "Oil_ret_22d_vol" in df_fit.columns:
        X_cols.append("Oil_ret_22d_vol")
    if "USD_ret_22d_vol" in df_fit.columns:
        X_cols.append("USD_ret_22d_vol")

    X = df_fit[X_cols].copy()
    X["SUI_z"] = sui_z
    if "direction" in df_fit.columns:
        X["SUI_dir"] = df_fit["direction"].astype(float)
    y = df_fit[f"RV_{horizon}d_fwd"].copy()

    valid = X.dropna().index.intersection(y.dropna().index)
    X, y = X.loc[valid], y.loc[valid]

    if len(X) >= 50:
        ridge = Ridge(alpha=1e-3)
        ridge.fit(X, y)
        y_hat = pd.Series(ridge.predict(X), index=y.index)
        rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
        fit_results = {
            "model": ridge,
            "rmse": rmse,
            "coefs": pd.Series(ridge.coef_, index=X.columns),
            "intercept": float(ridge.intercept_),
            "insample_pred": y_hat,
            "y": y,
        }

        st.success("Estimated HARX coefficients from historical SUI (Ridge OLS).")
        with st.expander("View estimated coefficients"):
            coefs_df = fit_results["coefs"].to_frame("coef").copy()
            coefs_df.loc["Intercept"] = fit_results["intercept"]
            st.dataframe(coefs_df.style.format({"coef": "{:.4f}"}))
            st.caption(f"In-sample RMSE: {fit_results['rmse']:.6f} (on RV raw scale)")

        if st.toggle("Use estimated coefficients for prediction", value=True):
            alpha = fit_results["intercept"]
            for name, val in fit_results["coefs"].items():
                if name == "RV_d":
                    beta_d = float(val)
                elif name == "RV_w":
                    beta_w = float(val)
                elif name == "RV_m":
                    beta_m = float(val)
                elif name == "VIX_10":
                    theta_vix = float(val)
                elif name == "Oil_ret_22d_vol":
                    theta_oil = float(val)
                elif name == "USD_ret_22d_vol":
                    theta_usd = float(val)
                elif name == "SUI_z":
                    gamma_sui = float(val)
                elif name == "SUI_dir":
                    delta_dir = float(val)

# ===============================
# Compute Today’s Prediction
# ===============================
# Use only the predictor columns for selecting the latest valid row
predictor_cols = ["RV_d", "RV_w", "RV_m", "VIX_level", "Oil_ret_22d_vol", "USD_ret_22d_vol"]
predictor_cols = [c for c in predictor_cols if c in features.columns]

valid_rows = features.dropna(subset=predictor_cols)
if valid_rows.empty:
    st.error("Not enough data to compute predictors yet. Try increasing the history length or deselecting some macro controls.")
    st.stop()

latest = valid_rows.iloc[-1]
latest_idx = latest.name

# SUI z-score proxy for today
if sui_hist is not None and "sui" in sui_hist.columns:
    tmp = sui_hist["sui"].copy()
    tmp.loc[pd.to_datetime(latest_idx)] = sui_now
    if len(tmp) >= 52:
        sui_z_today = (sui_now - tmp.rolling(52).mean().iloc[-1]) / tmp.rolling(52).std().iloc[-1]
    else:
        sui_z_today = (sui_now - tmp.mean()) / tmp.std(ddof=0)
else:
    sui_z_today = 0.0  # neutral if no history

vix_10 = (latest.get("VIX_level", np.nan) / 10.0) if "VIX_level" in latest else 0.0
x_terms = (
    alpha
    + beta_d * latest["RV_d"]
    + beta_w * latest["RV_w"]
    + beta_m * latest["RV_m"]
    + gamma_sui * sui_z_today
    + delta_dir * float(sui_sign)
    + (theta_vix * vix_10 if use_vix else 0.0)
    + (theta_oil * latest.get("Oil_ret_22d_vol", 0.0) if use_oil else 0.0)
    + (theta_usd * latest.get("USD_ret_22d_vol", 0.0) if use_usd else 0.0)
)

pred_rv_h = max(float(x_terms), 0.0)
pred_rv_h_ann = float(to_annualized(pd.Series([pred_rv_h]), h=horizon).iloc[0])

naive_baseline = features[f"RV_{horizon}d_fwd"].dropna()
naive_today = float(naive_baseline.iloc[-2]) if len(naive_baseline) >= 2 else np.nan
naive_today_ann = float(to_annualized(pd.Series([naive_today]), h=horizon).iloc[0]) if not np.isnan(naive_today) else np.nan

# ===============================
# Headline Metrics
# ===============================
st.markdown("<div class='hdr'>Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subhdr'>HAR baseline + today’s SUI and macro inputs.</div>", unsafe_allow_html=True)

mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
with mcol1:
    st.markdown(
        "<div class='metric-card'>"
        f"<div class='metric-label'>Predicted {horizon}-day Realized Vol</div>"
        f"<div class='metric-value'>{pred_rv_h:.4f}</div>"
        f"<div class='metric-sub'>Raw (not annualized)</div>"
        "</div>", unsafe_allow_html=True
    )
with mcol2:
    st.markdown(
        "<div class='metric-card'>"
        f"<div class='metric-label'>Predicted Annualized Vol</div>"
        f"<div class='metric-value'>{pred_rv_h_ann:.2%}</div>"
        f"<div class='metric-sub'>Scaled from {horizon}d RV</div>"
        "</div>", unsafe_allow_html=True
    )
with mcol3:
    sub = f"Naive: {naive_today_ann:.2%}" if not np.isnan(naive_today_ann) else "Naive: n/a"
    st.markdown(
        "<div class='metric-card'>"
        f"<div class='metric-label'>Benchmark (Last-obs Annualized)</div>"
        f"<div class='metric-value'>{sub}</div>"
        f"<div class='metric-sub'>For quick comparison</div>"
        "</div>", unsafe_allow_html=True
    )

# ===============================
# Charts
# ===============================
hist_cols = ["RV_d", "RV_w", "RV_m"]
if "VIX_level" in features.columns:
    hist_cols.append("VIX_level")
fig1 = go.Figure()
for c in hist_cols:
    fig1.add_trace(go.Scatter(x=features.index, y=features[c], mode="lines", name=c))
fig1.update_layout(title="Baseline Features Over Time", height=360, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig1, use_container_width=True)

if fit_results is not None:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fit_results["y"].index, y=fit_results["y"], mode="lines", name="Actual RV_h"))
    fig2.add_trace(go.Scatter(x=fit_results["insample_pred"].index, y=fit_results["insample_pred"], mode="lines", name="Fitted RV_h"))
    fig2.update_layout(title=f"In-sample Fit (HARX with SUI) — h={horizon}d", height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# Per-Ticker Snapshot (today)
# ===============================
st.markdown("<div class='hdr'>Per-Ticker Snapshot</div>", unsafe_allow_html=True)
st.markdown("<div class='subhdr'>Today’s daily return and 22-day volatility. Prediction uses equal-weighted sector.</div>", unsafe_allow_html=True)

snap = pd.DataFrame(index=prices.columns)
last_close = prices.ffill().iloc[-1]
prev_close = prices.ffill().iloc[-2]
snap["Last Close"] = last_close
snap["1d Return"] = (last_close / prev_close) - 1.0
vol_22 = prices.pct_change().rolling(22).std().iloc[-1]
snap["22d Vol (daily)"] = vol_22
st.dataframe(snap.round(4))

# ===============================
# Export
# ===============================
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    out = pd.DataFrame({
        "asof": [pd.to_datetime(latest_idx)],
        "horizon_days": [horizon],
        "pred_rv_h_raw": [pred_rv_h],
        "pred_rv_h_annualized": [pred_rv_h_ann],
        "sui_now": [sui_now],
        "sui_direction_sign": [sui_sign],
    })
    csv = out.to_csv(index=False).encode()
    st.download_button("⬇️ Download today’s prediction (CSV)", csv, file_name="sui_vol_prediction_today.csv", mime="text/csv")

with exp_col2:
    hist_export = features.copy()
    hist_export[f"Pred_today_{horizon}d_RV_raw"] = pred_rv_h
    hist_export[f"Pred_today_{horizon}d_RV_ann"] = pred_rv_h_ann
    csv2 = hist_export.to_csv().encode()
    st.download_button("⬇️ Download features & prediction context (CSV)", csv2, file_name="sui_vol_features_context.csv", mime="text/csv")

# ===============================
# Methods Box (for thesis appendix)
# ===============================
st.divider()
st.markdown("<div class='hdr'>Methods (Summary)</div>", unsafe_allow_html=True)
methods_md = f"""
**Target.** Let $r_t$ be the equal-weighted daily return across selected shipping tickers.
Define the horizon-{horizon} **realized volatility** as $RV^{{({horizon})}}_t = \\sqrt{{\\sum_{{i=1}}^{{{horizon}}} r_{{t+i}}^2}}$.
We report both raw $RV^{{({horizon})}}_t$ and its annualized counterpart $RV^{{({horizon}),ann}}_t = RV^{{({horizon})}}_t\\,\\sqrt{{252/{horizon}}}$.

**Baseline (HAR).** We construct daily/weekly/monthly components $(RV_d, RV_w, RV_m)$ as rolling standard deviations of $r_t$ over 1, 5, and 22 days.

**SUI inputs.** The user enters the **current SUI level** and whether it **increased/decreased** relative to the last reading.
If a historical SUI series is uploaded, we compute a 52-week rolling z-score; otherwise we center SUI at zero for today (the coefficient $\\gamma$ controls sensitivity).

**Prediction (HARX).** We form a linear forecast for horizon-{horizon} RV:
$$\\widehat{{RV}}^{{({horizon})}}_t = \\alpha + \\beta_d RV_d + \\beta_w RV_w + \\beta_m RV_m + \\gamma\\,SUI_z + \\delta\\,\\text{{Dir}} + \\theta_{{VIX}}\\,VIX/10 + \\theta_{{Oil}}\\,\\sigma_{{22}}(\\Delta\\text{{Oil}}) + \\theta_{{USD}}\\,\\sigma_{{22}}(\\Delta\\text{{USD}}).$$

**Estimation.** If historical SUI is provided, we fit the coefficients by Ridge OLS (small $\\ell_2$ penalty) on the aligned sample.
Otherwise, sliders expose prior coefficients for transparent scenario analysis.

**Caveats.** This is a reduced-form forecast. If SUI is lower-frequency than prices, MIDAS or state-space variants can be added later without changing the UI.
"""
st.markdown(methods_md)

st.markdown("<div class='foot'>Data: Yahoo Finance (live each run). Optional news counts via Guardian/NYT APIs if secrets are set. All computations run locally.</div>", unsafe_allow_html=True)
