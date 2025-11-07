import os
import re
import json
import gzip
import zipfile
import chardet
import base64
from io import BytesIO
from datetime import datetime, date

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Prophet import (fallback logic)
USE_PROPHET = False
try:
    from prophet import Prophet
    USE_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet
        USE_PROPHET = True
    except Exception:
        USE_PROPHET = False

load_dotenv()
pio.templates.default = "plotly_white"

st.set_page_config(page_title="📦 Business Insight Hub", layout="wide")
st.title("📦 Business Insight Hub Dashboard")
st.markdown('Upload your business data, analyze trends, and generate accurate sales forecasts for data-driven decisions.</div>', unsafe_allow_html=True)

# ========= Simple UI==============
st.markdown("""
<style>
.one-liner { font-size:13px; color:#555; margin-bottom:8px; }
.small-muted { font-size:12px; color:#6c6c6c; }
</style>
""", unsafe_allow_html=True)


# ---------------- Utilities ----------------
def read_any_file(uploaded_file):
    """Read CSV/Excel/JSON/GZ/ZIP/PDF (tables) robustly."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".gz"):
            raw = gzip.open(uploaded_file, "rb").read()
            enc = chardet.detect(raw).get("encoding") or "utf-8"
            return pd.read_csv(BytesIO(raw), encoding=enc)
        if name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                first = z.namelist()[0]
                with z.open(first) as f:
                    raw = f.read()
                enc = chardet.detect(raw).get("encoding") or "utf-8"
                return pd.read_csv(BytesIO(raw), encoding=enc)
        raw = uploaded_file.read()
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        uploaded_file.seek(0)
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file, encoding=enc)
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        if name.endswith(".json"):
            uploaded_file.seek(0)
            try:
                return pd.json_normalize(json.load(uploaded_file))
            except Exception:
                uploaded_file.seek(0)
                return pd.read_json(uploaded_file)
        if name.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                tables = []
                for p in pdf.pages:
                    t = p.extract_tables()
                    if t:
                        for tbl in t:
                            dfp = pd.DataFrame(tbl[1:], columns=tbl[0])
                            tables.append(dfp)
                if tables:
                    return pd.concat(tables, ignore_index=True)
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding=enc)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def detect_columns(df):
    """
    Enhanced auto-detection for key columns:
    - Date/time column (date, day, ds, timestamp, txn_dt, etc.)
    - Target/numeric column (sales, revenue, amount, total, value, qty, etc.)
    - Store and Department/Category columns
    Returns: dict with detected names or None if critical ones missing.
    """
    cols = df.columns.tolist()
    lower_map = {c.lower(): c for c in cols}

    def best_match(keywords):
        for k in lower_map:
            if any(word in k for word in keywords):
                return lower_map[k]
        return None

    date_col = best_match(["date", "ds", "day", "time", "timestamp", "txn", "period", "week", "month"])
    target_col = best_match(["sales", "revenue", "amount", "total", "value", "qty", "demand", "target", "y"])
    store_col = best_match(["store", "shop", "location", "branch", "outlet", "zone", "region", "market"])
    dept_col = best_match(["dept", "department", "category", "cat", "segment", "division", "class"])

    # fallback if not found: try numeric column for target
    if not target_col:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            target_col = num_cols[0]

    if date_col and target_col:
        return {"date": date_col, "target": target_col, "store": store_col, "dept": dept_col}
    return None


def prepare_data_for_prophet(df, date_col, target_col):
    dfc = df.copy()
    dfc = dfc[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
    dfc["ds"] = pd.to_datetime(dfc["ds"], errors="coerce")
    dfc["y"] = pd.to_numeric(dfc["y"], errors="coerce")
    dfc = dfc.dropna(subset=["ds"])
    dfc["y"] = dfc["y"].fillna(0)
    return dfc


def resample_and_fill(df_ds_y, freq):
    df = df_ds_y.copy()
    df = df.set_index("ds").sort_index()
    df_res = df.resample(freq).sum(numeric_only=True)
    if df_res["y"].isna().any():
        df_res["y"] = df_res["y"].interpolate(method="linear").fillna(method="ffill").fillna(0)
    return df_res.reset_index()


def compute_metrics_by_merge(actual_df, pred_df):
    a = actual_df.rename(columns=lambda c: "ds" if c.lower() in ("ds", "date") else c)
    if "y" not in a.columns:
        for col in a.columns:
            if col != "ds":
                a = a.rename(columns={col: "y"})
                break
    p = pred_df.rename(columns=lambda c: "ds" if c.lower() in ("ds", "date") else c)
    if "yhat" not in p.columns:
        for col in p.columns:
            if col != "ds":
                p = p.rename(columns={col: "yhat"})
                break
    merged = pd.merge(a[["ds", "y"]], p[["ds", "yhat"]], on="ds", how="inner").dropna(subset=["y", "yhat"])
    if merged.empty:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "merged": merged}

    actual, pred = merged["y"].values, merged["yhat"].values
    mae = np.mean(np.abs(actual - pred))
    mse = np.mean((actual - pred) ** 2)
    rmse = np.sqrt(mse)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan
    denom = (np.abs(actual) + np.abs(pred)) / 2
    smape = np.mean(np.abs(actual - pred) / denom) * 100 if np.any(denom != 0) else np.nan

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "merged": merged}


def create_pdf(store_name, dept_name, forecast_tail, bullets, fig):
    fname = f"forecast_report_{datetime.now().strftime('%Y%m%d')}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(fname, pagesize=letter)
    content = []

    content.append(Paragraph("<b>Sales Forecast Report</b>", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d')}", styles["Normal"]))
    content.append(Paragraph(f"Store: {store_name} | Department: {dept_name}", styles["Normal"]))
    content.append(Spacer(1, 12))

    try:
        ft = forecast_tail.copy()
        ft["Date"] = pd.to_datetime(ft["Date"]).dt.date
        table_data = [ft.columns.tolist()] + ft.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        ]))
        content.append(Paragraph("<b>Forecasted Values</b>", styles["Heading2"]))
        content.append(table)
        content.append(Spacer(1, 14))
    except Exception:
        pass

    try:
        img_buf = BytesIO()
        fig.savefig(img_buf, bbox_inches="tight")
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=480, height=240)
        content.append(Paragraph("<b>Forecast Visualization</b>", styles["Heading2"]))
        content.append(rl_img)
        content.append(Spacer(1, 14))
    except Exception:
        pass

    content.append(Paragraph("<b>Insights & Recommendations</b>", styles["Heading2"]))
    for b in bullets or ["No specific recommendations for this forecast."]:
        content.append(Paragraph(f"• {b}", styles["Normal"]))

    doc.build(content)
    return fname


def run_prophet_forecast(df_resampled, periods, freq, seasonality_weekly=True, seasonality_yearly=True, seasonality_daily=False):
    if not USE_PROPHET:
        raise RuntimeError("Prophet not installed.")
    m = Prophet(
        growth="linear",
        daily_seasonality=seasonality_daily,
        weekly_seasonality=seasonality_weekly,
        yearly_seasonality=seasonality_yearly
    )
    m.fit(df_resampled.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy(), m


# ---------------- UI ----------------
tabs = st.tabs(["📂 Upload Data", "📊 Store & Department Performance Overview", "📈 Forecast", "💬 Actionable Insights", ])

# ---------------------------------------------------
# 📂 Tab 0: Upload Data
# ---------------------------------------------------
with tabs[0]:
    st.markdown("### Step 1: Upload Your Data File")
    st.markdown('<div class="one-liner">Upload CSV file. The app will detect date & sales columns automatically.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload dataset file")

    if uploaded_file:
        df = read_any_file(uploaded_file)
        if df is not None:
            df = df.round(0)
            st.session_state.df_raw = df
            st.subheader("Complete dataset")
            st.dataframe(df, height=600, use_container_width=True)
            st.success("Dataset loaded successfully!")

# ---------------------------------------------------
# 📊 Tab 1: Actual Data Insights
# ---------------------------------------------------
with tabs[1]:
    
    import plotly.express as px
    import pandas as pd
    import streamlit as st
    st.markdown("## 📊 Store & Department Performance Overview")
    st.caption("Visualize key patterns in your uploaded dataset before forecasting.")

    # 1️⃣ Check if data is uploaded
    if "df_raw" not in st.session_state:
        st.warning("⚠️ Please upload data first in the '📂 Upload Data' tab.")
        st.stop()

    df = st.session_state.df_raw.copy()

    # 2️⃣ Try to detect essential columns dynamically
    possible_store = [c for c in df.columns if "store" in c.lower()]
    possible_dept = [c for c in df.columns if "dept" in c.lower() or "department" in c.lower()]
    possible_sales = [c for c in df.columns if "sale" in c.lower() or "amount" in c.lower() or "revenue" in c.lower()]

    # Validate presence
    if not (possible_store and possible_dept and possible_sales):
        st.error("Missing one or more required columns: Store, Department, Sales.")
        st.stop()

    store_col = possible_store[0]
    dept_col = possible_dept[0]
    sales_col = possible_sales[0]

    # Coerce numeric sales
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)

    # ---------------------------------------------------
    # 3️⃣ Store × Department Heatmap
    # ---------------------------------------------------
    st.subheader("Store × Department Heatmap")
    pivot_df = (
        df.groupby([store_col, dept_col], as_index=False)[sales_col]
        .sum()
        .pivot(index=store_col, columns=dept_col, values=sales_col)
        .fillna(0)
    )
    pivot_df = pivot_df.round(0).astype(int)
    pivot_df = pivot_df.applymap(lambda x: int(x))

    if pivot_df.shape[0] > 1 and pivot_df.shape[1] > 1:
        fig_heatmap = px.imshow(
            pivot_df,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            labels=dict(x="Department", y="Store", color="Sales"),
            aspect="actual",
        )
        fig_heatmap.update_traces(
            hovertemplate="Store: %{y}<br>Department: %{x}<br>Sales: %{z:.0f}<extra></extra>"
        )
        fig_heatmap.update_layout(
            title="Store × Department Sales Heatmap",
            xaxis_title="Department",
            yaxis_title="Store",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        top_store = (
            df.groupby([store_col, dept_col])[sales_col]
            .sum()
            .reset_index()
            .sort_values(sales_col, ascending=False)
            .iloc[0]
        )
        st.success(f"🏆 Top Performer: {top_store[store_col]} – {top_store[dept_col]} ({top_store[sales_col]:,.0f})")
    else:
        st.info("Not enough variety in Store/Department to display a meaningful heatmap.")

    st.markdown("---")

    # ---------------------------------------------------
    # 4️⃣ Top 3 Departments by Sales
    # ---------------------------------------------------
    st.subheader("Top 3 Departments by Sales")

    dept_sales = (
        df.groupby(dept_col, as_index=False)[sales_col]
        .sum()
        .sort_values(sales_col, ascending=False)
        .head(3)
    )

    if not dept_sales.empty:
        fig_top3 = px.bar(
            dept_sales,
            x=dept_col,
            y=sales_col,
            text=sales_col,
            title="Top 3 Departments by Total Sales",
            color=sales_col,
            color_continuous_scale="YlOrRd",
        )
        fig_top3.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_top3.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_top3, use_container_width=True)
    else:
        st.info("No sales data found to display top departments.")


with tabs[2]:
    import os
    import pandas as pd
    from datetime import date
    import streamlit as st
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    st.markdown("### Step 2: Forecast Settings & Results")

    if "df_raw" not in st.session_state:
        st.info("Please upload a dataset first.")
    else:
        df_raw = st.session_state.df_raw.copy()
        cols_map = detect_columns(df_raw)
        if not cols_map:
            st.error("Could not auto-detect required columns.")
        else:
            date_col, target_col = cols_map["date"], cols_map["target"]
            store_col, dept_col = cols_map["store"], cols_map["dept"]

            st.markdown("**Data slice**")
            if store_col:
                stores = sorted(df_raw[store_col].astype(str).unique())
                selected_store = st.selectbox("Select Store (or All)", ["All"] + stores)
            else:
                selected_store = "All"
            if dept_col:
                depts = sorted(df_raw[dept_col].astype(str).unique())
                selected_dept = st.selectbox("Select Department (or All)", ["All"] + depts)
            else:
                selected_dept = "All"

            filtered = df_raw.copy()
            if store_col and selected_store != "All":
                filtered = filtered[filtered[store_col].astype(str) == selected_store]
            if dept_col and selected_dept != "All":
                filtered = filtered[filtered[dept_col].astype(str) == selected_dept]

            st.subheader("Filtered Data")
            st.dataframe(filtered, height=350, use_container_width=True)

            prepared = prepare_data_for_prophet(filtered, date_col, target_col)
            if prepared.empty:
                st.error("No usable data found.")
            else:
                c1, c2 = st.columns(2)
                freq = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[
                    c1.selectbox("Resample Frequency", ["Daily", "Weekly", "Monthly"], index=0)
                ]
                periods = c2.slider("Forecast Horizon", 1, 100, 30)
                s1, s2, s3 = st.columns(3)
                season_weekly = s1.checkbox("Weekly seasonality", True)
                season_yearly = s2.checkbox("Yearly seasonality", True)
                season_daily = s3.checkbox("Daily seasonality", False)

                with st.spinner("Running Prophet forecast..."):
                    df_resampled = resample_and_fill(prepared, freq)
                    forecast_df, model_obj = run_prophet_forecast(
                        df_resampled, periods, freq,
                        season_weekly, season_yearly, season_daily
                    )

                # ✅ Convert and remove timestamps safely
                for df_ in [forecast_df, df_resampled]:
                    df_["ds"] = pd.to_datetime(df_["ds"], errors="coerce")
                    df_["ds"] = df_["ds"].apply(lambda x: x.date() if pd.notnull(x) else None)

                # ✅ Round forecast values
                forecast_df["yhat_lower"] = forecast_df["yhat_lower"].round(0)
                forecast_df["yhat_upper"] = forecast_df["yhat_upper"].round(0)

                # ✅ Prepare tail period results
                tail_periods = forecast_df.tail(periods).rename(
                    columns={
                        "ds": "Date",
                        "yhat": "Forecast",
                        "yhat_lower": "Lower",
                        "yhat_upper": "Upper"
                    }
                )[["Date", "Forecast", "Lower", "Upper"]]
                tail_periods["Forecast"] = tail_periods["Forecast"].round(0)

                st.subheader("Forecasted Results")
                st.dataframe(tail_periods, use_container_width=True)

                # ✅ Compute model metrics
                metrics = compute_metrics_by_merge(
                    df_resampled[["ds", "y"]],
                    forecast_df[["ds", "yhat"]]
                )

                st.markdown("### 📊 Model Performance")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{metrics['MAE']:.0f}")
                m2.metric("MSE", f"{metrics['MSE']:.0f}")
                m3.metric("RMSE", f"{metrics['RMSE']:.0f}")
                m4.metric("MAPE", f"{metrics['MAPE']:.2f}%")

                # ✅ Forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_resampled["ds"], y=df_resampled["y"],
                    mode="lines+markers", name="Actual"
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"], y=forecast_df["yhat"],
                    mode="lines", name="Forecast"
                ))
                st.plotly_chart(fig, use_container_width=True)

                # ✅ Compute growth and insights
                Historical_Average = df_resampled["y"].mean()
                Forecast_Average = tail_periods["Forecast"].mean()
                growth_pct = (
                    (Forecast_Average - Historical_Average) / Historical_Average * 100
                    if Historical_Average else 0
                )
                
                
                st.subheader("⚠️ Alerts & Recommendations")
                bullets = []
                if growth_pct > 20:
                    bullets.append("Demand rising >20% — consider increasing stock.")
                elif growth_pct > 5:
                    bullets.append("Moderate growth (5–20%) — increase reorder points slightly.")
                elif growth_pct < -10:
                    bullets.append("Demand drop >10% — reduce orders or promote clearance.")
                else:
                    bullets.append("Stable demand — maintain current inventory levels.")
                bullets.append(
                    f"Historical average: {Historical_Average:.1f}, "
                    f"Forecast average: {Forecast_Average:.1f} ({growth_pct:+.1f}%)."
                )

                for b in bullets:
                    st.info(b)

                # ✅ Save everything in session_state for Actionable Insights tab
                st.session_state["df_forecast"] = tail_periods
                st.session_state["store"] = selected_store
                st.session_state["dept"] = selected_dept
                st.session_state["gen_date"] = str(date.today())
                st.session_state["metrics"] = metrics
                st.session_state["growth_pct"] = growth_pct
                st.session_state["Forecast_Average"] = Forecast_Average
                st.session_state["Historical_Average"] = Historical_Average



with tabs[3]:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    import tempfile
    import os
    import re
    import time
    from fpdf import FPDF
    from datetime import datetime
    import plotly.io as pio

    # ---------------------------
    # Header
    # ---------------------------
    st.markdown("## Actionable Insights Dashboard")

    # ---------------------------
    # Access stored data
    # ---------------------------
    if (
        "df_forecast" not in st.session_state
        or st.session_state["df_forecast"] is None
        or st.session_state["df_forecast"].empty
    ):
        st.warning("Please generate a forecast first in the Forecast tab.")
        st.stop()

    df_forecast = st.session_state["df_forecast"].copy()
    df_raw = st.session_state.get("df_raw", None)

    store = st.session_state.get("store", "All Stores")
    dept = st.session_state.get("dept", "All Departments")
    gen_date = st.session_state.get("gen_date", str(datetime.today().date()))

    # ---------------------------
    # Normalize columns (keeps original logic)
    # ---------------------------
    df = df_forecast.copy()
    df.columns = df.columns.str.strip()
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or cl == "ds":
            rename_map[c] = "Date"
        elif "forecast" in cl or "yhat" in cl:
            rename_map[c] = "Forecast"
        elif "lower" in cl:
            rename_map[c] = "Lower"
        elif "upper" in cl:
            rename_map[c] = "Upper"
    df.rename(columns=rename_map, inplace=True)

    if "Date" not in df.columns or "Forecast" not in df.columns:
        st.error("Missing 'Date' or 'Forecast' columns. Please re-run the Forecast tab.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Forecast"]).sort_values("Date").reset_index(drop=True)

    if "Store" not in df.columns:
        df["Store"] = store
    if "Department" not in df.columns:
        df["Department"] = dept
    if "Target" not in df.columns:
        df["Target"] = df["Forecast"].mean() * 0.95

    # ---------------------------
    # Frequency selection (keeps behavior)
    # ---------------------------
    freq = st.selectbox("Select Frequency for Analysis", ["Daily", "Weekly", "Monthly"], index=0)
    rolling_window = 7 if freq == "Daily" else 4 if freq == "Weekly" else 2
    st.caption(f"Rolling average window automatically set to {rolling_window} periods.")

    if freq == "Daily":
        resampled = df.copy()
        resampled["Date_ref"] = resampled["Date"]
    elif freq == "Weekly":
        df["_week_start"] = df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        resampled = df.groupby("_week_start", as_index=False)["Forecast"].sum()
        resampled["Date_ref"] = resampled["_week_start"]
    else:
        df["_month_start"] = df["Date"].dt.to_period("M").apply(lambda r: r.start_time)
        resampled = df.groupby("_month_start", as_index=False)["Forecast"].sum()
        resampled["Date_ref"] = resampled["_month_start"]

    # ---------------------------
    # KPIs (keeps your formulas)
    # ---------------------------
    Historical_Average = resampled["Forecast"].mean() * 0.9 if not resampled.empty else 0
    Forecast_Average = resampled["Forecast"].mean() if not resampled.empty else 0
    growth_pct = ((Forecast_Average - Historical_Average) / Historical_Average * 100) if Historical_Average else 0
    trend_dir = "increase" if growth_pct > 0 else "decline"

    high_perf_threshold = Historical_Average * 1.2
    high_perf = resampled[resampled["Forecast"] > high_perf_threshold]
    high_perf_count = len(high_perf)
    top_day = resampled.loc[resampled["Forecast"].idxmax()] if not resampled.empty else None
    top_value = f"{top_day['Forecast']:.0f}" if top_day is not None else "0"
    top_date = top_day["Date_ref"].strftime("%Y-%m-%d") if top_day is not None else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Historical Average", f"{Historical_Average:.0f}")
    c2.metric("Forecast Average", f"{Forecast_Average:.0f}")
    c3.metric("High-Performing Periods", f"{high_perf_count}")

    # present high performing periods (rounded / sanitized)
    hp_list = high_perf["Date_ref"].dt.strftime("%Y-%m-%d").tolist() if not high_perf.empty else []
    st.markdown(
        f'<div style="background-color:#0078D7;color:white;padding:10px;border-radius:8px;margin-top:5px;">'
        f'<b>High-performing periods:</b> {", ".join(hp_list) if hp_list else "None detected"}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ---------------------------
    # Executive Summary
    # ---------------------------
    st.markdown("### Executive Summary")
    st.markdown(f"""
- **Store:** {store}  
- **Department:** {dept}  
- **Report Date:** {gen_date}  
- **Frequency:** {freq}  
- **Trend:** Expected **{trend_dir} of {abs(growth_pct):.1f}%**  
- **Historical Average:** {Historical_Average:.0f}  
- **Forecast Average:** {Forecast_Average:.0f}
    """)

    # ---------------------------
    # Forecast Trend + Peak marker (keeps look & behavior)
    # ---------------------------
    resampled["RollingAverage"] = resampled["Forecast"].rolling(rolling_window, min_periods=1).mean()
    peak_date = resampled.loc[resampled["Forecast"].idxmax(), "Date_ref"] if not resampled.empty else None
    peak_value = resampled["Forecast"].max() if not resampled.empty else 0

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=resampled["Date_ref"], y=resampled["Forecast"],
                                   mode="lines+markers", name="Forecast"))
    fig_trend.add_trace(go.Scatter(x=resampled["Date_ref"], y=resampled["RollingAverage"],
                                   mode="lines", name="Rolling Avg", line=dict(dash="dash")))
    if peak_date is not None:
        fig_trend.add_trace(go.Scatter(x=[peak_date], y=[peak_value],
                                       mode="markers+text", name="Peak",
                                       text=[f"Peak: {peak_value:.0f}"],
                                       textposition="top center",
                                       marker=dict(size=12, symbol="star")))
        fig_trend.update_traces(hovertemplate="Date: %{x}<br>Sales: %{y:.0f}<extra></extra>")

    st.markdown("### Forecasted Sales Trend & Peak Period")
    fig_trend.update_layout(title="Forecasted Sales Trend & Peak Period", template="plotly_white", height=450)
    st.plotly_chart(fig_trend, use_container_width=True, key="forecast_trend_chart")



    # ---------------------------
    # PDF export: helpers (now comprehensive)
    # ---------------------------
    def sanitize_text(text: str) -> str:
        # remove emojis / fancy unicode that FPDF with latin-1 may choke on
        if text is None:
            return ""
        return re.sub(r"[^\x00-\x7F]+", "", str(text))

    def save_plotly_png(fig, path, scale=2):
        try:
            # try to use the kaleido engine
            pio.write_image(fig, path, format="png", engine="kaleido", scale=scale)
            # small wait loop until file exists
            for _ in range(10):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return True
                time.sleep(0.1)
        except Exception as e:
            st.warning(f"Could not save plotly image {path}: {e}")
        return False

    def ensure_figures(tmpdir, df_forecast_local, df_raw_local, figs_from_state=True):
        """
        Return dict of saved image paths for the PDF:
         - Forecast_Trend
         - Peak_Date
         - Waterfall
         - Actual_Heatmap
        Preference order: use figures stored in session_state, else regenerate from data.
        """
        imgs = {}

        # 1) Forecast Trend (prefer session_state fig_trend)
        fig = st.session_state.get("fig_trend") if figs_from_state else None
        if fig is None and df_forecast_local is not None and not df_forecast_local.empty:
            try:
                dcol = df_forecast_local.columns[0]; fcol = df_forecast_local.columns[1]
                tmp = df_forecast_local.copy()
                tmp[dcol] = pd.to_datetime(tmp[dcol])
                fig = px.line(tmp, x=dcol, y=fcol, title="Forecast Trend")
                # add rolling avg if possible
                tmp["rolling"] = tmp[fcol].rolling(window=max(1, min(7, len(tmp)//4)), min_periods=1).mean()
                fig.add_scatter(x=tmp[dcol], y=tmp["rolling"], mode="lines", name="Rolling Avg")
            except Exception:
                fig = None
        if fig is not None:
            p = os.path.join(tmpdir, "forecast_trend.png")
            if save_plotly_png(fig, p):
                imgs["Forecast_Trend"] = p

        # 2) Peak Date (prefer session_state fig_peak)
        figp = st.session_state.get("fig_peak") if figs_from_state else None
        if figp is None and df_forecast_local is not None and not df_forecast_local.empty:
            try:
                dcol = df_forecast_local.columns[0]; fcol = df_forecast_local.columns[1]
                tmp = df_forecast_local.copy()
                tmp[dcol] = pd.to_datetime(tmp[dcol])
                peak_row = tmp.loc[tmp[fcol].idxmax()]
                figp = px.line(tmp, x=dcol, y=fcol, title="Peak Forecast Date")
                figp.add_scatter(x=[peak_row[dcol]], y=[peak_row[fcol]], mode="markers+text",
                                 text=[f"Peak: {peak_row[fcol]:.0f}"], textposition="top center")
            except Exception:
                figp = None
        if figp is not None:
            p = os.path.join(tmpdir, "peak_date.png")
            if save_plotly_png(figp, p):
                imgs["Peak_Date"] = p

        # 3) Waterfall (prefer session_state fig_comparison)
        figw = st.session_state.get("fig_comparison") if figs_from_state else None
        if figw is None and df_forecast_local is not None and not df_forecast_local.empty:
            try:
                f_total = df_forecast_local.iloc[:, 1].sum()
                tgt = (st.session_state.get("target_total") or (f_total * 0.95))
                wf = px.bar(x=["Target", "Forecast"], y=[tgt, f_total], title="Forecast vs Target")
                figw = wf
            except Exception:
                figw = None
        if figw is not None:
            p = os.path.join(tmpdir, "Waterfall.png")
            if save_plotly_png(figw, p):
                imgs["Waterfall"] = p

        # 4) Actuals heatmap (prefer session_state fig_heatmap)
        figh = st.session_state.get("fig_heatmap") if figs_from_state else None
        if figh is None and df_raw_local is not None and not df_raw_local.empty:
            try:
                raw = df_raw_local.copy()
                date_col = next((c for c in raw.columns if "date" in c.lower()), None)
                store_col = next((c for c in raw.columns if "store" in c.lower() or "location" in c.lower()), None)
                dept_col = next((c for c in raw.columns if "dept" in c.lower() or "category" in c.lower()), None)
                sales_col = next((c for c in raw.columns if "sale" in c.lower() or "amount" in c.lower() or "revenue" in c.lower()), None)
                if store_col and dept_col and sales_col:
                    pivot = raw.groupby([dept_col, store_col])[sales_col].sum().unstack(fill_value=0)
                    # ensure square-ish cells by keeping aspect="equal" if possible
                    figh = px.imshow(pivot, text_auto=True, aspect="auto", origin="lower", title="Actuals: Department × Store Sales")
                else:
                    if date_col and sales_col:
                        s = raw.groupby(date_col)[sales_col].sum().reset_index()
                        figh = px.line(s, x=date_col, y=sales_col, title="Actuals Over Time")
                    else:
                        figh = None
            except Exception:
                figh = None
        if figh is not None:
            p = os.path.join(tmpdir, "heatmap.png")
            if save_plotly_png(figh, p):
                imgs["Actual_Heatmap"] = p

    # ---------------------------
    # build PDF (comprehensive, includes requested changes)
    # ---------------------------
    def build_pdf(output_path):
        df_forecast_local = st.session_state.get("df_forecast")
        df_raw_local = st.session_state.get("df_raw")
        store_local = st.session_state.get("store", "All Stores")
        dept_local = st.session_state.get("dept", "All Departments")
        gen_date_local = st.session_state.get("gen_date", str(pd.Timestamp.now().date()))
        # compute high-performing periods (same logic)
        high_perf_list_local = []
        if df_forecast_local is not None and not df_forecast_local.empty:
            try:
                dcol = df_forecast_local.columns[0]; fcol = df_forecast_local.columns[1]
                tmp = df_forecast_local.copy()
                tmp[dcol] = pd.to_datetime(tmp[dcol])
                hist_avg = tmp[fcol].mean() * 0.9
                high_perf_list_local = tmp[tmp[fcol] > hist_avg * 1.2][dcol].dt.strftime("%Y-%m-%d").tolist()
            except Exception:
                high_perf_list_local = []

        tmpdir = tempfile.mkdtemp()
        imgs = ensure_figures(tmpdir, df_forecast_local, df_raw_local, figs_from_state=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)

        # PAGE 1: Cover
        pdf.add_page()
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 12, sanitize_text("Business Insight Hub — Actionable Insights Report"), ln=True, align="C")
        pdf.ln(6)

        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)
        metadata = (
            f"Generated: {gen_date_local}\n"
            f"Store: {store_local}\n"
            f"Department: {dept_local}\n"
        )
        pdf.multi_cell(0, 6, sanitize_text(metadata))
        pdf.ln(4)

        # Executive summary block
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 8, sanitize_text("Executive Summary"), ln=True)
        pdf.ln(1)
        pdf.set_font("Arial", "", 11)
        exec_text = (
            "This report shows actual performance, forecast results, and recommended focus areas.\n"
            "Sections include: Actuals heatmap, Forecast results table, Visual insights, and analysis notes."
        )
        pdf.multi_cell(0, 6, sanitize_text(exec_text))
        pdf.ln(4)

        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, sanitize_text("High-Performing Periods:"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, sanitize_text(", ".join(high_perf_list_local) if high_perf_list_local else "None detected"))
        pdf.ln(6)

        # PAGE 2: Actuals heatmap
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, sanitize_text("1) Actual Insights (Heatmap)"), ln=True)
        pdf.ln(2)
        if "Actual_Heatmap" in imgs:
            pdf.image(imgs["Actual_Heatmap"], x=15, w=180)
            pdf.ln(4)
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 6, sanitize_text("Actuals heatmap — darker cell = higher sales."))
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 8, sanitize_text("Actuals heatmap not available."), ln=True)
        pdf.ln(6)

        # PAGE 3: Forecast table (sample)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, sanitize_text("2) Forecasted Results (Sample)"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.ln(2)
        if df_forecast_local is not None and not df_forecast_local.empty:
            df_tbl = df_forecast_local.copy().head(20)
            cols = [str(c) for c in df_tbl.columns]
            col_count = len(cols)
            # width space and column width
            table_w = pdf.w - 20
            col_w = max(20, table_w / col_count)
            pdf.set_font("Arial", "B", 9)
            pdf.set_fill_color(0, 51, 102)
            pdf.set_text_color(255, 255, 255)
            for c in cols:
                pdf.cell(col_w, 7, sanitize_text(c[:15]), border=1, align="C", fill=True)
            pdf.ln()
            pdf.set_font("Arial", "", 8)
            pdf.set_text_color(0, 0, 0)
            for _, row in df_tbl.iterrows():
                for val in row:
                    # round numeric values, sanitize text
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        cell_text = f"{int(round(val))}"
                    else:
                        cell_text = sanitize_text(str(val)[:15])
                    pdf.cell(col_w, 6, cell_text, border=1, align="C")
                pdf.ln()
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.cell(0, 8, sanitize_text("Forecast table not available."), ln=True)
        pdf.ln(6)

        # PAGE 4: Visual insights (Trend, Peak, Waterfall)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, sanitize_text("3) Visual Insights"), ln=True)
        pdf.ln(2)
        chart_order = ["Forecast_Trend", "Peak_Date", "Waterfall"]
        captions = {
            "Forecast_Trend": "Forecast trend with rolling average and peak marker.",
            "Peak_Date": "Peak demand date highlighted for prioritized action.",
            "Comparision": "Variance between forecast and target highlighting gap."
        }
        for name in chart_order:
            if name in imgs:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, sanitize_text(name.replace("_", " ")), ln=True)
                pdf.image(imgs[name], x=15, w=180)
                pdf.ln(4)
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, sanitize_text(captions.get(name, "")))
                pdf.ln(4)
            else:
                pdf.set_font("Arial", "I", 10)
                pdf.cell(0, 7, sanitize_text(f"Chart not available: {name}"), ln=True)
                pdf.ln(4)

        # PAGE last: KPIs & Notes
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, sanitize_text("4) Actionable Insights & Notes"), ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.ln(2)
        # include some computed KPIs
        kpi_text = (
            f"Historical Average: {int(round(Historical_Average))} | "
            f"Forecast Average: {int(round(Forecast_Average))} | "
            f"High-performing periods: {', '.join(high_perf_list_local) if high_perf_list_local else 'None'}"
        )
        pdf.multi_cell(0, 6, sanitize_text(kpi_text))
        pdf.ln(6)
        summary_text = st.session_state.get("actionable_summary") or ""
        if summary_text:
            pdf.multi_cell(0, 6, sanitize_text(summary_text))
        pdf.ln(6)
        pdf.set_font("Arial", "I", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 6, sanitize_text("Generated by Business Insight Hub."))

        # Save
        pdf.output(output_path)

    # ---------------------------
    # Streamlit UI integration (inside tab[3])
    # ---------------------------
    if st.button("Generate PDF Report"):
        tmpf = tempfile.mkdtemp()
        outpath = os.path.join(tmpf, "Actionable_Insights_Report.pdf")
        # Build images and PDF
        imgs = ensure_figures(tmpf, st.session_state.get("df_forecast"), st.session_state.get("df_raw"))
        build_pdf(outpath)
        with open(outpath, "rb") as f:
            st.download_button(
                "Download Report (PDF)",
                f.read(),
                file_name=f"actionable_insight_{dept}_{store}_{gen_date}.pdf",
                mime="application/pdf"
            )
