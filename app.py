import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from datetime import timedelta

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Sales & Product Analysis", layout="wide")

# -------------------------------------------------
# Configuration & Constants
# -------------------------------------------------
DROP_FEATURES = [
    "CompanyName",
    "CustGroupName",
    "CompanyChainName",
    "PRODUCTNAME"
]
DATE_FEATURES = [
    "InvoiceDate"
]
CATEGORICAL_FEATURES = [
    "CustGroup",
    "State"
]
CAT_NUMERIC_FEATURES = [
    "DATAAREAID",
    "CompanyChain",
    "ItemNumber"
]
NUMERICAL_FEATURES = [
    "INVOICEDQUANTITY",
    "QTYInKG/Ltr"
]
PREPROCESSING_ONLY = [
    "SALESORDERORIGINCODE"
]

REQUIRED_COLS = (
    DROP_FEATURES + DATE_FEATURES + CATEGORICAL_FEATURES + 
    CAT_NUMERIC_FEATURES + NUMERICAL_FEATURES + PREPROCESSING_ONLY
)

CACHE_DIR = "./data/cache"
PROCESSED_CACHE_FILE = os.path.join(CACHE_DIR, "sales_data_processed_v2.parquet")
MISSING_SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "missing_summary_v2.parquet")

# -------------------------------------------------
# Data Loading Functions
# -------------------------------------------------
@st.cache_data
def load_metadata():
    """Loads metadata from the Excel file."""
    file_path = "data/SmartData_Table_Information.xlsx"
    if not os.path.exists(file_path):
        return None, None, None

    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        
        sales_fields_df = all_sheets.get("SalesData_Fields")
        product_fields_df = all_sheets.get("Product_Fields")
        product_df = all_sheets.get("Sample_Product & Category")
        
        def clean_cols(df):
            if df is None: return None
            df.columns = (
                df.columns.astype(str).str.strip()
                .str.replace(r"[\s\xa0]+", " ", regex=True)
            )
            return df

        sales_fields_df = clean_cols(sales_fields_df)
        product_fields_df = clean_cols(product_fields_df)
        product_df = clean_cols(product_df)

        if product_df is not None:
             product_df = product_df.rename(columns={"ITEMNUMBER": "ItemNumber"})

        return product_df, sales_fields_df, product_fields_df
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None, None, None

def _load_raw_csvs():
    """Internal function to load raw CSVs."""
    input_paths = [
        "./data/Sales_Data_OctDec2025",
        "./data/Sales_Data_OctDec2024"
    ]
    csv_files = []
    for path in input_paths:
        csv_files.extend(glob.glob(os.path.join(path, "*.csv")))
    
    if not csv_files:
        return pd.DataFrame()

    return pd.concat(
        [pd.read_csv(f, low_memory=False, usecols=lambda c: c in REQUIRED_COLS) for f in csv_files],
        ignore_index=True
    )

def _calculate_missing_summary(df):
    """Calculates missing values summary on the raw dataframe."""
    if df.empty: return pd.DataFrame()
    missing = df.isnull().sum()
    
    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Percentage": (missing.values / len(df)) * 100
    })
    return summary.sort_values("Missing Count", ascending=False)

def _process_data(df):
    """Internal function to clean and add features."""
    if df.empty: return df
    df = df.copy()
    
    # Clean
    cols_to_drop = [c for c in DROP_FEATURES if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "ItemNumber"])
    
    df["INVOICEDQUANTITY"] = df["INVOICEDQUANTITY"].fillna(0)
    df["QTYInKG/Ltr"] = df.get("QTYInKG/Ltr", pd.Series(0)).fillna(0)
    
    if "SALESORDERORIGINCODE" in df.columns:
        df["SALESORDERORIGINCODE"] = df["SALESORDERORIGINCODE"].fillna("unknown")
        
    if "State" in df.columns:
         df["State"] = df["State"].astype(str).str.strip().str.lower().str.title()

    # Add Date Features
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["WeekOfYear"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    
    # Create a Month-Day feature for year-over-year comparison (e.g., "10-01", "12-31")
    df["MonthDay"] = df["InvoiceDate"].dt.strftime("%m-%d")
    
    # Filter for Oct-Dec only (as requested)
    df = df[df["Month"].isin([10, 11, 12])]

    return df

@st.cache_data(show_spinner=False)
def load_and_process_data():
    """
    Loads data from cache if available, otherwise loads from CSV, processes, and caches.
    Returns: (processed_df, missing_summary_df)
    """
    # Check cache
    if os.path.exists(PROCESSED_CACHE_FILE) and os.path.exists(MISSING_SUMMARY_CACHE_FILE):
        try:
            processed_df = pd.read_parquet(PROCESSED_CACHE_FILE)
            missing_summary_df = pd.read_parquet(MISSING_SUMMARY_CACHE_FILE)
            
            # Validate cache content (ensure new features exist)
            if "MonthDay" not in processed_df.columns:
                raise ValueError("Cache outdated: Missing 'MonthDay' column")
                
            return processed_df, missing_summary_df
        except Exception as e:
            # Cache is invalid or outdated, proceed to reload
            pass
    
    # Load from source
    with st.spinner("Loading raw data from CSVs..."):
        raw_df = _load_raw_csvs()
    
    if raw_df.empty:
        return raw_df, pd.DataFrame()
        
    # Calculate Missing Summary on RAW data (before dropping cols)
    with st.spinner("Analyzing data quality..."):
        missing_summary_df = _calculate_missing_summary(raw_df)

    # Process
    with st.spinner("Processing and feature engineering..."):
        processed_df = _process_data(raw_df)
    
    # Save to cache
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        processed_df.to_parquet(PROCESSED_CACHE_FILE, index=False)
        missing_summary_df.to_parquet(MISSING_SUMMARY_CACHE_FILE, index=False)
    except Exception as e:
        st.warning(f"Failed to cache data: {e}")
        
    return processed_df, missing_summary_df

# -------------------------------------------------
# Aggregation & Metrics (Cached)
# -------------------------------------------------
@st.cache_data
def aggregate_sales(df):
    """Aggregates sales data."""
    if df.empty: return pd.DataFrame()
    
    group_cols = ["Year", "Month", "ItemNumber", "State", "CustGroup"]
    group_cols = [c for c in group_cols if c in df.columns]
    
    agg_df = (
        df.groupby(group_cols)
        .agg(
            total_qty=("INVOICEDQUANTITY", "sum"),
            avg_qty=("INVOICEDQUANTITY", "mean"),
            order_count=("InvoiceDate", "nunique")
        )
        .reset_index()
    )
    return agg_df

@st.cache_data
def compute_pattern_metrics(df):
    """Computes demand pattern metrics."""
    if df.empty: return pd.DataFrame()
    
    metrics = (
        df.groupby("ItemNumber")["total_qty"]
        .agg(
            mean_qty="mean",
            std_qty="std",
            zero_ratio=lambda x: (x == 0).mean()
        )
        .reset_index()
    )
    
    metrics["cv"] = metrics["std_qty"] / metrics["mean_qty"]
    metrics["cv"] = metrics["cv"].replace([np.inf, -np.inf], np.nan)
    
    return metrics

@st.cache_data
def classify_patterns(metrics_df):
    """Classifies items into demand patterns."""
    if metrics_df.empty: return metrics_df
    
    def classify(row):
        if row["zero_ratio"] > 0.3:
            return "Intermittent"
        elif row["cv"] < 0.5:
            return "Stable"
        elif row["cv"] > 1.0:
            return "Volatile"
        else:
            return "Seasonal"

    metrics_df["demand_pattern"] = metrics_df.apply(classify, axis=1)
    return metrics_df

# -------------------------------------------------
# Item Analysis Helper Functions
# -------------------------------------------------

def get_item_time_series(df, item_number, freq_col):
    """Generates time series data for a specific item."""
    item_df = df[df["ItemNumber"] == item_number].copy()
    
    ts_df = (
        item_df.groupby(["Year", freq_col])
        .agg(total_qty=("INVOICEDQUANTITY", "sum"))
        .reset_index()
    )
    
    ts_df = ts_df.sort_values(["Year", freq_col])
    
    ts_df["yoy_pct"] = (
        ts_df.groupby(freq_col)["total_qty"]
        .pct_change() * 100
    )
    
    return ts_df

def get_rolling_average(df, item_number, window=7):
    """Calculates rolling average for an item."""
    item_df = df[df["ItemNumber"] == item_number].copy()
    daily_df = (
        item_df.groupby("InvoiceDate")
        .agg(total_qty=("INVOICEDQUANTITY", "sum"))
        .reset_index()
        .sort_values("InvoiceDate")
    )
    
    daily_df[f"Rolling_{window}d"] = daily_df["total_qty"].rolling(window=window, min_periods=1).mean()
    return daily_df

def get_daily_seasonality_comparison(df, item_number):
    """
    Aggregates daily sales by Month-Day for 2024 and 2025 comparison.
    """
    item_df = df[df["ItemNumber"] == item_number].copy()
    
    # Group by Year and MonthDay
    daily_season = (
        item_df.groupby(["Year", "MonthDay"])
        .agg(total_qty=("INVOICEDQUANTITY", "sum"))
        .reset_index()
    )
    
    # Explicitly sort by MonthDay to ensure correct x-axis order
    daily_season = daily_season.sort_values("MonthDay")
    
    return daily_season

def generate_forecast(df, item_number, horizon=30):
    """
    Generates a simple forecast based on Day-of-Week seasonality.
    """
    item_df = df[df["ItemNumber"] == item_number].copy()
    if item_df.empty:
        return pd.DataFrame()

    # 1. Calculate Day-of-Week Seasonality
    dow_avg = item_df.groupby("DayOfWeek")["INVOICEDQUANTITY"].mean()
    overall_avg = item_df["INVOICEDQUANTITY"].mean()
    
    if overall_avg == 0:
        seasonality = pd.Series(1.0, index=range(7))
    else:
        seasonality = dow_avg / overall_avg
        
    # 2. Base Level (Last 30 days average)
    last_date = item_df["InvoiceDate"].max()
    start_base = last_date - timedelta(days=30)
    base_level = item_df[item_df["InvoiceDate"] > start_base]["INVOICEDQUANTITY"].mean()
    
    if pd.isna(base_level):
        base_level = overall_avg

    # 3. Generate Future Dates
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
    
    # 4. Calculate Forecast
    forecast_data = []
    for date in future_dates:
        dow = date.dayofweek
        factor = seasonality.get(dow, 1.0)
        forecast_val = base_level * factor
        forecast_data.append({"InvoiceDate": date, "Forecast": forecast_val})
        
    return pd.DataFrame(forecast_data)

# -------------------------------------------------
# Main Application
# -------------------------------------------------
st.title("📊 Sales & Product Analysis Dashboard")

# Load Data (Cached to Disk)
clean_df, missing_summary_df = load_and_process_data()
product_df, sales_fields_df, product_fields_df = load_metadata()

if clean_df.empty:
    st.warning("No sales data available. Please check the data directory.")
    st.stop()

# Aggregations (Cached in Memory)
with st.spinner("Aggregating Data..."):
    agg_df = aggregate_sales(clean_df)
    metrics_df = compute_pattern_metrics(agg_df)
    pattern_df = classify_patterns(metrics_df)

# -------------------------------------------------
# KPIs
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(clean_df):,}")
col2.metric("Total Quantity Sold", f"{clean_df['INVOICEDQUANTITY'].sum():,.0f}")
col3.metric("Unique Products", clean_df["ItemNumber"].nunique())
col4.metric("Unique Customers", clean_df["CustGroup"].nunique())

st.divider()

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tabs = st.tabs([
    "📈 Sales Analysis",
    "🧠 Demand Patterns",
    "🔍 Item Analysis",
    "⚠️ Data Quality"
])

# -------------------------------------------------
# Tab 1: Sales Analysis
# -------------------------------------------------
with tabs[0]:
    st.subheader("Sales Trends")
    daily_sales = clean_df.groupby("InvoiceDate")["INVOICEDQUANTITY"].sum().reset_index()
    fig_sales = px.line(daily_sales, x="InvoiceDate", y="INVOICEDQUANTITY", title="Daily Sales Quantity")
    st.plotly_chart(fig_sales, use_container_width=True)
    
    st.subheader("Sales by State")
    state_sales = clean_df.groupby("State")["INVOICEDQUANTITY"].sum().reset_index().sort_values("INVOICEDQUANTITY", ascending=False)
    fig_state = px.bar(state_sales, x="State", y="INVOICEDQUANTITY", title="Total Sales by State")
    st.plotly_chart(fig_state, use_container_width=True)

# -------------------------------------------------
# Tab 2: Demand Patterns
# -------------------------------------------------
with tabs[1]:
    st.subheader("Demand Pattern Classification")
    
    if not pattern_df.empty:
        pattern_counts = pattern_df["demand_pattern"].value_counts().reset_index()
        pattern_counts.columns = ["Pattern", "Count"]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(pattern_counts, hide_index=True, width="stretch")
        with c2:
            fig_pie = px.pie(pattern_counts, values="Count", names="Pattern", title="Distribution of Demand Patterns")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.subheader("Item-Level Metrics")
        st.dataframe(pattern_df, width="stretch")
    else:
        st.info("Not enough data to compute patterns.")

# -------------------------------------------------
# Tab 3: Item Analysis
# -------------------------------------------------
with tabs[2]:
    st.subheader("Deep Dive: Item Analysis")
    st.markdown("Analyze individual product performance across the **Oct-Dec** periods of 2024 and 2025.")
    
    top_items = (
        agg_df.groupby("ItemNumber")["total_qty"].sum()
        .sort_values(ascending=False).head(100).index.tolist()
    )
    all_items = sorted(agg_df["ItemNumber"].unique())
    
    selected_item = st.selectbox(
        "Select Item Number", 
        options=all_items,
        index=all_items.index(top_items[0]) if top_items else 0
    )
    
    if selected_item:
        # Metadata & Pattern
        item_name = "Unknown"
        if product_df is not None and "ItemNumber" in product_df.columns and "PRODUCTNAME" in product_df.columns:
            item_meta = product_df[product_df["ItemNumber"] == selected_item]
            if not item_meta.empty:
                item_name = item_meta.iloc[0]["PRODUCTNAME"]
        
        pattern_info = pattern_df[pattern_df["ItemNumber"] == selected_item]
        pattern_type = pattern_info.iloc[0]["demand_pattern"] if not pattern_info.empty else "N/A"
        
        st.info(f"**Item:** {selected_item} | **Name:** {item_name} | **Pattern:** {pattern_type}")
        
        # --- Prepare Data for Plots ---
        ts_month = get_item_time_series(clean_df, selected_item, "Month")
        ts_week = get_item_time_series(clean_df, selected_item, "WeekOfYear")
        ts_day = get_item_time_series(clean_df, selected_item, "DayOfWeek")
        ts_rolling = get_rolling_average(clean_df, selected_item, window=30) 
        daily_seasonality = get_daily_seasonality_comparison(clean_df, selected_item)
        
        # --- Plotting ---
        
        # 0. Daily Seasonality Comparison (New)
        st.markdown("### 🗓️ Daily Sales Pattern: 2024 vs 2025 (Oct-Dec)")
        st.markdown("Compare the daily sales rhythm across the same months in different years.")
        
        # Ensure Year is treated as discrete category
        daily_seasonality["Year"] = daily_seasonality["Year"].astype(str)
        
        # Get unique sorted MonthDays for the x-axis order
        sorted_month_days = sorted(daily_seasonality["MonthDay"].unique())
        
        fig_season = px.line(daily_seasonality, x="MonthDay", y="total_qty", color="Year", 
                             title="Daily Sales Comparison (Aligned by Day)",
                             markers=True,
                             category_orders={"MonthDay": sorted_month_days}) # FORCE correct order
        fig_season.update_xaxes(type='category') 
        st.plotly_chart(fig_season, use_container_width=True)

        # 1. Month-wise
        st.markdown("### 📅 Month-wise Performance")
        c1, c2 = st.columns(2)
        with c1:
            fig_m_raw = px.line(ts_month, x="Month", y="total_qty", color="Year", markers=True, title="Monthly Sales (Raw)")
            fig_m_raw.update_xaxes(dtick=1)
            st.plotly_chart(fig_m_raw, use_container_width=True)
        with c2:
            fig_m_yoy = px.bar(ts_month, x="Month", y="yoy_pct", color="Year", title="Monthly YoY Growth (%)")
            fig_m_yoy.update_xaxes(dtick=1)
            st.plotly_chart(fig_m_yoy, use_container_width=True)

        # 2. Week-wise
        st.markdown("### 📆 Week-wise Performance")
        c3, c4 = st.columns(2)
        with c3:
            fig_w_raw = px.line(ts_week, x="WeekOfYear", y="total_qty", color="Year", markers=True, title="Weekly Sales (Raw)")
            st.plotly_chart(fig_w_raw, use_container_width=True)
        with c4:
            fig_w_yoy = px.bar(ts_week, x="WeekOfYear", y="yoy_pct", color="Year", title="Weekly YoY Growth (%)")
            st.plotly_chart(fig_w_yoy, use_container_width=True)

        # 3. Day-of-Week
        st.markdown("### 🗓️ Day-of-Week Performance")
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        ts_day["DayName"] = ts_day["DayOfWeek"].map(day_map)
        
        c5, c6 = st.columns(2)
        with c5:
            fig_d_raw = px.line(ts_day, x="DayName", y="total_qty", color="Year", markers=True, title="Day-of-Week Sales (Raw)")
            st.plotly_chart(fig_d_raw, use_container_width=True)
        with c6:
            fig_d_yoy = px.bar(ts_day, x="DayName", y="yoy_pct", color="Year", title="Day-of-Week YoY Growth (%)")
            st.plotly_chart(fig_d_yoy, use_container_width=True)

        # 4. Rolling Average
        st.markdown("### 📈 Rolling Average Trend")
        fig_roll = px.line(ts_rolling, x="InvoiceDate", y=["total_qty", "Rolling_30d"], 
                           title="Daily Sales vs 30-Day Rolling Average",
                           labels={"value": "Quantity", "variable": "Metric"})
        st.plotly_chart(fig_roll, use_container_width=True)

        # 5. Forecasting
        st.markdown("### 🔮 30-Day Forecast")
        st.markdown("This forecast projects future sales based on the item's **Day-of-Week seasonality** and recent average sales volume.")
        
        forecast_df = generate_forecast(clean_df, selected_item, horizon=30)
        
        if not forecast_df.empty:
            # Combine history (last 60 days) and forecast for plotting
            last_60_days = ts_rolling.tail(60).copy()
            last_60_days["Type"] = "Historical"
            last_60_days = last_60_days.rename(columns={"total_qty": "Quantity"})
            
            forecast_plot_df = forecast_df.copy()
            forecast_plot_df["Type"] = "Forecast"
            forecast_plot_df = forecast_plot_df.rename(columns={"Forecast": "Quantity"})
            
            combined_plot = pd.concat([last_60_days[["InvoiceDate", "Quantity", "Type"]], 
                                     forecast_plot_df[["InvoiceDate", "Quantity", "Type"]]])
            
            # Explicitly sort by InvoiceDate
            combined_plot = combined_plot.sort_values("InvoiceDate")
            
            fig_forecast = px.line(combined_plot, x="InvoiceDate", y="Quantity", color="Type",
                                   title="Sales Forecast (Next 30 Days)",
                                   color_discrete_map={"Historical": "blue", "Forecast": "orange"})
            fig_forecast.update_traces(mode="lines+markers")
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("Not enough data to generate a forecast.")

# -------------------------------------------------
# Tab 4: Data Quality
# -------------------------------------------------
with tabs[3]:
    st.subheader("⚠️ Data Quality Summary (Raw Data)")
    st.markdown("This summary is calculated on the **raw data** before any cleaning or row dropping.")
    
    if not missing_summary_df.empty:
        try:
            st.dataframe(
                missing_summary_df.style.format({"Percentage": "{:.2f}%"})
                .background_gradient(cmap="Reds", subset=["Missing Count"]),
                width="stretch"
            )
        except ImportError:
            st.warning("Install `matplotlib` to see color gradients. Displaying plain table.")
            st.dataframe(
                missing_summary_df.style.format({"Percentage": "{:.2f}%"}),
                width="stretch"
            )
        except Exception as e:
            st.error(f"Error displaying table: {e}")
            st.dataframe(missing_summary_df, width="stretch")
        
        # Specific check for ItemNumber
        item_missing = missing_summary_df[missing_summary_df["Column"] == "ItemNumber"]
        if not item_missing.empty:
            count = item_missing.iloc[0]["Missing Count"]
            st.error(f"🚨 **ItemNumber** has {count} missing values! These rows are dropped during processing.")
    else:
        st.success("No missing values detected in the columns of interest!")
