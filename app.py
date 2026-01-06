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
    # "CustGroupName", # Keep for display
    # "CompanyChainName", # Keep for display
    # "PRODUCTNAME" # Keep for display
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
PROCESSED_CACHE_FILE = os.path.join(CACHE_DIR, "sales_data_processed_v6.parquet")
MISSING_SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "missing_summary_v6.parquet")

# ... (inside _process_data or load_and_process_data) ...

@st.cache_data(show_spinner=False)
def load_and_process_data():
    """
    Loads data from cache if available, otherwise loads from CSV, processes, merges category/name, and caches.
    Returns: (processed_df, missing_summary_df)
    """
    # Check cache
    if os.path.exists(PROCESSED_CACHE_FILE) and os.path.exists(MISSING_SUMMARY_CACHE_FILE):
        try:
            processed_df = pd.read_parquet(PROCESSED_CACHE_FILE)
            missing_summary_df = pd.read_parquet(MISSING_SUMMARY_CACHE_FILE)
            
            # Validate cache content
            if "MonthDay" not in processed_df.columns or "Category" not in processed_df.columns or "ProductName" not in processed_df.columns or "ItemDisplay" not in processed_df.columns:
                raise ValueError("Cache outdated")
                
            return processed_df, missing_summary_df
        except Exception as e:
            pass
    
    # Load from source
    with st.spinner("Loading raw data from CSVs..."):
        raw_df = _load_raw_csvs()
    
    if raw_df.empty:
        return raw_df, pd.DataFrame()
        
    # Calculate Missing Summary
    with st.spinner("Analyzing data quality..."):
        missing_summary_df = _calculate_missing_summary(raw_df)

    # Process
    with st.spinner("Processing and feature engineering..."):
        processed_df = _process_data(raw_df)
        
        # Merge Product Category & Name
        try:
            meta_path = "data/SmartData_Table_Information.xlsx"
            if os.path.exists(meta_path):
                meta_df = pd.read_excel(meta_path, sheet_name="Sample_Product & Category", engine="openpyxl")
                # Clean column names
                meta_df.columns = (
                    meta_df.columns.astype(str).str.strip()
                    .str.replace(r"[\s\xa0]+", " ", regex=True)
                )
                
                cols_to_use = ["ITEMNUMBER"]
                rename_map = {"ITEMNUMBER": "ItemNumber"}
                
                if "product_category_name" in meta_df.columns:
                    cols_to_use.append("product_category_name")
                    rename_map["product_category_name"] = "Category"
                    
                if "PRODUCTNAME" in meta_df.columns:
                    cols_to_use.append("PRODUCTNAME")
                    rename_map["PRODUCTNAME"] = "ProductName"
                
                if len(cols_to_use) > 1:
                    meta_df = meta_df[cols_to_use].rename(columns=rename_map)
                    # Merge
                    processed_df = processed_df.merge(meta_df, on="ItemNumber", how="left")
                    processed_df["Category"] = processed_df["Category"].fillna("Unknown")
                    processed_df["ProductName"] = processed_df["ProductName"].fillna("Unknown")
                else:
                    processed_df["Category"] = "Unknown"
                    processed_df["ProductName"] = "Unknown"
            else:
                processed_df["Category"] = "Unknown"
                processed_df["ProductName"] = "Unknown"
        except Exception as e:
            st.warning(f"Failed to merge product metadata: {e}")
            processed_df["Category"] = "Unknown"
            processed_df["ProductName"] = "Unknown"
    
    # Create ItemDisplay column
    processed_df["ItemDisplay"] = processed_df["ItemNumber"].astype(str) + " - " + processed_df["ProductName"].astype(str)
            
    # Save to cache
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        processed_df.to_parquet(PROCESSED_CACHE_FILE, index=False)
        missing_summary_df.to_parquet(MISSING_SUMMARY_CACHE_FILE, index=False)
    except Exception as e:
        st.warning(f"Failed to cache data: {e}")
        
    return processed_df, missing_summary_df



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

    # Fill Names
    if "CustGroupName" in df.columns:
        df["CustGroupName"] = df["CustGroupName"].fillna("Unknown")
    if "CompanyChainName" in df.columns:
        df["CompanyChainName"] = df["CompanyChainName"].fillna("Unknown")

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



# -------------------------------------------------
# Aggregation & Metrics (Cached)
# -------------------------------------------------
@st.cache_data
def aggregate_sales(df):
    """Aggregates sales data."""
    if df.empty: return pd.DataFrame()
    
    group_cols = ["Year", "Month", "ItemNumber", "State", "CustGroup", "ItemDisplay"]
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
        df.groupby(["ItemNumber", "ItemDisplay"])["total_qty"]
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

def calculate_custom_forecast_comparison(df, w_weight=0.6, d_weight=0.4):
    """
    Calculates custom forecast on historical data for comparison.
    Formula: (W1+W2+W3)/3 * w_weight + (D1+D2)/2 * d_weight
    W1, W2, W3 = Lags 7, 14, 21 (Same day previous weeks)
    D1, D2 = Lags 2, 1 (Day before yesterday, Yesterday)
    """
    if df.empty:
        return pd.DataFrame()
        
    # Aggregate to daily level first (in case of multiple records per day)
    daily_df = (
        df.groupby("InvoiceDate")
        .agg(Actual=("INVOICEDQUANTITY", "sum"))
        .reset_index()
        .sort_values("InvoiceDate")
    )
    
    # Reindex to ensure continuous dates (fill missing days with 0)
    # This is crucial for correct lag calculation
    full_idx = pd.date_range(start=daily_df["InvoiceDate"].min(), end=daily_df["InvoiceDate"].max(), freq="D")
    daily_df = daily_df.set_index("InvoiceDate").reindex(full_idx, fill_value=0).reset_index()
    daily_df = daily_df.rename(columns={"index": "InvoiceDate"})
    
    # Calculate Lags
    # W components (Weekly lags)
    daily_df["W1"] = daily_df["Actual"].shift(7).fillna(0)
    daily_df["W2"] = daily_df["Actual"].shift(14).fillna(0)
    daily_df["W3"] = daily_df["Actual"].shift(21).fillna(0)
    
    # D components (Daily lags)
    daily_df["D1"] = daily_df["Actual"].shift(2).fillna(0) # Day before yesterday
    daily_df["D2"] = daily_df["Actual"].shift(1).fillna(0) # Yesterday
    
    # Calculate Forecast
    # Note: We need all components to be present, so first 21 days will be NaN
    daily_df["W_Avg"] = (daily_df["W1"] + daily_df["W2"] + daily_df["W3"]) / 3
    daily_df["D_Avg"] = (daily_df["D1"] + daily_df["D2"]) / 2
    
    daily_df["Forecast"] = daily_df["W_Avg"] * w_weight + daily_df["D_Avg"] * d_weight
    
    return daily_df

def plot_yoy_comparison(df, title_prefix="Sales"):
    """
    Generates a YoY comparison plot (2024 vs 2025) aligned by Month-Day.
    """
    # Group by Year and MonthDay
    daily_season = (
        df.groupby(["Year", "MonthDay"])
        .agg(total_qty=("INVOICEDQUANTITY", "sum"))
        .reset_index()
    )
    
    # Explicitly sort by MonthDay
    daily_season = daily_season.sort_values("MonthDay")
    
    # Ensure Year is string for discrete color
    daily_season["Year"] = daily_season["Year"].astype(str)
    
    sorted_month_days = sorted(daily_season["MonthDay"].unique())
    
    fig = px.line(daily_season, x="MonthDay", y="total_qty", color="Year", 
                  title=f"{title_prefix}: 2024 vs 2025 Comparison",
                  markers=True,
                  category_orders={"MonthDay": sorted_month_days})
    fig.update_xaxes(type='category', tickangle=-45)
    fig.update_layout(margin=dict(b=80))
    return fig

def plot_period_performance(df, period_col, title_suffix, x_label=None):
    """
    Generates Raw and YoY Growth plots for a given period (Month, Week, Day).
    """
    # Aggregate
    agg = df.groupby([period_col, "Year"])["INVOICEDQUANTITY"].sum().reset_index()
    agg.columns = [period_col, "Year", "total_qty"]
    agg["Year"] = agg["Year"].astype(str)
    
    # Calculate YoY Growth
    pivot = agg.pivot(index=period_col, columns="Year", values="total_qty").fillna(0)
    if "2024" in pivot.columns and "2025" in pivot.columns:
        pivot["yoy_pct"] = ((pivot["2025"] - pivot["2024"]) / pivot["2024"]) * 100
    else:
        pivot["yoy_pct"] = 0
    
    pivot = pivot.reset_index()
    # Merge back to get long format for plotting
    merged = agg.merge(pivot[[period_col, "yoy_pct"]], on=period_col, how="left")
    
    # Sort if needed (for Day of Week)
    if period_col == "DayOfWeek":
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        merged["DayName"] = merged["DayOfWeek"].map(day_map)
        x_col = "DayName"
    else:
        x_col = period_col

    # Plot Raw
    fig_raw = px.line(merged, x=x_col, y="total_qty", color="Year", markers=True, title=f"{title_suffix} (Raw)")
    fig_raw.update_xaxes(tickangle=-45)
    fig_raw.update_layout(margin=dict(b=80))
    if period_col == "Month":
        fig_raw.update_xaxes(dtick=1)
        
    # Plot YoY
    fig_yoy = px.bar(merged, x=x_col, y="yoy_pct", color="Year", title=f"{title_suffix} YoY Growth (%)")
    fig_yoy.update_xaxes(tickangle=-45)
    fig_yoy.update_layout(margin=dict(b=80))
    if period_col == "Month":
        fig_yoy.update_xaxes(dtick=1)
        
    return fig_raw, fig_yoy

def plot_day_of_week_forecast_comparison(comp_df):
    """
    Plots Average Actual vs Average Forecast by Day of Week.
    """
    if comp_df.empty:
        return None
        
    comp_df["DayOfWeek"] = comp_df["InvoiceDate"].dt.dayofweek
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    comp_df["DayName"] = comp_df["DayOfWeek"].map(day_map)
    
    # Group by Day
    dow_perf = comp_df.groupby("DayName")[["Actual", "Forecast"]].mean().reindex(day_map.values()).reset_index()
    
    # Melt for plotting
    dow_melt = dow_perf.melt(id_vars="DayName", value_vars=["Actual", "Forecast"], var_name="Type", value_name="Avg_Qty")
    
    fig = px.bar(dow_melt, x="DayName", y="Avg_Qty", color="Type", barmode="group",
                 title="Day-of-Week Performance: Actual vs Forecast",
                 color_discrete_map={"Actual": "#1E90FF", "Forecast": "#FF8C00"})
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(margin=dict(b=80))
    return fig

def plot_period_performance_by_item(df, period_col, title_suffix):
    """
    Generates Raw and YoY Growth plots split by Item.
    Raw: Line chart (Color=Item, Dash=Year)
    YoY: Bar chart (Color=Item, x=Period)
    """
    # Aggregate
    agg = df.groupby([period_col, "Year", "ItemNumber", "ProductName", "ItemDisplay"])["INVOICEDQUANTITY"].sum().reset_index()
    agg.columns = [period_col, "Year", "ItemNumber", "ProductName", "ItemDisplay", "total_qty"]
    agg["Year"] = agg["Year"].astype(str)
    
    # Sort if needed
    if period_col == "DayOfWeek":
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        agg["DayName"] = agg["DayOfWeek"].map(day_map)
        x_col = "DayName"
        # Sort order
        sorter = list(day_map.values())
    else:
        x_col = period_col
        sorter = None

    # Calculate dynamic height
    n_items = agg["ItemNumber"].nunique()
    n_rows = (n_items + 2) // 3
    fig_height = max(400, n_rows * 350)

    # Plot Raw
    fig_raw = px.line(agg, x=x_col, y="total_qty", color="Year", line_dash="Year",
                      markers=True, title=f"{title_suffix} (Raw - Split by Item)",
                      hover_data=["ProductName"],
                      facet_col="ItemDisplay", facet_col_wrap=3,
                      facet_row_spacing=0.15, height=fig_height)
    if period_col == "Month":
        fig_raw.update_xaxes(dtick=1)
    if sorter:
        fig_raw.update_xaxes(categoryorder="array", categoryarray=sorter)
    
    # Fix x-axis visibility
    # Fix x-axis visibility
    fig_raw.update_xaxes(tickangle=-45, matches=None, showticklabels=True)
    fig_raw.update_layout(margin=dict(b=80)) # Add margin for rotated labels
    
    # Improve layout for subplots
    fig_raw.update_yaxes(matches=None) # Allow independent y-axes if needed, or 'y' to share
    fig_raw.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
    # Calculate YoY Growth per Item
    pivot = agg.pivot_table(index=[period_col, "ItemNumber", "ProductName", "ItemDisplay"], columns="Year", values="total_qty", fill_value=0).reset_index()
    if "2024" in pivot.columns and "2025" in pivot.columns:
        pivot["yoy_pct"] = ((pivot["2025"] - pivot["2024"]) / pivot["2024"].replace(0, 1)) * 100 # Avoid div by zero
    else:
        pivot["yoy_pct"] = 0
        
    if period_col == "DayOfWeek":
        pivot["DayName"] = pivot["DayOfWeek"].map(day_map)
    
    # Plot YoY
    fig_yoy = px.bar(pivot, x=x_col, y="yoy_pct", color="ItemNumber", barmode="group",
                     title=f"{title_suffix} YoY Growth (%) - Split by Item",
                     hover_data=["ProductName"],
                     facet_col="ItemDisplay", facet_col_wrap=3,
                     facet_row_spacing=0.15, height=fig_height)
    if period_col == "Month":
        fig_yoy.update_xaxes(dtick=1)
    if sorter:
        fig_yoy.update_xaxes(categoryorder="array", categoryarray=sorter)
    
    # Fix x-axis visibility
    # Fix x-axis visibility
    fig_yoy.update_xaxes(tickangle=-45, matches=None, showticklabels=True)
    fig_yoy.update_layout(margin=dict(b=80))
    
    fig_yoy.update_yaxes(matches=None)
    fig_yoy.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
    return fig_raw, fig_yoy

def plot_forecast_comparison_by_item(plot_df):
    """
    Plots Actual vs Forecast with faceting by Item.
    """
    if plot_df.empty:
        return None
        
    # Melt for plotting
    plot_melt = plot_df.melt(id_vars=["InvoiceDate", "ItemNumber", "ItemDisplay"], 
                           value_vars=["Actual", "Forecast"], 
                           var_name="Type", value_name="Qty")
    
    # Calculate dynamic height
    n_items = plot_df["ItemNumber"].nunique()
    n_rows = (n_items + 2) // 3
    fig_height = max(400, n_rows * 350)

    fig = px.line(plot_melt, x="InvoiceDate", y="Qty", color="Type", line_dash="Type",
                  title=f"Actual vs Forecasted Sales (Split by Item)",
                  markers=True,
                  color_discrete_map={"Actual": "#1E90FF", "Forecast": "#FF8C00"},
                  facet_col="ItemDisplay", facet_col_wrap=3,
                  facet_row_spacing=0.15, height=fig_height)
                  
    fig.update_xaxes(tickangle=-45, matches=None, showticklabels=True)
    fig.update_layout(margin=dict(b=80))
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def plot_dow_forecast_by_item(comp_df):
    """
    Plots Average Actual vs Average Forecast by Day of Week, split by Item.
    """
    if comp_df.empty:
        return None
        
    comp_df["DayOfWeek"] = comp_df["InvoiceDate"].dt.dayofweek
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    comp_df["DayName"] = comp_df["DayOfWeek"].map(day_map)
    
    # Group by Day and Item
    dow_perf = comp_df.groupby(["DayName", "ItemNumber", "ItemDisplay"])[["Actual", "Forecast"]].mean().reset_index()
    
    # Melt
    dow_melt = dow_perf.melt(id_vars=["DayName", "ItemNumber", "ItemDisplay"], value_vars=["Actual", "Forecast"], var_name="Type", value_name="Avg_Qty")
    
    # Plot: Color=Item, Pattern=Type
    # Calculate dynamic height
    n_items = comp_df["ItemNumber"].nunique()
    n_rows = (n_items + 2) // 3
    fig_height = max(400, n_rows * 350)

    fig = px.bar(dow_melt, x="DayName", y="Avg_Qty", color="Type", pattern_shape="ItemNumber", barmode="group",
                 title="Day-of-Week Performance: Actual vs Forecast (Split by Item)",
                 category_orders={"DayName": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
                 color_discrete_map={"Actual": "#1E90FF", "Forecast": "#FF8C00"},
                 facet_col="ItemDisplay", facet_col_wrap=3,
                 facet_row_spacing=0.15, height=fig_height)
                 
    fig.update_xaxes(tickangle=-45, matches=None, showticklabels=True)
    fig.update_layout(margin=dict(b=80))
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

def render_analysis_tab(df, group_col, label, tab_key_prefix):
    """
    Renders a standard analysis tab for a given grouping column (e.g., Category, State).
    """
    st.subheader(f"📦 {label} Analysis")
    st.markdown(f"Analyze item performance within a specific {label}.")
    
    # Ensure column exists
    if group_col not in df.columns:
        st.error(f"Column '{group_col}' not found in data.")
        return
        
    # --- Prepare Group Selection Options ---
    # If we have a name column for the group, use it
    group_name_col = None
    if group_col == "CustGroup" and "CustGroupName" in df.columns:
        group_name_col = "CustGroupName"
    elif group_col == "CompanyChain" and "CompanyChainName" in df.columns:
        group_name_col = "CompanyChainName"
        
    # Get unique groups
    if group_name_col:
        # Create "ID - Name" label
        unique_groups = df[[group_col, group_name_col]].drop_duplicates().sort_values(group_col)
        unique_groups["DisplayLabel"] = unique_groups[group_col].astype(str) + " - " + unique_groups[group_name_col].astype(str)
        group_options = unique_groups["DisplayLabel"].tolist()
        # Map label back to ID
        group_map = dict(zip(unique_groups["DisplayLabel"], unique_groups[group_col]))
    else:
        group_options = sorted(df[group_col].astype(str).unique())
        group_map = {g: g for g in group_options}

    selected_group_label = st.selectbox(f"Select {label}", options=group_options, key=f"{tab_key_prefix}_select")
    
    if selected_group_label:
        # Resolve ID
        selected_group_id = group_map[selected_group_label]
        
        # Filter by ID (robust)
        if isinstance(selected_group_id, (int, float)):
             group_df = df[df[group_col] == selected_group_id]
        else:
             group_df = df[df[group_col].astype(str) == str(selected_group_id)]
        
        # --- Prepare Item Selection Options ---
        # Get all items in group with names
        if "ItemDisplay" in df.columns:
            unique_items = group_df[["ItemNumber", "ItemDisplay"]].drop_duplicates().sort_values("ItemNumber")
            item_options = unique_items["ItemDisplay"].tolist()
            item_map = dict(zip(unique_items["ItemDisplay"], unique_items["ItemNumber"]))
        else:
            item_options = sorted(group_df["ItemNumber"].unique())
            item_map = {i: i for i in item_options}
        
        # Select Items
        selected_item_labels = st.multiselect(
            f"Select Items to Compare ({len(item_options)} items found)", 
            options=item_options,
            default=item_options[:5] if len(item_options) > 5 else item_options, # Default to top 5
            key=f"{tab_key_prefix}_item_select"
        )
        
        if selected_item_labels:
            # Map labels back to IDs
            selected_item_ids = [item_map[label] for label in selected_item_labels]
            plot_df = group_df[group_df["ItemNumber"].isin(selected_item_ids)]
            
            # 1. Month-wise
            st.markdown("### 📅 Month-wise Performance")
            fig_m_raw, fig_m_yoy = plot_period_performance_by_item(plot_df, "Month", "Monthly Sales")
            st.plotly_chart(fig_m_raw, use_container_width=True)
            st.plotly_chart(fig_m_yoy, use_container_width=True)

            # 2. Week-wise
            st.markdown("### 📆 Week-wise Performance")
            fig_w_raw, fig_w_yoy = plot_period_performance_by_item(plot_df, "WeekOfYear", "Weekly Sales")
            st.plotly_chart(fig_w_raw, use_container_width=True)
            st.plotly_chart(fig_w_yoy, use_container_width=True)

            # 3. Day-of-Week
            st.markdown("### 🗓️ Day-of-Week Performance")
            fig_d_raw, fig_d_yoy = plot_period_performance_by_item(plot_df, "DayOfWeek", "Day-of-Week Sales")
            st.plotly_chart(fig_d_raw, use_container_width=True)
            st.plotly_chart(fig_d_yoy, use_container_width=True)

            # 4. Actual vs Forecast Comparison (Item-wise)
            st.markdown("### 🎯 Actual vs Forecast Comparison (Split by Item)")
            
            c_w, c_d = st.columns(2)
            w_weight = c_w.number_input("Week Weight (W_Avg)", value=0.6, step=0.1, min_value=0.0, max_value=1.0, key=f"{tab_key_prefix}_w")
            d_weight = c_d.number_input("Day Weight (D_Avg)", value=0.4, step=0.1, min_value=0.0, max_value=1.0, key=f"{tab_key_prefix}_d")
            
            # Calculate forecast for EACH item individually
            all_forecasts = []
            for item in selected_item_ids:
                single_item_df = plot_df[plot_df["ItemNumber"] == item]
                if not single_item_df.empty:
                    f_df = calculate_custom_forecast_comparison(single_item_df, w_weight, d_weight)
                    if not f_df.empty:
                        f_df["ItemNumber"] = item
                        if "ItemDisplay" in single_item_df.columns:
                            f_df["ItemDisplay"] = single_item_df["ItemDisplay"].iloc[0]
                        all_forecasts.append(f_df)
            
            if all_forecasts:
                comp_df = pd.concat(all_forecasts)
                plot_forecast_df = comp_df.dropna(subset=["Forecast"])
                
                if not plot_forecast_df.empty:
                    st.caption(f"ℹ️ Calculation using: Week Weight = **{w_weight}**, Day Weight = **{d_weight}**")
                    
                    # Use faceted function
                    fig_comp = plot_forecast_comparison_by_item(plot_forecast_df)
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # New: Day-of-Week Actual vs Forecast (Split by Item)
                    st.markdown("### 📊 Day-of-Week Performance: Actual vs Forecast")
                    fig_dow_comp = plot_dow_forecast_by_item(plot_forecast_df)
                    if fig_dow_comp:
                        st.plotly_chart(fig_dow_comp, use_container_width=True)
                else:
                    st.warning("Not enough data to calculate forecast.")
            else:
                st.warning("No data available for forecast.")

            # Summary Table
            st.markdown("### 📋 Item Performance Summary")
            summary_table = (
                plot_df.groupby(["ItemNumber", "ProductName"])
                .agg(
                    Total_Sales=("INVOICEDQUANTITY", "sum"),
                    Avg_Daily_Sales=("INVOICEDQUANTITY", "mean"),
                    Days_Active=("InvoiceDate", "nunique")
                )
                .reset_index()
                .sort_values("Total_Sales", ascending=False)
            )
            st.dataframe(summary_table, hide_index=True, width="stretch")
            
        else:
            st.info("Select items to visualize.")

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

# -------------------------------------------------
# Sidebar Filters
# -------------------------------------------------
st.sidebar.header("Filters")

# Helper to add "All" option implicitly (empty selection = All)
def add_filter(label, column):
    options = sorted(clean_df[column].astype(str).unique())
    selected = st.sidebar.multiselect(f"Select {label}", options)
    return selected

selected_dataareaid = add_filter("DATAAREAID", "DATAAREAID")
selected_custgroup = add_filter("CustGroup", "CustGroup")
selected_companychain = add_filter("CompanyChain", "CompanyChain")
selected_state = add_filter("State", "State")

# Apply Filters
if selected_dataareaid:
    clean_df = clean_df[clean_df["DATAAREAID"].astype(str).isin(selected_dataareaid)]
if selected_custgroup:
    clean_df = clean_df[clean_df["CustGroup"].astype(str).isin(selected_custgroup)]
if selected_companychain:
    clean_df = clean_df[clean_df["CompanyChain"].astype(str).isin(selected_companychain)]
if selected_state:
    clean_df = clean_df[clean_df["State"].astype(str).isin(selected_state)]

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
    "📦 Category Analysis",
    "🗺️ State Analysis",
    "👥 CustGroup Analysis",
    "🏢 CompanyChain Analysis",
    "🧠 Demand Patterns",
    "🔍 Item Analysis",
    "⚠️ Data Quality"
])

# -------------------------------------------------
# Tab 1: Sales Analysis
# -------------------------------------------------
with tabs[0]:
    st.subheader("Sales Trends")
    st.subheader("Sales Trends (YoY Comparison)")
    
    # YoY Comparison for Total Sales
    fig_sales_yoy = plot_yoy_comparison(clean_df, title_prefix="Total Sales")
    st.plotly_chart(fig_sales_yoy, use_container_width=True)
    
    st.markdown("### 🗓️ Daily Sales (Continuous)")
    
    st.subheader("Sales by State")
    state_sales = clean_df.groupby("State")["INVOICEDQUANTITY"].sum().reset_index().sort_values("INVOICEDQUANTITY", ascending=False)
    fig_state = px.bar(state_sales, x="State", y="INVOICEDQUANTITY", title="Total Sales by State")
    fig_state.update_xaxes(tickangle=-45)
    fig_state.update_layout(margin=dict(b=80))
    st.plotly_chart(fig_state, use_container_width=True)

# -------------------------------------------------
# Tab 2: Category Analysis
# -------------------------------------------------
with tabs[1]:
    render_analysis_tab(clean_df, "Category", "Category", "cat")

# -------------------------------------------------
# Tab 3: State Analysis
# -------------------------------------------------
with tabs[2]:
    render_analysis_tab(clean_df, "State", "State", "state")

# -------------------------------------------------
# Tab 4: CustGroup Analysis
# -------------------------------------------------
with tabs[3]:
    render_analysis_tab(clean_df, "CustGroup", "Customer Group", "cust")

# -------------------------------------------------
# Tab 5: CompanyChain Analysis
# -------------------------------------------------
with tabs[4]:
    render_analysis_tab(clean_df, "CompanyChain", "Company Chain", "chain")

# -------------------------------------------------
# Tab 6: Demand Patterns
# -------------------------------------------------
with tabs[5]:
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
# Tab 4: Item Analysis
# -------------------------------------------------
# -------------------------------------------------
# Tab 7: Item Analysis
# -------------------------------------------------
with tabs[6]:
    st.subheader("Deep Dive: Item Analysis")
    st.markdown("Analyze individual product performance across the **Oct-Dec** periods of 2024 and 2025.")
    
    # Get unique items with display label
    unique_items = agg_df[["ItemNumber", "ItemDisplay"]].drop_duplicates().sort_values("ItemNumber")
    item_options = unique_items["ItemDisplay"].tolist()
    item_map = dict(zip(unique_items["ItemDisplay"], unique_items["ItemNumber"]))
    
    top_items_id = (
        agg_df.groupby("ItemNumber")["total_qty"].sum()
        .sort_values(ascending=False).head(100).index.tolist()
    )
    
    # Find display label for top item
    default_index = 0
    if top_items_id:
        top_id = top_items_id[0]
        # Find label safely
        match = unique_items[unique_items["ItemNumber"] == top_id]
        if not match.empty:
            top_label = match["ItemDisplay"].iloc[0]
            if top_label in item_options:
                default_index = item_options.index(top_label)
    
    selected_item_label = st.selectbox(
        "Select Item", 
        options=item_options,
        index=default_index
    )
    
    if selected_item_label:
        selected_item = item_map[selected_item_label]
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
        item_df_filtered = clean_df[clean_df["ItemNumber"] == selected_item].copy()
        
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
        fig_season.update_xaxes(type='category', tickangle=-45) 
        fig_season.update_layout(margin=dict(b=80))
        st.plotly_chart(fig_season, use_container_width=True)

        # 1. Month-wise
        st.markdown("### 📅 Month-wise Performance")
        fig_m_raw, fig_m_yoy = plot_period_performance(item_df_filtered, "Month", "Monthly Sales")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fig_m_raw, use_container_width=True)
        with c2: st.plotly_chart(fig_m_yoy, use_container_width=True)

        # 2. Week-wise
        st.markdown("### 📆 Week-wise Performance")
        fig_w_raw, fig_w_yoy = plot_period_performance(item_df_filtered, "WeekOfYear", "Weekly Sales")
        c3, c4 = st.columns(2)
        with c3: st.plotly_chart(fig_w_raw, use_container_width=True)
        with c4: st.plotly_chart(fig_w_yoy, use_container_width=True)

        # 3. Day-of-Week
        st.markdown("### 🗓️ Day-of-Week Performance")
        fig_d_raw, fig_d_yoy = plot_period_performance(item_df_filtered, "DayOfWeek", "Day-of-Week Sales")
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(fig_d_raw, use_container_width=True)
        with c6: st.plotly_chart(fig_d_yoy, use_container_width=True)

        # 5. Actual vs Forecast Comparison (Custom Formula)
        st.markdown("### 🎯 Actual vs Forecast Comparison")
        st.markdown("""
        **Formula**: `Forecast = (W_Avg * Week_Weight) + (D_Avg * Day_Weight)`
        *   `W_Avg`: Average of same day for last 3 weeks (Lag 7, 14, 21)
        *   `D_Avg`: Average of last 2 days (Lag 1, 2)
        """)
        
        c_w, c_d = st.columns(2)
        w_weight = c_w.number_input("Week Weight (W_Avg)", value=0.6, step=0.1, min_value=0.0, max_value=1.0)
        d_weight = c_d.number_input("Day Weight (D_Avg)", value=0.4, step=0.1, min_value=0.0, max_value=1.0)
        
        comp_df = calculate_custom_forecast_comparison(item_df_filtered, w_weight, d_weight)
        
        if not comp_df.empty:
            # Filter out the initial NaN period for cleaner plotting
            plot_df = comp_df.dropna(subset=["Forecast"])
            
            if not plot_df.empty:
                st.caption(f"ℹ️ Calculation using: Week Weight = **{w_weight}**, Day Weight = **{d_weight}**")
                
                fig_comp = px.line(plot_df, x="InvoiceDate", y=["Actual", "Forecast"],
                                   title=f"Actual vs Forecasted Sales (W={w_weight}, D={d_weight})",
                                   color_discrete_map={"Actual": "#1E90FF", "Forecast": "#FF8C00"},
                                   hover_data={"W_Avg": ":.2f", "D_Avg": ":.2f"})
                fig_comp.update_traces(mode="lines+markers")
                fig_comp.update_xaxes(tickangle=-45)
                fig_comp.update_layout(margin=dict(b=80))
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # New: Day-of-Week Actual vs Forecast
                st.markdown("### 📊 Day-of-Week Performance: Actual vs Forecast")
                fig_dow_comp = plot_day_of_week_forecast_comparison(plot_df)
                if fig_dow_comp:
                    st.plotly_chart(fig_dow_comp, use_container_width=True)
            else:
                st.warning("Not enough data to calculate forecast (Need at least 21 days of history).")



# -------------------------------------------------
# Tab 8: Data Quality
# -------------------------------------------------
with tabs[7]:
    st.subheader("⚠️ Data Quality Checks")
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
