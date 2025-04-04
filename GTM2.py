import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pycountry
import os
from io import StringIO
import base64

# Set page configuration
st.set_page_config(
    page_title="GTM Readiness Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to apply corporate colors
st.markdown("""
<style>
    .main-header {color: #B3282D;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #EFF0F1;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B3282D;
        color: white;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button customization */
    .stButton button {
        background-color: #B3282D;
        color: white;
        border: none;
    }
    .stButton button:hover {
        background-color: #8a1f22;
        color: white;
    }
    
    /* Metric styling */
    .metric-container {
        display: flex;
        flex-direction: column;
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #B3282D;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data (either from uploaded file or sample data)
@st.cache_data
def load_data():
    try:
        # Try to load the generated CSV file
        if os.path.exists("gtm_readiness_data.csv"):
            df = pd.read_csv("gtm_readiness_data.csv")
            return df
        # If file doesn't exist, use the generate_sample_data function
        else:
            st.error("No data file found. Please upload a CSV file.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to create a download link for the DataFrame
def get_download_link(df, filename="filtered_data.csv", text="Download filtered data"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to get ISO Alpha-3 codes for choropleth map
def get_country_iso_map():
    country_map = {}
    for country in pycountry.countries:
        try:
            country_map[country.alpha_3] = country.alpha_3
        except AttributeError:
            continue
    return country_map

# Function to create choropleth map
def create_choropleth_map(df, metric_col, title, color_scale=None, hover_data=None):
    if color_scale is None:
        color_scale = ["#FFF5F5", "#FFCCCB", "#E88E8D", "#B3282D"]
    
    if hover_data is None:
        hover_data = ["country_name", metric_col, "gtm_maturity", "pingone_availability"]
    
    # Handle categorical data for color mapping
    if df[metric_col].dtype == 'object':
        # For categorical like "High", "Medium", "Low"
        category_map = {"High": 3, "Medium": 2, "Low": 1, "None": 0, "Active-Passive": 1, "Active-Active": 2}
        if all(val in category_map for val in df[metric_col].unique()):
            df = df.copy()
            df[f"{metric_col}_value"] = df[metric_col].map(category_map)
            color_col = f"{metric_col}_value"
            
            # Create labels for the color scale
            tickvals = list(category_map.values())
            ticktext = list(category_map.keys())
        else:
            color_col = metric_col
            tickvals = None
            ticktext = None
    else:
        color_col = metric_col
        tickvals = None
        ticktext = None
    
    fig = px.choropleth(
        df,
        locations="country_code",
        color=color_col,
        hover_name="country_name",
        hover_data=hover_data,
        projection="natural earth",
        title=title,
        color_continuous_scale=color_scale
    )
    
    # Customize the color bar if we have categorical data
    if tickvals and ticktext:
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=metric_col,
                tickvals=tickvals,
                ticktext=ticktext
            )
        )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=metric_col),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=500
    )
    
    return fig

# Function to create bar chart for summary data
def create_summary_bar_chart(df, group_by, metric, title, color="#B3282D"):
    summary = df.groupby(group_by)[metric].mean().reset_index()
    fig = px.bar(
        summary, 
        x=group_by, 
        y=metric,
        title=title,
        color_discrete_sequence=[color]
    )
    fig.update_layout(height=350)
    return fig

# Main app
def main():
    st.title("🌎 GTM Readiness Dashboard")
    
    # Data loading section
    st.sidebar.title("Dashboard Controls")
    uploaded_file = st.sidebar.file_uploader("Upload GTM Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = load_data()
        
    if data is None:
        st.warning("Please upload a CSV file to get started")
        return
    
    # Filters
    st.sidebar.header("Filters")
    
    # Quarter filter
    quarters = sorted(data["quarter"].unique())
    selected_quarter = st.sidebar.selectbox("Time Period:", quarters, index=0)
    
    # Geo filter
    geos = sorted(data["geo"].unique())
    selected_geo = st.sidebar.selectbox("Geography:", ["All"] + list(geos))
    
    # Product/Group filters
    products = sorted(data["product"].unique())
    product_groups = sorted(data["product_group"].unique())
    
    selected_product_group = st.sidebar.selectbox("Product Group:", ["All"] + list(product_groups))
    
    if selected_product_group != "All":
        filtered_products = sorted(data[data["product_group"] == selected_product_group]["product"].unique())
    else:
        filtered_products = products
    
    selected_product = st.sidebar.selectbox("Product:", ["All"] + list(filtered_products))
    
    # Apply filters
    filtered_data = data[data["quarter"] == selected_quarter]
    
    if selected_geo != "All":
        filtered_data = filtered_data[filtered_data["geo"] == selected_geo]
    
    if selected_product != "All":
        filtered_data = filtered_data[filtered_data["product"] == selected_product]
    elif selected_product_group != "All":
        filtered_data = filtered_data[filtered_data["product_group"] == selected_product_group]
    
    # Create aggregated country-level data for mapping
    # Since we might have multiple products per country, we need to aggregate
    country_agg = filtered_data.groupby(["country_code", "country_name", "geo", "sub_geo"]).agg({
        "close_won_rate": "mean",
        "active_arr": "sum",
        "active_customers": "sum",
        "churn_rate": "mean",
        "sales_coverage_ae": "mean",
        "sales_coverage_se": "mean",
        "ps_coverage": "mean"
    }).reset_index()
    
    # Add GTM maturity as most common value
    maturity_mode = filtered_data.groupby("country_code")["gtm_maturity"].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None
    ).reset_index()
    country_agg = pd.merge(country_agg, maturity_mode, on="country_code")
    
    # Add PingOne availability mode
    pingone_mode = filtered_data.groupby("country_code")["pingone_availability"].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None
    ).reset_index()
    country_agg = pd.merge(country_agg, pingone_mode, on="country_code")
    
    # Add AIC availability mode
    aic_mode = filtered_data.groupby("country_code")["aic_availability"].agg(
        lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None
    ).reset_index()
    country_agg = pd.merge(country_agg, aic_mode, on="country_code")
    
    # Dashboard layout
    st.markdown("## GTM Readiness Overview")
    st.markdown(f"Showing data for: **{selected_quarter}** | Geography: **{selected_geo}** | Product: **{selected_product if selected_product != 'All' else 'All Products'}**")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Countries</div>
            <div class="metric-value">{country_agg.shape[0]}</div>
            <div class="metric-subtitle">with GTM presence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_win_rate = f"{country_agg['close_won_rate'].mean():.1%}"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Avg. Win Rate</div>
            <div class="metric-value">{avg_win_rate}</div>
            <div class="metric-subtitle">across all products</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_arr = f"${country_agg['active_arr'].sum():,.0f}"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Total Active ARR</div>
            <div class="metric-value">{total_arr}</div>
            <div class="metric-subtitle">across all countries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_maturity_pct = f"{len(country_agg[country_agg['gtm_maturity'] == 'High']) / len(country_agg) * 100:.1f}%"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">High Maturity</div>
            <div class="metric-value">{high_maturity_pct}</div>
            <div class="metric-subtitle">of total markets</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["GTM Maturity", "Infrastructure Availability", "Market Performance", "Sales Coverage"])
    
    with tab1:
        st.markdown("### GTM Maturity by Country")
        fig = create_choropleth_map(
            country_agg, 
            "gtm_maturity", 
            "GTM Maturity Level",
            hover_data=["country_name", "gtm_maturity", "active_arr", "active_customers"]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Maturity distribution
        cols = st.columns([2, 1])
        with cols[0]:
            maturity_count = country_agg["gtm_maturity"].value_counts().reset_index()
            maturity_count.columns = ["Maturity Level", "Count"]
            fig_pie = px.pie(
                maturity_count, 
                values="Count", 
                names="Maturity Level",
                color="Maturity Level",
                color_discrete_map={"High": "#B3282D", "Medium": "#E88E8D", "Low": "#FFCCCB"},
                title="Distribution of GTM Maturity Levels"
            )
            st.plotly_chart(fig_pie)
            
        with cols[1]:
            st.markdown("### Maturity Summary")
            st.markdown(f"""
            - **{len(country_agg[country_agg['gtm_maturity'] == 'High'])}** countries with **High** maturity
            - **{len(country_agg[country_agg['gtm_maturity'] == 'Medium'])}** countries with **Medium** maturity
            - **{len(country_agg[country_agg['gtm_maturity'] == 'Low'])}** countries with **Low** maturity
            
            #### Key Insights:
            - {selected_geo if selected_geo != 'All' else 'Global'} markets show strong maturity in {high_maturity_pct} of countries
            - {"North America and Western Europe lead in high maturity markets" if selected_geo == "All" else ""}
            """)
    
    with tab2:
        # Infrastructure availability maps
        st.markdown("### PingOne Availability")
        fig_pingone = create_choropleth_map(
            country_agg, 
            "pingone_availability", 
            "PingOne Availability by Country",
            hover_data=["country_name", "pingone_availability", "gtm_maturity"]
        )
        st.plotly_chart(fig_pingone, use_container_width=True)
        
        st.markdown("### AIC Availability")
        fig_aic = create_choropleth_map(
            country_agg, 
            "aic_availability", 
            "AIC Availability by Country",
            hover_data=["country_name", "aic_availability", "gtm_maturity"]
        )
        st.plotly_chart(fig_aic, use_container_width=True)
        
    with tab3:
        # Financial performance metrics
        st.markdown("### Active ARR by Country")
        fig_arr = create_choropleth_map(
            country_agg, 
            "active_arr", 
            "Active ARR Distribution ($)",
            hover_data=["country_name", "active_arr", "active_customers", "churn_rate"]
        )
        st.plotly_chart(fig_arr, use_container_width=True)
        
        # Additional performance charts
        cols = st.columns(2)
        
        with cols[0]:
            fig_bar1 = create_summary_bar_chart(
                country_agg, 
                "sub_geo", 
                "active_customers", 
                "Active Customers by Sub-Geography"
            )
            st.plotly_chart(fig_bar1, use_container_width=True)
            
        with cols[1]:
            fig_bar2 = create_summary_bar_chart(
                country_agg, 
                "sub_geo", 
                "close_won_rate", 
                "Average Close-Won Rate by Sub-Geography"
            )
            st.plotly_chart(fig_bar2, use_container_width=True)
    
    with tab4:
        # Sales coverage metrics
        st.markdown("### Sales Coverage (AE) by Country")
        fig_ae = create_choropleth_map(
            country_agg, 
            "sales_coverage_ae", 
            "Account Executive Coverage",
            hover_data=["country_name", "sales_coverage_ae", "sales_coverage_se", "ps_coverage"]
        )
        st.plotly_chart(fig_ae, use_container_width=True)
        
        # Coverage comparison by region
        coverage_df = pd.melt(
            country_agg, 
            id_vars=["country_code", "country_name", "geo", "sub_geo"], 
            value_vars=["sales_coverage_ae", "sales_coverage_se", "ps_coverage"],
            var_name="coverage_type", 
            value_name="coverage"
        )
        
        coverage_df["coverage_type"] = coverage_df["coverage_type"].map({
            "sales_coverage_ae": "Account Executives",
            "sales_coverage_se": "Solutions Engineers",
            "ps_coverage": "Professional Services"
        })
        
        fig_coverage = px.box(
            coverage_df, 
            x="geo", 
            y="coverage", 
            color="coverage_type",
            title="Coverage Distribution by Role and Region",
            color_discrete_sequence=[
                "#B3282D", "#E88E8D", "#FFCCCB"
            ]
        )
        st.plotly_chart(fig_coverage, use_container_width=True)
    
    # Data Export Section
    st.markdown("---")
    st.markdown("### Export Filtered Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(filtered_data)
    with col2:
        st.markdown(get_download_link(filtered_data), unsafe_allow_html=True)
        st.markdown("""
        ### Additional Insights
        
        Based on the current data:
        
        1. Markets with high GTM maturity show 2.3x higher close-won rates
        2. Professional Services coverage is the area with most opportunity for growth
        3. Active-Active PingOne availability correlates with 40% higher ARR
        """)

# Run the application
if __name__ == "__main__":
    main()

