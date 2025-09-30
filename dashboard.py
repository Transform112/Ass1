
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure the page
st.set_page_config(
    page_title="COVID-19 Global Data Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load COVID-19 data from CSV file"""
    try:
        # You'll need to upload your country_wise_latest.csv file
        df = pd.read_csv('country_wise_latest.csv')
        return df
    except FileNotFoundError:
        # Create sample data if file is not found
        sample_data = {
            'Country/Region': ['US', 'Brazil', 'India', 'Russia', 'South Africa', 'Mexico', 'Peru', 'Chile', 'United Kingdom', 'Iran'],
            'Confirmed': [4290259, 2442375, 1640000, 816000, 503000, 385036, 355000, 336000, 295000, 278000],
            'Deaths': [148011, 87618, 35747, 13334, 8153, 43374, 13384, 8722, 45358, 14405],
            'Recovered': [1325804, 1846641, 1091771, 592611, 347224, 275000, 245000, 308000, 0, 242000],
            'Active': [2816444, 508116, 512482, 210055, 147623, 66662, 96616, 19278, 249642, 21595],
            'WHO Region': ['Americas', 'Americas', 'South-East Asia', 'Europe', 'Africa', 'Americas', 'Americas', 'Americas', 'Europe', 'Eastern Mediterranean'],
            'Deaths / 100 Cases': [3.45, 3.59, 2.18, 1.63, 1.62, 11.26, 3.77, 2.60, 15.38, 5.18],
            'Recovered / 100 Cases': [30.90, 75.58, 66.57, 72.65, 69.05, 71.38, 69.01, 91.67, 0.00, 87.05],
            '1 week % increase': [17.0, 21.5, 15.2, 8.7, 12.3, 25.4, 18.9, 5.2, 11.8, 14.6]
        }
        df = pd.DataFrame(sample_data)
        st.warning("⚠️ CSV file not found. Using sample data. Please upload your 'country_wise_latest.csv' file.")
        return df

# Main dashboard function
def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 COVID-19 Global Data Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info"><h2>🔧 Dashboard Controls</h2></div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your COVID-19 CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Data uploaded successfully!")

    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")

    # WHO Region filter
    if 'WHO Region' in df.columns:
        regions = ['All'] + list(df['WHO Region'].unique())
        selected_region = st.sidebar.selectbox("Select WHO Region", regions)

        if selected_region != 'All':
            df_filtered = df[df['WHO Region'] == selected_region]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
        selected_region = 'All'

    # Top N countries selector
    top_n = st.sidebar.slider("Number of top countries to display", 5, 20, 10)

    # Metric selector
    metric_options = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    available_metrics = [col for col in metric_options if col in df.columns]
    selected_metric = st.sidebar.selectbox("Select metric for analysis", available_metrics)

    # Analysis type
    analysis_type = st.sidebar.radio(
        "Choose analysis type",
        ["Overview", "Detailed Country Analysis", "Regional Comparison", "Misleading vs Correct Visualizations"]
    )

    # Main content based on analysis type
    if analysis_type == "Overview":
        show_overview(df_filtered, top_n, selected_metric)
    elif analysis_type == "Detailed Country Analysis":
        show_detailed_analysis(df_filtered, top_n)
    elif analysis_type == "Regional Comparison":
        show_regional_analysis(df)
    else:
        show_misleading_vs_correct(df)

def show_overview(df, top_n, metric):
    """Display overview dashboard"""

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_confirmed = df['Confirmed'].sum() if 'Confirmed' in df.columns else 0
        st.metric("🦠 Total Confirmed", f"{total_confirmed:,}")

    with col2:
        total_deaths = df['Deaths'].sum() if 'Deaths' in df.columns else 0
        st.metric("💀 Total Deaths", f"{total_deaths:,}")

    with col3:
        total_recovered = df['Recovered'].sum() if 'Recovered' in df.columns else 0
        st.metric("💚 Total Recovered", f"{total_recovered:,}")

    with col4:
        total_active = df['Active'].sum() if 'Active' in df.columns else 0
        st.metric("🔴 Total Active", f"{total_active:,}")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 Top {top_n} Countries by {metric}")
        top_countries = df.nlargest(top_n, metric)

        fig = px.bar(
            top_countries,
            x=metric,
            y='Country/Region',
            orientation='h',
            color=metric,
            color_continuous_scale='viridis',
            title=f"Top {top_n} Countries by {metric}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌍 Geographic Distribution")
        if 'WHO Region' in df.columns:
            region_data = df.groupby('WHO Region')[metric].sum().reset_index()

            fig = px.pie(
                region_data,
                values=metric,
                names='WHO Region',
                title=f"{metric} Distribution by WHO Region",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) >= 2:
        corr_matrix = df[available_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of COVID-19 Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_detailed_analysis(df, top_n):
    """Show detailed country analysis"""

    st.subheader("🔍 Detailed Country Analysis")

    # Country selector
    countries = st.multiselect(
        "Select countries for comparison",
        df['Country/Region'].tolist(),
        default=df['Country/Region'].head(5).tolist()
    )

    if countries:
        df_selected = df[df['Country/Region'].isin(countries)]

        # Metrics comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Cases Comparison")
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Confirmed',
                x=df_selected['Country/Region'],
                y=df_selected['Confirmed'],
                marker_color='blue'
            ))

            if 'Deaths' in df.columns:
                fig.add_trace(go.Bar(
                    name='Deaths',
                    x=df_selected['Country/Region'],
                    y=df_selected['Deaths'],
                    marker_color='red'
                ))

            if 'Recovered' in df.columns:
                fig.add_trace(go.Bar(
                    name='Recovered',
                    x=df_selected['Country/Region'],
                    y=df_selected['Recovered'],
                    marker_color='green'
                ))

            fig.update_layout(
                title="COVID-19 Cases by Country",
                xaxis_title="Country",
                yaxis_title="Number of Cases",
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Recovery & Death Rates")
            if 'Deaths / 100 Cases' in df.columns and 'Recovered / 100 Cases' in df.columns:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_selected['Country/Region'],
                    y=df_selected['Deaths / 100 Cases'],
                    mode='markers+lines',
                    name='Death Rate %',
                    marker=dict(size=10, color='red')
                ))

                fig.add_trace(go.Scatter(
                    x=df_selected['Country/Region'],
                    y=df_selected['Recovered / 100 Cases'],
                    mode='markers+lines',
                    name='Recovery Rate %',
                    marker=dict(size=10, color='green')
                ))

                fig.update_layout(
                    title="Death & Recovery Rates per 100 Cases",
                    xaxis_title="Country",
                    yaxis_title="Rate (%)",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("📋 Selected Countries Data")
        st.dataframe(df_selected)

def show_regional_analysis(df):
    """Show regional comparison analysis"""

    st.subheader("🌍 Regional Analysis")

    if 'WHO Region' in df.columns:
        region_stats = df.groupby('WHO Region').agg({
            'Confirmed': ['sum', 'mean'],
            'Deaths': ['sum', 'mean'],
            'Recovered': ['sum', 'mean'] if 'Recovered' in df.columns else ['sum'],
            'Active': ['sum', 'mean'] if 'Active' in df.columns else ['sum']
        }).round(0)

        # Flatten column names
        region_stats.columns = [f'{col[0]}_{col[1]}' for col in region_stats.columns]
        region_stats = region_stats.reset_index()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Total Cases by Region")
            fig = px.bar(
                x=region_stats['WHO Region'],
                y=region_stats['Confirmed_sum'],
                color=region_stats['Confirmed_sum'],
                color_continuous_scale='Blues',
                title="Total Confirmed Cases by WHO Region"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Average Cases by Region")
            fig = px.bar(
                x=region_stats['WHO Region'],
                y=region_stats['Confirmed_mean'],
                color=region_stats['Confirmed_mean'],
                color_continuous_scale='Reds',
                title="Average Confirmed Cases by WHO Region"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Regional comparison table
        st.subheader("📋 Regional Statistics Summary")
        st.dataframe(region_stats)
    else:
        st.warning("WHO Region data not available for regional analysis")

def show_misleading_vs_correct(df):
    """Show misleading vs correct visualizations"""

    st.subheader("⚠️ Misleading vs Correct Visualizations")
    st.markdown("This section demonstrates how the same data can be presented in misleading and correct ways.")

    if 'Deaths / 100 Cases' in df.columns:
        top_countries = df.nlargest(10, 'Deaths / 100 Cases')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("❌ Misleading Visualization")
            st.markdown("**Problem:** Y-axis doesn't start from zero, exaggerating differences")

            fig = px.bar(
                top_countries,
                x='Country/Region',
                y='Deaths / 100 Cases',
                color='Deaths / 100 Cases',
                color_continuous_scale='Reds',
                title="COVID-19 Deaths per 100 Cases (Misleading)"
            )
            # Set y-axis to start from minimum value (misleading)
            fig.update_yaxes(range=[top_countries['Deaths / 100 Cases'].min() * 0.9, 
                                  top_countries['Deaths / 100 Cases'].max() * 1.1])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("✅ Correct Visualization")
            st.markdown("**Solution:** Y-axis starts from zero, showing true proportions")

            fig = px.bar(
                top_countries,
                x='Country/Region',
                y='Deaths / 100 Cases',
                color='Deaths / 100 Cases',
                color_continuous_scale='Blues',
                title="COVID-19 Deaths per 100 Cases (Accurate)"
            )
            # Set y-axis to start from zero (correct)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        # Explanation
        st.markdown("""
        ### 📚 Key Learning Points:
        - **Misleading Chart:** The y-axis starts from the minimum value, making small differences appear large
        - **Correct Chart:** The y-axis starts from zero, showing the true proportion and context
        - **Best Practice:** Always consider the full scale when comparing quantities
        """)

# Sidebar info
def show_sidebar_info():
    """Show information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About This Dashboard")
    st.sidebar.markdown("""
    This dashboard provides interactive visualizations of COVID-19 data with:
    - 📊 Multiple chart types and filters
    - 🌍 Regional analysis capabilities  
    - 📈 Country comparison tools
    - ⚠️ Examples of misleading vs correct visualizations

    **Data Source:** Country-wise COVID-19 latest data
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Dashboard Features")
    st.sidebar.markdown("""
    - **Interactive Filters:** Filter by region, select metrics
    - **Multiple Views:** Overview, detailed analysis, regional comparison
    - **Data Upload:** Upload your own CSV files
    - **Responsive Design:** Works on different screen sizes
    - **Educational Content:** Learn about data visualization best practices
    """)

if __name__ == "__main__":
    show_sidebar_info()
    main()
