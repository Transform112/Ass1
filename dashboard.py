
import streamlit as st
import pandas as pd
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
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load COVID-19 data from CSV file"""
    try:
        # Try to load the user's CSV file
        df = pd.read_csv('country_wise_latest.csv')
        return df, True
    except FileNotFoundError:
        # Create comprehensive sample data if file is not found
        sample_data = {
            'Country/Region': [
                'US', 'Brazil', 'India', 'Russia', 'South Africa', 'Mexico', 'Peru', 
                'Chile', 'United Kingdom', 'Iran', 'China', 'Italy', 'Turkey', 
                'Germany', 'Pakistan', 'Saudi Arabia', 'France', 'Bangladesh'
            ],
            'Confirmed': [
                4290259, 2442375, 1640000, 816000, 503000, 385036, 355000, 
                336000, 295000, 278000, 85000, 245000, 220000, 208000, 
                285000, 275000, 165000, 230000
            ],
            'Deaths': [
                148011, 87618, 35747, 13334, 8153, 43374, 13384, 
                8722, 45358, 14405, 4634, 35123, 5526, 9148, 
                6097, 2866, 30305, 3000
            ],
            'Recovered': [
                1325804, 1846641, 1091771, 592611, 347224, 275000, 245000, 
                308000, 0, 242000, 79716, 196016, 203423, 187700, 
                267420, 250000, 82166, 130500
            ],
            'Active': [
                2816444, 508116, 512482, 210055, 147623, 66662, 96616, 
                19278, 249642, 21595, 490, 13861, 11051, 11152, 
                11483, 22134, 52529, 96500
            ],
            'WHO Region': [
                'Americas', 'Americas', 'South-East Asia', 'Europe', 'Africa', 
                'Americas', 'Americas', 'Americas', 'Europe', 'Eastern Mediterranean',
                'Western Pacific', 'Europe', 'Europe', 'Europe', 
                'Eastern Mediterranean', 'Eastern Mediterranean', 'Europe', 'South-East Asia'
            ],
            'Deaths / 100 Cases': [
                3.45, 3.59, 2.18, 1.63, 1.62, 11.26, 3.77, 
                2.60, 15.38, 5.18, 5.45, 14.31, 2.51, 4.40, 
                2.14, 1.04, 18.37, 1.30
            ],
            'Recovered / 100 Cases': [
                30.90, 75.58, 66.57, 72.65, 69.05, 71.38, 69.01, 
                91.67, 0.00, 87.05, 93.75, 79.97, 92.47, 90.23, 
                93.89, 90.91, 49.80, 56.74
            ],
            '1 week % increase': [
                17.0, 21.5, 15.2, 8.7, 12.3, 25.4, 18.9, 
                5.2, 11.8, 14.6, 0.5, 8.9, 12.1, 6.7, 
                16.8, 9.2, 13.4, 19.3
            ]
        }
        df = pd.DataFrame(sample_data)
        return df, False

# Main dashboard function
def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 COVID-19 Global Data Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    df, is_real_data = load_data()

    if not is_real_data:
        st.warning("⚠️ Using sample data. Upload your 'country_wise_latest.csv' file for actual data analysis.")
    else:
        st.success("✅ Data loaded successfully from your CSV file!")

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-info"><h2>🔧 Dashboard Controls</h2></div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your COVID-19 CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Data uploaded successfully!")
            is_real_data = True
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")

    # Display basic info about the dataset
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"**Countries:** {len(df)}")
    st.sidebar.write(f"**Total Cases:** {df['Confirmed'].sum():,}")
    st.sidebar.write(f"**Data Source:** {'User Upload' if is_real_data else 'Sample Data'}")

    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")

    # WHO Region filter
    if 'WHO Region' in df.columns:
        regions = ['All'] + sorted(list(df['WHO Region'].unique()))
        selected_region = st.sidebar.selectbox("Select WHO Region", regions)

        if selected_region != 'All':
            df_filtered = df[df['WHO Region'] == selected_region]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
        selected_region = 'All'

    # Top N countries selector
    max_countries = min(len(df_filtered), 25)
    top_n = st.sidebar.slider("Number of top countries to display", 5, max_countries, min(10, max_countries))

    # Metric selector
    metric_options = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    available_metrics = [col for col in metric_options if col in df.columns]
    selected_metric = st.sidebar.selectbox("Select metric for analysis", available_metrics)

    # Analysis type
    analysis_type = st.sidebar.radio(
        "Choose analysis type",
        ["📊 Overview", "🔍 Country Comparison", "🌍 Regional Analysis", "📚 Data Education"]
    )

    # Display filtered data info
    if selected_region != 'All':
        st.sidebar.markdown(f"**Filtered Region:** {selected_region}")
        st.sidebar.write(f"**Countries in region:** {len(df_filtered)}")

    # Main content based on analysis type
    if analysis_type == "📊 Overview":
        show_overview(df_filtered, top_n, selected_metric)
    elif analysis_type == "🔍 Country Comparison":
        show_detailed_analysis(df_filtered, top_n)
    elif analysis_type == "🌍 Regional Analysis":
        show_regional_analysis(df)
    else:
        show_data_education(df)

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

    # Additional derived metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        mortality_rate = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0
        st.metric("📉 Global Mortality Rate", f"{mortality_rate:.2f}%")

    with col2:
        recovery_rate = (total_recovered / total_confirmed * 100) if total_confirmed > 0 else 0
        st.metric("📈 Global Recovery Rate", f"{recovery_rate:.2f}%")

    with col3:
        active_rate = (total_active / total_confirmed * 100) if total_confirmed > 0 else 0
        st.metric("🟡 Active Cases Rate", f"{active_rate:.2f}%")

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
            title=f"Top {top_n} Countries by {metric}",
            hover_data={
                'Confirmed': ':,',
                'Deaths': ':,',
                'Recovered': ':,' if 'Recovered' in df.columns else False
            }
        )
        fig.update_layout(height=500, showlegend=False)
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
                color_discrete_sequence=px.colors.qualitative.Set3,
                hover_data=[metric]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("WHO Region data not available for geographic distribution.")

    # Growth rate analysis if available
    if '1 week % increase' in df.columns:
        st.subheader("📈 Weekly Growth Rate Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fastest_growing = df.nlargest(10, '1 week % increase')
            fig = px.bar(
                fastest_growing,
                x='1 week % increase',
                y='Country/Region',
                orientation='h',
                color='1 week % increase',
                color_continuous_scale='Reds',
                title="Top 10 Countries by Weekly Growth Rate",
                labels={'1 week % increase': 'Weekly Growth (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Scatter plot of confirmed cases vs growth rate
            fig = px.scatter(
                df,
                x='Confirmed',
                y='1 week % increase',
                size='Deaths',
                color='WHO Region' if 'WHO Region' in df.columns else None,
                hover_name='Country/Region',
                title="Cases vs Weekly Growth Rate",
                log_x=True,
                labels={'1 week % increase': 'Weekly Growth (%)', 'Confirmed': 'Total Confirmed Cases'}
            )
            fig.update_layout(height=400)
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
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.info("""
        **Correlation Interpretation:**
        - Values close to 1: Strong positive correlation
        - Values close to -1: Strong negative correlation  
        - Values close to 0: No linear correlation
        """)

def show_detailed_analysis(df, top_n):
    """Show detailed country analysis"""

    st.subheader("🔍 Detailed Country Analysis")

    # Country selector with search
    default_countries = df.nlargest(5, 'Confirmed')['Country/Region'].tolist()
    countries = st.multiselect(
        "Select countries for comparison (you can search by typing)",
        df['Country/Region'].tolist(),
        default=default_countries,
        help="Start typing to search for countries"
    )

    if not countries:
        st.warning("Please select at least one country for analysis.")
        return

    df_selected = df[df['Country/Region'].isin(countries)]

    # Summary statistics
    st.subheader("📋 Selected Countries Summary")

    # Create summary table
    summary_cols = ['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']
    if 'WHO Region' in df.columns:
        summary_cols.insert(1, 'WHO Region')

    available_summary_cols = [col for col in summary_cols if col in df.columns]
    summary_df = df_selected[available_summary_cols].copy()

    # Add calculated columns
    if 'Deaths' in df.columns and 'Confirmed' in df.columns:
        summary_df['Mortality Rate (%)'] = (summary_df['Deaths'] / summary_df['Confirmed'] * 100).round(2)

    if 'Recovered' in df.columns and 'Confirmed' in df.columns:
        summary_df['Recovery Rate (%)'] = (summary_df['Recovered'] / summary_df['Confirmed'] * 100).round(2)

    st.dataframe(summary_df, use_container_width=True)

    # Metrics comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Cases Comparison")

        # Prepare data for grouped bar chart
        metrics = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        available_metrics = [m for m in metrics if m in df.columns]

        fig = go.Figure()
        colors = {'Confirmed': 'blue', 'Deaths': 'red', 'Recovered': 'green', 'Active': 'orange'}

        for metric in available_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_selected['Country/Region'],
                y=df_selected[metric],
                marker_color=colors.get(metric, 'gray'),
                text=df_selected[metric],
                texttemplate='%{text:,}',
                textposition='outside'
            ))

        fig.update_layout(
            title="COVID-19 Cases by Country",
            xaxis_title="Country",
            yaxis_title="Number of Cases",
            barmode='group',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Rates Comparison")
        if 'Deaths / 100 Cases' in df.columns or 'Recovered / 100 Cases' in df.columns:
            fig = go.Figure()

            if 'Deaths / 100 Cases' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df_selected['Country/Region'],
                    y=df_selected['Deaths / 100 Cases'],
                    mode='markers+lines',
                    name='Death Rate per 100 Cases',
                    marker=dict(size=12, color='red'),
                    line=dict(color='red', width=2)
                ))

            if 'Recovered / 100 Cases' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df_selected['Country/Region'],
                    y=df_selected['Recovered / 100 Cases'],
                    mode='markers+lines',
                    name='Recovery Rate per 100 Cases',
                    marker=dict(size=12, color='green'),
                    line=dict(color='green', width=2)
                ))

            fig.update_layout(
                title="Death & Recovery Rates per 100 Cases",
                xaxis_title="Country",
                yaxis_title="Rate (%)",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Rate data not available for comparison.")

    # Individual country deep dive
    if len(countries) == 1:
        st.subheader(f"🔬 Deep Dive: {countries[0]}")
        country_data = df_selected.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confirmed", f"{country_data['Confirmed']:,}")
        with col2:
            st.metric("Deaths", f"{country_data['Deaths']:,}")
        with col3:
            if 'Recovered' in df.columns:
                st.metric("Recovered", f"{country_data['Recovered']:,}")
        with col4:
            if 'Active' in df.columns:
                st.metric("Active", f"{country_data['Active']:,}")

        # Show all available metrics for the country
        st.subheader("📊 All Available Metrics")
        country_metrics = {}
        for col in df.columns:
            if col != 'Country/Region' and pd.api.types.is_numeric_dtype(df[col]):
                country_metrics[col] = country_data[col]

        metrics_df = pd.DataFrame(list(country_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)

def show_regional_analysis(df):
    """Show regional comparison analysis"""

    st.subheader("🌍 Regional Analysis")

    if 'WHO Region' not in df.columns:
        st.warning("WHO Region data not available for regional analysis")
        return

    # Regional statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    region_stats = df.groupby('WHO Region')[numeric_cols].agg(['sum', 'mean', 'std']).round(0)

    # Flatten column names
    region_stats.columns = [f'{col[0]}_{col[1]}' for col in region_stats.columns]
    region_stats = region_stats.reset_index()

    # Key metrics by region
    st.subheader("📊 Regional Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Total Cases by Region")
        if 'Confirmed_sum' in region_stats.columns:
            fig = px.bar(
                region_stats,
                x='WHO Region',
                y='Confirmed_sum',
                color='Confirmed_sum',
                color_continuous_scale='Blues',
                title="Total Confirmed Cases by WHO Region",
                text='Confirmed_sum'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Average Cases by Region")
        if 'Confirmed_mean' in region_stats.columns:
            fig = px.bar(
                region_stats,
                x='WHO Region',
                y='Confirmed_mean',
                color='Confirmed_mean',
                color_continuous_scale='Reds',
                title="Average Confirmed Cases by WHO Region",
                text='Confirmed_mean'
            )
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Regional comparison radar chart
    st.subheader("🎯 Regional Performance Radar")

    # Select key metrics for radar chart
    radar_metrics = ['Confirmed_sum', 'Deaths_sum', 'Recovered_sum'] if 'Recovered_sum' in region_stats.columns else ['Confirmed_sum', 'Deaths_sum']
    available_radar_metrics = [col for col in radar_metrics if col in region_stats.columns]

    if len(available_radar_metrics) >= 2:
        # Normalize the data for better radar visualization
        radar_df = region_stats[['WHO Region'] + available_radar_metrics].copy()

        for metric in available_radar_metrics:
            radar_df[f'{metric}_norm'] = (radar_df[metric] / radar_df[metric].max()) * 100

        fig = go.Figure()

        for _, row in radar_df.iterrows():
            values = [row[f'{metric}_norm'] for metric in available_radar_metrics]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the shape
                theta=[metric.replace('_sum', '').replace('_', ' ').title() for metric in available_radar_metrics] + [available_radar_metrics[0].replace('_sum', '').replace('_', ' ').title()],
                fill='toself',
                name=row['WHO Region']
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Regional Performance Comparison (Normalized to 100%)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Detailed regional statistics table
    st.subheader("📋 Detailed Regional Statistics")

    # Select most relevant columns for display
    display_cols = ['WHO Region']
    for base_col in ['Confirmed', 'Deaths', 'Recovered', 'Active']:
        for stat in ['sum', 'mean']:
            col_name = f'{base_col}_{stat}'
            if col_name in region_stats.columns:
                display_cols.append(col_name)

    display_stats = region_stats[display_cols]
    st.dataframe(display_stats, use_container_width=True)

    # Regional insights
    st.subheader("💡 Regional Insights")

    if 'Confirmed_sum' in region_stats.columns:
        most_affected_region = region_stats.loc[region_stats['Confirmed_sum'].idxmax(), 'WHO Region']
        least_affected_region = region_stats.loc[region_stats['Confirmed_sum'].idxmin(), 'WHO Region']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Most Affected Region:** {most_affected_region}")
        with col2:
            st.info(f"**Least Affected Region:** {least_affected_region}")
        with col3:
            total_regions = len(region_stats)
            st.info(f"**Total WHO Regions:** {total_regions}")

def show_data_education(df):
    """Show educational content about data visualization"""

    st.subheader("📚 Data Visualization Best Practices")

    # Educational content selector
    education_topic = st.selectbox(
        "Choose learning topic",
        ["Misleading vs Correct Charts", "Chart Type Selection", "Color Usage", "Data Interpretation"]
    )

    if education_topic == "Misleading vs Correct Charts":
        show_misleading_vs_correct(df)
    elif education_topic == "Chart Type Selection":
        show_chart_selection_guide(df)
    elif education_topic == "Color Usage":
        show_color_guide(df)
    else:
        show_interpretation_guide(df)

def show_misleading_vs_correct(df):
    """Show misleading vs correct visualizations"""

    st.subheader("⚠️ Misleading vs Correct Visualizations")
    st.markdown("Learn how the same data can be presented in misleading and correct ways.")

    if 'Deaths / 100 Cases' in df.columns:
        top_countries = df.nlargest(8, 'Deaths / 100 Cases')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ❌ Misleading Visualization")
            st.markdown("**Problems:** Y-axis doesn't start from zero, exaggerating differences")

            fig = px.bar(
                top_countries,
                x='Country/Region',
                y='Deaths / 100 Cases',
                color='Deaths / 100 Cases',
                color_continuous_scale='Reds',
                title="COVID-19 Deaths per 100 Cases (Misleading)"
            )
            # Set y-axis to start from minimum value (misleading)
            y_min = max(0, top_countries['Deaths / 100 Cases'].min() * 0.8)
            y_max = top_countries['Deaths / 100 Cases'].max() * 1.1
            fig.update_yaxes(range=[y_min, y_max])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Why this is misleading:**
            - Truncated Y-axis makes small differences look huge
            - Visual impact doesn't match actual data proportions
            - Can lead to wrong conclusions
            """)

        with col2:
            st.markdown("#### ✅ Correct Visualization")
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
            fig.update_yaxes(range=[0, max(100, top_countries['Deaths / 100 Cases'].max() * 1.1)])
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Why this is better:**
            - Y-axis starts from zero baseline
            - True proportions are visible
            - Honest representation of the data
            """)

        # Additional examples
        st.markdown("---")
        st.subheader("🔍 More Examples")

        tab1, tab2 = st.tabs(["Scale Manipulation", "Cherry-picking Data"])

        with tab1:
            st.markdown("#### Scale Manipulation Example")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**❌ Logarithmic Scale (Misleading for this context)**")
                fig = px.scatter(
                    df.head(10),
                    x='Confirmed',
                    y='Deaths',
                    title="Deaths vs Confirmed (Log Scale)",
                    log_x=True,
                    log_y=True
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Log scale compresses large differences, making all points appear similar.")

            with col2:
                st.markdown("**✅ Linear Scale (Appropriate)**")
                fig = px.scatter(
                    df.head(10),
                    x='Confirmed',
                    y='Deaths',
                    title="Deaths vs Confirmed (Linear Scale)"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Linear scale shows true relationships and differences.")

        with tab2:
            st.markdown("#### Data Selection Impact")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**❌ Cherry-picked: Only Worst Cases**")
                worst_cases = df.nlargest(5, 'Deaths / 100 Cases')
                fig = px.bar(worst_cases, x='Country/Region', y='Deaths / 100 Cases', 
                           title="Selected Countries (Worst Only)")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Showing only extreme cases creates bias.")

            with col2:
                st.markdown("**✅ Representative Sample**")
                random_sample = df.sample(n=5, random_state=42)
                fig = px.bar(random_sample, x='Country/Region', y='Deaths / 100 Cases',
                           title="Representative Sample")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Random or systematic sampling gives balanced view.")

def show_chart_selection_guide(df):
    """Show guide for selecting appropriate chart types"""

    st.subheader("📊 Choosing the Right Chart Type")

    # Interactive chart type comparison
    chart_purpose = st.selectbox(
        "What do you want to show?",
        ["Compare quantities", "Show distribution", "Display relationships", "Track changes over time", "Show parts of whole"]
    )

    sample_data = df.head(8)

    if chart_purpose == "Compare quantities":
        st.markdown("### Best for Comparing Quantities: Bar Charts")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Horizontal Bar Chart (Recommended)**")
            fig = px.bar(sample_data, x='Confirmed', y='Country/Region', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            st.success("Easy to read country names, natural left-to-right reading")

        with col2:
            st.markdown("**❌ Pie Chart (Not Recommended)**")
            fig = px.pie(sample_data, values='Confirmed', names='Country/Region')
            st.plotly_chart(fig, use_container_width=True)
            st.warning("Hard to compare similar values, cluttered labels")

    elif chart_purpose == "Show distribution":
        st.markdown("### Best for Showing Distribution: Histograms & Box Plots")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Histogram**")
            fig = px.histogram(df, x='Deaths / 100 Cases', nbins=15, title="Distribution of Death Rates")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**✅ Box Plot**")
            if 'WHO Region' in df.columns:
                fig = px.box(df, y='Deaths / 100 Cases', x='WHO Region', 
                           title="Death Rate Distribution by Region")
                st.plotly_chart(fig, use_container_width=True)

    elif chart_purpose == "Display relationships":
        st.markdown("### Best for Relationships: Scatter Plots")

        fig = px.scatter(df, x='Confirmed', y='Deaths', size='Active',
                        color='WHO Region' if 'WHO Region' in df.columns else None,
                        hover_name='Country/Region',
                        title="Relationship: Confirmed Cases vs Deaths")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Shows correlation, outliers, and patterns clearly")

    elif chart_purpose == "Show parts of whole":
        st.markdown("### Best for Parts of Whole: Pie Charts or Stacked Bars")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Pie Chart (Good for <7 categories)**")
            if 'WHO Region' in df.columns:
                region_data = df.groupby('WHO Region')['Confirmed'].sum().reset_index()
                fig = px.pie(region_data, values='Confirmed', names='WHO Region')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**✅ Stacked Bar (Good for many categories)**")
            if 'WHO Region' in df.columns:
                fig = px.bar(sample_data, x='Country/Region', y='Confirmed', color='WHO Region')
                st.plotly_chart(fig, use_container_width=True)

def show_color_guide(df):
    """Show guide for using colors effectively"""

    st.subheader("🎨 Effective Use of Color in Data Visualization")

    st.markdown("### Color Psychology in Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Good Color Choices:**")
        st.markdown("""
        - **Red**: Danger, high values, deaths
        - **Green**: Success, recovery, positive trends  
        - **Blue**: Neutral, total counts, professional
        - **Orange**: Warning, active cases
        - **Gray**: Inactive, background data
        """)

        # Example with good colors
        fig = go.Figure()
        metrics = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        colors = ['blue', 'red', 'green', 'orange']
        sample = df.head(5)

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=sample['Country/Region'],
                    y=sample[metric],
                    marker_color=colors[i]
                ))

        fig.update_layout(title="Good Color Usage", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**❌ Poor Color Choices:**")
        st.markdown("""
        - **Random bright colors**: Distracting
        - **Red for positive data**: Confusing
        - **Too many similar colors**: Hard to distinguish
        - **Neon/bright colors**: Eye strain
        - **Color only encoding**: Accessibility issues
        """)

        # Example with poor colors
        fig = go.Figure()
        poor_colors = ['lime', 'magenta', 'cyan', 'yellow']

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=sample['Country/Region'],
                    y=sample[metric],
                    marker_color=poor_colors[i]
                ))

        fig.update_layout(title="Poor Color Usage", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Accessibility Considerations")
    st.info("""
    **Color-blind friendly practices:**
    - Use patterns or shapes in addition to color
    - Avoid red-green combinations alone
    - Test with colorbrewer2.org palettes
    - Provide legend/labels
    - Use sufficient contrast ratios
    """)

def show_interpretation_guide(df):
    """Show guide for interpreting data correctly"""

    st.subheader("🧠 Data Interpretation Best Practices")

    st.markdown("### Common Interpretation Pitfalls")

    # Correlation vs Causation
    st.markdown("#### 1. Correlation vs Causation")

    if 'Confirmed' in df.columns and 'Deaths' in df.columns:
        fig = px.scatter(df, x='Confirmed', y='Deaths', 
                        hover_name='Country/Region',
                        title="Deaths vs Confirmed Cases")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**❌ Wrong Interpretation:**")
            st.error("'Higher confirmed cases cause more deaths'")
            st.markdown("This assumes causation from correlation.")

        with col2:
            st.markdown("**✅ Correct Interpretation:**")
            st.success("'Countries with more confirmed cases tend to have more deaths'")
            st.markdown("This acknowledges correlation without implying causation.")

    # Sample size considerations
    st.markdown("#### 2. Sample Size Matters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Small Sample (Top 3 Countries)**")
        small_sample = df.head(3)
        avg_mortality = (small_sample['Deaths'].sum() / small_sample['Confirmed'].sum() * 100)
        st.metric("Average Mortality Rate", f"{avg_mortality:.2f}%")
        st.warning("Small sample may not be representative")

    with col2:
        st.markdown("**Full Dataset**")
        full_mortality = (df['Deaths'].sum() / df['Confirmed'].sum() * 100)
        st.metric("Average Mortality Rate", f"{full_mortality:.2f}%")
        st.success("Larger sample is more reliable")

    # Context importance
    st.markdown("#### 3. Context is Crucial")

    st.info("""
    **Always consider:**
    - **Time period**: When was this data collected?
    - **Data source**: How reliable and complete is it?
    - **Methodology**: How were cases counted?
    - **Population**: Per-capita vs absolute numbers
    - **External factors**: Healthcare capacity, testing rates, demographics
    """)

    # Statistical significance
    st.markdown("#### 4. Statistical Significance")

    if 'WHO Region' in df.columns and len(df['WHO Region'].unique()) > 1:
        region_stats = df.groupby('WHO Region')['Deaths / 100 Cases'].agg(['mean', 'std', 'count']).round(2)
        st.subheader("Regional Death Rate Statistics")
        st.dataframe(region_stats)

        st.info("""
        **Key points:**
        - **Mean**: Average value for the region
        - **Std**: Standard deviation (variability)
        - **Count**: Number of countries in sample
        - **Interpretation**: Larger standard deviation = more variability within region
        """)

# Sidebar info
def show_sidebar_info():
    """Show information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About This Dashboard")
    st.sidebar.markdown("""
    This interactive dashboard provides comprehensive analysis of COVID-19 data with:

    **🔍 Analysis Features:**
    - Multiple visualization types
    - Regional and country comparisons  
    - Statistical correlation analysis
    - Educational content on data visualization

    **🎛️ Interactive Controls:**
    - Dynamic filtering by region
    - Customizable metric selection
    - File upload capability
    - Responsive chart updates
    """)

    st.sidebar.markdown("### 🚀 How to Use")
    st.sidebar.markdown("""
    1. **Upload Data**: Use the file uploader for your CSV
    2. **Select Filters**: Choose region and metrics
    3. **Explore Views**: Switch between analysis types
    4. **Learn**: Check the Data Education section
    """)

    st.sidebar.markdown("### 📊 Data Format")
    st.sidebar.markdown("""
    Expected CSV columns:
    - Country/Region
    - Confirmed, Deaths, Recovered, Active
    - WHO Region
    - Deaths / 100 Cases
    - Recovered / 100 Cases
    - 1 week % increase
    """)

if __name__ == "__main__":
    show_sidebar_info()
    main()
