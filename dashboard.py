
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="COVID-19 Analytics Dashboard", 
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
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess COVID-19 data"""
    df = pd.read_csv("country_wise_latest.csv")

    # Add calculated fields for better analysis
    df['Mortality_Rate'] = (df['Deaths'] / df['Confirmed'] * 100).round(2)
    df['Recovery_Rate'] = (df['Recovered'] / df['Confirmed'] * 100).round(2)
    df['Active_Rate'] = (df['Active'] / df['Confirmed'] * 100).round(2)
    df['Deaths_per_100_Cases'] = df['Deaths per 100 Cases'].fillna(0)
    df['Recovered_per_100_Cases'] = df['Recovered per 100 Cases'].fillna(0)

    return df

def create_insights_section():
    """Create insights section with key findings"""
    st.markdown('<div class="section-header">🔍 Key Insights & Story</div>', unsafe_allow_html=True)

    insights = [
        "**Global Impact**: The COVID-19 pandemic has affected every corner of the world, with significant variations in case counts, mortality rates, and recovery patterns across different countries and regions.",

        "**Mortality Patterns**: Countries with higher case counts don't necessarily have proportionally higher mortality rates, suggesting differences in healthcare systems, demographics, and policy responses.",

        "**Recovery Success**: Recovery rates vary significantly across nations, reflecting differences in medical infrastructure, treatment protocols, and public health measures.",

        "**Data Correlations**: Strong correlations exist between confirmed cases and deaths, but the relationship varies by country, indicating different pandemic trajectories and response effectiveness."
    ]

    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

def create_advanced_visualizations(df, filtered_df, countries):
    """Create advanced visualizations for deeper analysis"""

    st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)

    # Box Plot Analysis
    st.markdown("### 📦 Distribution Analysis with Box Plots")
    st.markdown("**Understanding Data Distribution**: Box plots reveal the spread, outliers, and quartiles of COVID-19 metrics, helping identify countries with unusual patterns.")

    col1, col2 = st.columns(2)

    with col1:
        fig_box1, ax_box1 = plt.subplots(figsize=(8, 6))

        # Prepare data for box plot
        box_data = []
        labels = []
        for country in countries:
            country_data = df[df['Country/Region'] == country]
            if not country_data.empty:
                box_data.append([
                    country_data['Mortality_Rate'].values[0],
                    country_data['Recovery_Rate'].values[0],
                    country_data['Active_Rate'].values[0]
                ])
                labels.append(country)

        if box_data:
            box_data = np.array(box_data).T
            bp = ax_box1.boxplot(box_data, labels=['Mortality', 'Recovery', 'Active'], patch_artist=True)
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax_box1.set_title('Rate Distribution Across Selected Countries')
            ax_box1.set_ylabel('Percentage (%)')
            plt.xticks(rotation=45)

        st.pyplot(fig_box1)

    with col2:
        fig_box2, ax_box2 = plt.subplots(figsize=(8, 6))

        # Box plot for global top 20 countries by confirmed cases
        top_20 = df.nlargest(20, 'Confirmed')
        metrics_for_box = ['Mortality_Rate', 'Recovery_Rate']
        box_data_global = [top_20[metric].values for metric in metrics_for_box]

        bp2 = ax_box2.boxplot(box_data_global, labels=['Mortality Rate', 'Recovery Rate'], patch_artist=True)
        colors2 = ['#ffcc99', '#99ccff']
        for patch, color in zip(bp2['boxes'], colors2):
            patch.set_facecolor(color)
        ax_box2.set_title('Global Distribution (Top 20 Countries)')
        ax_box2.set_ylabel('Percentage (%)')

        st.pyplot(fig_box2)

    # Pair Plot Analysis
    st.markdown("### 🔗 Relationship Analysis with Pair Plot")
    st.markdown("**Exploring Correlations**: This pair plot matrix shows relationships between different COVID-19 metrics, revealing patterns and correlations that might not be immediately obvious.")

    # Create pair plot data
    pair_plot_data = filtered_df[['Confirmed', 'Deaths', 'Recovered', 'Active', 'Mortality_Rate', 'Recovery_Rate']].copy()

    fig_pair, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig_pair.suptitle('COVID-19 Metrics Relationship Matrix', fontsize=16, y=0.95)

    metrics = ['Confirmed', 'Deaths', 'Recovered']
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            ax = axes[i, j]
            if i == j:
                # Diagonal - histogram
                ax.hist(pair_plot_data[metric1], bins=10, alpha=0.7, color=plt.cm.Set3(i))
                ax.set_title(f'{metric1} Distribution')
            else:
                # Off-diagonal - scatter plot
                scatter = ax.scatter(pair_plot_data[metric2], pair_plot_data[metric1], 
                                   alpha=0.7, c=pair_plot_data['Mortality_Rate'], 
                                   cmap='viridis', s=50)
                ax.set_xlabel(metric2)
                ax.set_ylabel(metric1)

                # Add country labels for interesting points
                for idx, country in enumerate(filtered_df['Country/Region']):
                    if idx < 5:  # Label top 5 countries
                        ax.annotate(country, 
                                  (pair_plot_data.iloc[idx][metric2], pair_plot_data.iloc[idx][metric1]),
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=8, alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig_pair)

    # Enhanced Heatmap Analysis
    st.markdown("### 🌡️ Comprehensive Correlation Heatmap")
    st.markdown("**Deep Correlation Analysis**: This enhanced heatmap includes calculated rates and provides a comprehensive view of how different metrics relate to each other.")

    col1, col2 = st.columns(2)

    with col1:
        # Correlation heatmap for selected countries
        fig_heat1, ax_heat1 = plt.subplots(figsize=(10, 8))

        correlation_metrics = ['Confirmed', 'Deaths', 'Recovered', 'Active', 
                             'Mortality_Rate', 'Recovery_Rate', 'Deaths per 100 Cases']

        corr_matrix = filtered_df[correlation_metrics].corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax_heat1)
        ax_heat1.set_title('Selected Countries - Correlation Matrix')

        st.pyplot(fig_heat1)

    with col2:
        # Global correlation heatmap (top 30 countries)
        fig_heat2, ax_heat2 = plt.subplots(figsize=(10, 8))

        top_30_global = df.nlargest(30, 'Confirmed')
        corr_matrix_global = top_30_global[correlation_metrics].corr()

        sns.heatmap(corr_matrix_global, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax_heat2)
        ax_heat2.set_title('Global Top 30 - Correlation Matrix')

        st.pyplot(fig_heat2)

    # Violin Plot for Distribution Analysis
    st.markdown("### 🎻 Distribution Shape Analysis")
    st.markdown("**Understanding Data Shapes**: Violin plots show not just the distribution but also the density of data at different values, providing insights into the underlying patterns.")

    fig_violin, axes_violin = plt.subplots(2, 2, figsize=(15, 10))
    fig_violin.suptitle('COVID-19 Metrics Distribution Analysis', fontsize=16)

    violin_metrics = ['Mortality_Rate', 'Recovery_Rate', 'Deaths per 100 Cases', 'Recovered per 100 Cases']

    for i, metric in enumerate(violin_metrics):
        ax = axes_violin[i//2, i%2]

        # Prepare data for violin plot
        if len(countries) > 1:
            violin_data = []
            violin_labels = []
            for country in countries:
                country_data = df[df['Country/Region'] == country]
                if not country_data.empty and not pd.isna(country_data[metric].values[0]):
                    violin_data.append(country_data[metric].values[0])
                    violin_labels.append(country)

            if len(violin_data) > 1:
                ax.violinplot([violin_data], positions=[1])
                ax.set_xticks([1])
                ax.set_xticklabels(['Selected Countries'])
                ax.set_ylabel(metric.replace('_', ' '))
                ax.set_title(f'{metric.replace("_", " ")} Distribution')

                # Add individual points
                ax.scatter([1] * len(violin_data), violin_data, alpha=0.7, s=50, color='red')

                # Add country labels
                for j, (val, label) in enumerate(zip(violin_data, violin_labels)):
                    ax.annotate(label, (1, val), xytext=(10, 0), 
                              textcoords='offset points', fontsize=8, alpha=0.8)

    plt.tight_layout()
    st.pyplot(fig_violin)

def create_comparative_analysis(df, filtered_df, countries):
    """Create comparative analysis section"""
    st.markdown('<div class="section-header">⚖️ Comparative Analysis</div>', unsafe_allow_html=True)

    if len(countries) > 1:
        st.markdown("### 📊 Country Performance Comparison")
        st.markdown("**Benchmarking Performance**: Compare selected countries across multiple dimensions to understand relative performance and outcomes.")

        # Create comparison metrics
        comparison_metrics = []
        for country in countries:
            country_data = df[df['Country/Region'] == country]
            if not country_data.empty:
                metrics_dict = {
                    'Country': country,
                    'Confirmed Cases': f"{country_data['Confirmed'].values[0]:,}",
                    'Deaths': f"{country_data['Deaths'].values[0]:,}",
                    'Recovered': f"{country_data['Recovered'].values[0]:,}",
                    'Mortality Rate (%)': f"{country_data['Mortality_Rate'].values[0]:.2f}",
                    'Recovery Rate (%)': f"{country_data['Recovery_Rate'].values[0]:.2f}",
                    'Deaths per 100 Cases': f"{country_data['Deaths per 100 Cases'].values[0]:.2f}"
                }
                comparison_metrics.append(metrics_dict)

        comparison_df = pd.DataFrame(comparison_metrics)
        st.table(comparison_df)

        # Radar chart for multi-dimensional comparison
        st.markdown("### 🎯 Multi-Dimensional Performance Radar")
        st.markdown("**Holistic View**: This radar chart provides a comprehensive view of how countries perform across multiple metrics simultaneously.")

        fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Prepare radar chart data
        radar_metrics = ['Mortality_Rate', 'Recovery_Rate', 'Active_Rate']
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))

        for i, country in enumerate(countries):
            country_data = df[df['Country/Region'] == country]
            if not country_data.empty:
                values = [country_data[metric].values[0] for metric in radar_metrics]
                values += values[:1]  # Complete the circle

                ax_radar.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[i])
                ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels([metric.replace('_', ' ') for metric in radar_metrics])
        ax_radar.set_title('Country Performance Radar Chart', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        st.pyplot(fig_radar)

# Main Application
def main():
    # Load data
    df = load_data()

    # Header
    st.markdown('<div class="main-header">🌍 COVID-19 Advanced Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive Data Analysis and Storytelling Platform**")
    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("🎛️ Analysis Controls")
    st.sidebar.markdown("Select parameters to customize your analysis")

    countries = st.sidebar.multiselect(
        "Select Countries for Analysis",
        options=sorted(df['Country/Region'].unique()),
        default=["India", "US", "China", "Italy", "Spain"],
        help="Choose countries to compare and analyze"
    )

    analysis_type = st.sidebar.selectbox(
        "Analysis Focus",
        ["Complete Analysis", "Basic Overview", "Advanced Analytics Only", "Comparative Analysis Only"],
        help="Choose the depth of analysis"
    )

    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)

    # Filter data
    if countries:
        filtered_df = df[df['Country/Region'].isin(countries)]
    else:
        filtered_df = df.head(10)  # Show top 10 if no countries selected
        st.warning("No countries selected. Showing top 10 countries by confirmed cases.")
        countries = df.nlargest(10, 'Confirmed')['Country/Region'].tolist()

    # Key insights section
    create_insights_section()

    # Global metrics
    if analysis_type in ["Complete Analysis", "Basic Overview"]:
        st.markdown('<div class="section-header">🌐 Global Impact Overview</div>', unsafe_allow_html=True)
        st.markdown("**The Numbers Tell the Story**: These global metrics represent the cumulative impact of COVID-19 worldwide, highlighting the scale and severity of the pandemic.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_confirmed = df['Confirmed'].sum()
            st.metric(
                label="🦠 Total Confirmed Cases", 
                value=f"{total_confirmed:,}",
                help="Total number of confirmed COVID-19 cases worldwide"
            )

        with col2:
            total_deaths = df['Deaths'].sum()
            global_mortality = (total_deaths / total_confirmed * 100)
            st.metric(
                label="💀 Total Deaths", 
                value=f"{total_deaths:,}",
                delta=f"Mortality: {global_mortality:.2f}%",
                help="Total deaths and global mortality rate"
            )

        with col3:
            total_recovered = df['Recovered'].sum()
            global_recovery = (total_recovered / total_confirmed * 100)
            st.metric(
                label="💚 Total Recovered", 
                value=f"{total_recovered:,}",
                delta=f"Recovery: {global_recovery:.1f}%",
                help="Total recovered cases and global recovery rate"
            )

        with col4:
            total_active = df['Active'].sum()
            st.metric(
                label="🔴 Total Active Cases", 
                value=f"{total_active:,}",
                help="Currently active COVID-19 cases worldwide"
            )

    # Raw data section
    if show_raw_data:
        st.markdown('<div class="section-header">📋 Data Explorer</div>', unsafe_allow_html=True)
        st.markdown("**Raw Data Insights**: Explore the underlying data that powers this analysis.")
        st.dataframe(
            filtered_df.style.format({
                'Confirmed': '{:,}',
                'Deaths': '{:,}',
                'Recovered': '{:,}',
                'Active': '{:,}',
                'Mortality_Rate': '{:.2f}%',
                'Recovery_Rate': '{:.2f}%'
            }),
            use_container_width=True
        )

    # Basic visualizations
    if analysis_type in ["Complete Analysis", "Basic Overview"]:
        st.markdown('<div class="section-header">📊 Core Visualizations</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📈 Confirmed Cases by Country**")
            st.markdown("*This chart shows the total number of confirmed cases, providing a clear picture of the pandemic's reach in each selected country.*")

            fig1, ax1 = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(
                x="Confirmed",
                y="Country/Region",
                data=filtered_df.sort_values("Confirmed", ascending=False),
                palette="viridis",
                ax=ax1
            )
            ax1.set_title("Confirmed Cases by Country", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Number of Confirmed Cases")

            # Add value labels on bars
            for i, bar in enumerate(bars.patches):
                width = bar.get_width()
                ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{int(width):,}', ha='left', va='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.markdown("**🥧 Mortality Distribution**")
            st.markdown("*This pie chart illustrates the proportional distribution of deaths among selected countries, highlighting the relative impact.*")

            fig2, ax2 = plt.subplots(figsize=(8, 8))
            colors = plt.cm.Set3(np.arange(len(filtered_df)))
            wedges, texts, autotexts = ax2.pie(
                filtered_df["Deaths"],
                labels=filtered_df["Country/Region"],
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*filtered_df["Deaths"].sum()):,})',
                colors=colors,
                startangle=90
            )
            ax2.set_title("Deaths Distribution Among Selected Countries", fontsize=14, fontweight='bold')

            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            st.pyplot(fig2)

    # Advanced analytics
    if analysis_type in ["Complete Analysis", "Advanced Analytics Only"]:
        create_advanced_visualizations(df, filtered_df, countries)

    # Comparative analysis
    if analysis_type in ["Complete Analysis", "Comparative Analysis Only"] and len(countries) > 1:
        create_comparative_analysis(df, filtered_df, countries)

    # Footer insights
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Key Takeaways</div>', unsafe_allow_html=True)

    takeaways = [
        f"**Scale of Impact**: Among selected countries, {filtered_df.loc[filtered_df['Confirmed'].idxmax(), 'Country/Region']} has the highest confirmed cases with {filtered_df['Confirmed'].max():,} cases.",
        f"**Mortality Insights**: The country with the highest mortality rate among selections is {filtered_df.loc[filtered_df['Mortality_Rate'].idxmax(), 'Country/Region']} at {filtered_df['Mortality_Rate'].max():.2f}%.",
        f"**Recovery Success**: {filtered_df.loc[filtered_df['Recovery_Rate'].idxmax(), 'Country/Region']} shows the highest recovery rate at {filtered_df['Recovery_Rate'].max():.1f}%.",
        "**Data Patterns**: The visualizations reveal distinct patterns in how different countries experienced and managed the pandemic, with varying outcomes in mortality and recovery rates."
    ]

    for takeaway in takeaways:
        st.markdown(f'<div class="insight-box">{takeaway}</div>', unsafe_allow_html=True)

    # Technical notes
    with st.expander("📝 Technical Notes & Data Sources"):
        st.markdown("""
        **Data Processing Notes**:
        - Mortality Rate = (Deaths / Confirmed Cases) × 100
        - Recovery Rate = (Recovered / Confirmed Cases) × 100  
        - Active Rate = (Active Cases / Confirmed Cases) × 100

        **Visualization Techniques**:
        - Box plots show quartiles, medians, and outliers
        - Pair plots reveal correlations between variables
        - Heatmaps display correlation strength and direction
        - Violin plots show distribution density
        - Radar charts enable multi-dimensional comparison

        **Data Quality**: Please note that COVID-19 data reporting standards vary by country and may affect comparability.
        """)

if __name__ == "__main__":
    main()
