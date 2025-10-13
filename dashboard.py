import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="COVID-19 Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and black font fix
st.markdown("""
<style>
    /* Fix: Set main text color to black (or a dark shade) */
    .stApp, .main, body {
        color: #000000; /* Black font color */
    }
    
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
        color: #000000; /* Ensure text inside the box is also black */
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
    """Load and preprocess COVID-19 data with column name checks/fixes"""
    try:
        df = pd.read_csv("country_wise_latest.csv")
    except FileNotFoundError:
        st.error("Error: 'country_wise_latest.csv' not found. Please ensure the data file is in the same directory.")
        st.stop()
        return pd.DataFrame() 

    # --- Start Column Fix for common KeyErrors ---
    column_rename_map = {
        'Deaths / 100 Cases': 'Deaths per 100 Cases',
        'Recovered / 100 Cases': 'Recovered per 100 Cases',
        'Deaths per 100 cases': 'Deaths per 100 Cases',
        'Recovered per 100 cases': 'Recovered per 100 Cases',
        'Deaths-per-100-Cases': 'Deaths per 100 Cases',
        'Recovered-per-100-Cases': 'Recovered per 100 Cases',
    }
    
    for old_name, new_name in column_rename_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    # --- End Column Fix ---

    # Add calculated fields for better analysis
    df['Mortality_Rate'] = (df['Deaths'] / df['Confirmed'] * 100).round(2).fillna(0)
    df['Recovery_Rate'] = (df['Recovered'] / df['Confirmed'] * 100).round(2).fillna(0)
    df['Active_Rate'] = (df['Active'] / df['Confirmed'] * 100).round(2).fillna(0)
        
    df.loc[df['Confirmed'] == 0, ['Mortality_Rate', 'Recovery_Rate', 'Active_Rate']] = 0

    if 'Deaths per 100 Cases' not in df.columns:
        df['Deaths per 100 Cases'] = df['Mortality_Rate'] 
    
    if 'Recovered per 100 Cases' not in df.columns:
        df['Recovered per 100 Cases'] = df['Recovery_Rate'] 

    df['Deaths per 100 Cases'] = df['Deaths per 100 Cases'].fillna(0)
    df['Recovered per 100 Cases'] = df['Recovered per 100 Cases'].fillna(0)

    return df

def create_insights_section():
    """Create insights section with key findings"""
    st.markdown('<div class="section-header">Key Insights & Story</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)

    # Box Plot Analysis
    st.markdown("### Rate Distribution Across Selected Countries")
    st.markdown("**Understanding Data Distribution**: This box plot compares the median, quartiles, and range of key rates (Mortality, Recovery, Active) *across all selected countries*.")

    col1, col2 = st.columns(2)

    rate_metrics = filtered_df[['Country/Region', 'Mortality_Rate', 'Recovery_Rate', 'Active_Rate']]
    rate_df_melted = rate_metrics.melt(id_vars='Country/Region', 
                                     var_name='Rate Type', 
                                     value_name='Rate Value (%)')

    with col1:
        fig_box1, ax_box1 = plt.subplots(figsize=(8, 6))

        sns.boxplot(x='Rate Type', y='Rate Value (%)', data=rate_df_melted, 
                    palette={'Mortality_Rate': '#ff9999', 'Recovery_Rate': '#66b3ff', 'Active_Rate': '#99ff99'}, 
                    ax=ax_box1)
        sns.swarmplot(x='Rate Type', y='Rate Value (%)', data=rate_df_melted, color='black', size=5, ax=ax_box1)
        
        ax_box1.set_xticklabels(['Mortality', 'Recovery', 'Active'], rotation=0)
        ax_box1.set_title('Rate Distribution Across Selected Countries (with Data Points)')
        ax_box1.set_xlabel('')
        ax_box1.set_ylabel('Percentage (%)')

        st.pyplot(fig_box1)

    with col2:
        fig_box2, ax_box2 = plt.subplots(figsize=(8, 6))

        top_20 = df.nlargest(20, 'Confirmed')
        top_20_melted = top_20[['Mortality_Rate', 'Recovery_Rate']].melt(var_name='Rate Type', value_name='Rate Value (%)')

        sns.boxplot(x='Rate Type', y='Rate Value (%)', data=top_20_melted, 
                    palette={'Mortality_Rate': '#ffcc99', 'Recovery_Rate': '#99ccff'}, 
                    ax=ax_box2)
        sns.swarmplot(x='Rate Type', y='Rate Value (%)', data=top_20_melted, color='black', size=5, ax=ax_box2)

        ax_box2.set_xticklabels(['Mortality Rate', 'Recovery Rate'], rotation=0)
        ax_box2.set_title('Global Distribution (Top 20 Countries)')
        ax_box2.set_xlabel('')
        ax_box2.set_ylabel('Percentage (%)')

        st.pyplot(fig_box2)

    # Pair Plot Analysis
    st.markdown("### Relationship Analysis with Pair Plot")
    st.markdown("**Exploring Correlations**: This matrix shows relationships between different COVID-19 metrics, revealing patterns and correlations in the selected countries.")

    pair_plot_data = filtered_df[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Mortality_Rate']].copy()
    pair_plot_data['log_Confirmed'] = np.log1p(pair_plot_data['Confirmed'])
    pair_plot_data['log_Deaths'] = np.log1p(pair_plot_data['Deaths'])
    pair_plot_data['log_Recovered'] = np.log1p(pair_plot_data['Recovered'])

    fig_pair = sns.pairplot(
        pair_plot_data, 
        vars=['log_Confirmed', 'log_Deaths', 'log_Recovered'], 
        hue='Mortality_Rate', 
        palette='viridis',
        plot_kws={'alpha': 0.7, 's': 80},
        diag_kws={'palette': 'Set3'}
    )
    
    fig_pair.fig.suptitle('Log-Scaled COVID-19 Metrics Relationship Matrix (Color by Mortality Rate)', fontsize=16, y=1.02)
    
    new_labels = ['log(Confirmed)', 'log(Deaths)', 'log(Recovered)']
    for i, ax in enumerate(fig_pair.axes.flat):
        ax.set_xlabel(new_labels[i % 3] if i >= 3 else "")
        ax.set_ylabel(new_labels[i // 3] if i % 3 == 0 else "")


    st.pyplot(fig_pair)

    # Enhanced Heatmap Analysis
    st.markdown("### Comprehensive Correlation Heatmap")
    st.markdown("**Deep Correlation Analysis**: This enhanced heatmap includes calculated rates and provides a comprehensive view of how different metrics relate to each other.")

    col1, col2 = st.columns(2)

    correlation_metrics = ['Confirmed', 'Deaths', 'Recovered', 'Active', 
                         'Mortality_Rate', 'Recovery_Rate', 'Deaths per 100 Cases']

    with col1:
        fig_heat1, ax_heat1 = plt.subplots(figsize=(10, 8))

        corr_matrix = filtered_df[correlation_metrics].corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt=".2f", ax=ax_heat1)
        ax_heat1.set_title('Selected Countries - Correlation Matrix')

        st.pyplot(fig_heat1)

    with col2:
        fig_heat2, ax_heat2 = plt.subplots(figsize=(10, 8))

        top_30_global = df.nlargest(30, 'Confirmed')
        corr_matrix_global = top_30_global[correlation_metrics].corr()

        sns.heatmap(corr_matrix_global, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt=".2f", ax=ax_heat2)
        ax_heat2.set_title('Global Top 30 - Correlation Matrix')

        st.pyplot(fig_heat2)

    # Individual Country Rate Comparison
    st.markdown("### Individual Country Rate Comparison")
    st.markdown("**Direct Comparison**: This visualization clearly plots the different rate metrics for each selected country, enabling direct comparison.")

    individual_rate_df = filtered_df[['Country/Region', 'Mortality_Rate', 'Recovery_Rate', 'Active_Rate']].melt(
        id_vars='Country/Region', 
        var_name='Metric', 
        value_name='Rate (%)'
    )

    fig_bar_rate, ax_bar_rate = plt.subplots(figsize=(12, 6))

    sns.barplot(
        x='Country/Region', 
        y='Rate (%)', 
        hue='Metric', 
        data=individual_rate_df, 
        palette={'Mortality_Rate': 'red', 'Recovery_Rate': 'green', 'Active_Rate': 'blue'},
        ax=ax_bar_rate
    )

    ax_bar_rate.set_title('Comparative Case Rate Metrics by Country', fontsize=14, fontweight='bold')
    ax_bar_rate.set_xlabel('Country/Region', fontsize=12)
    ax_bar_rate.set_ylabel('Rate (%)', fontsize=12)
    ax_bar_rate.tick_params(axis='x', rotation=45)
    ax_bar_rate.legend(title='Metric')

    plt.tight_layout()
    st.pyplot(fig_bar_rate)


def create_comparative_analysis(df, filtered_df, countries):
    """Create comparative analysis section"""
    st.markdown('<div class="section-header">Comparative Analysis</div>', unsafe_allow_html=True)

    if len(countries) > 0:
        st.markdown("### Country Performance Comparison Table")
        st.markdown("**Benchmarking Performance**: Compare selected countries across multiple dimensions to understand relative performance and outcomes.")

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
        st.markdown("### Multi-Dimensional Performance Radar")
        st.markdown("**Holistic View**: This radar chart provides a comprehensive view of how countries perform across multiple metrics simultaneously.")

        fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        radar_metrics = ['Mortality_Rate', 'Recovery_Rate', 'Active_Rate']
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1] 

        colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))

        radar_df = filtered_df[radar_metrics].copy()
        
        for metric in radar_metrics:
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            if max_val > min_val:
                radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)
            else:
                radar_df[metric] = 0.5 
        
        # Invert (higher is worse) for consistent interpretation (higher score = better performance)
        radar_df['Mortality_Rate'] = 1 - radar_df['Mortality_Rate']
        radar_df['Active_Rate'] = 1 - radar_df['Active_Rate']

        country_list = filtered_df['Country/Region'].tolist() 
        
        for i, country in enumerate(country_list):
            country_data = radar_df.iloc[i] 
            values = [country_data[metric] for metric in radar_metrics]
            values += values[:1] 

            ax_radar.plot(angles, values, 'o-', linewidth=2, label=country, color=colors[i])
            ax_radar.fill(angles, values, alpha=0.25, color=colors[i])

        ax_radar.set_xticks(angles[:-1])
        display_metrics = ['Recovery Rate (Normalized)', 'Mortality Rate (Inverted Normalized)', 'Active Rate (Inverted Normalized)']
        ax_radar.set_xticklabels(display_metrics, fontsize=9)
        ax_radar.set_title('Normalized Country Performance Radar Chart', pad=20)
        ax_radar.set_rlabel_position(0)
        ax_radar.set_yticks(np.linspace(0, 1, 6)) 
        ax_radar.set_yticklabels([f'{val:.1f}' for val in np.linspace(0, 1, 6)], color="grey", size=8)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        st.pyplot(fig_radar)

# Main Application
def main():
    # Load data
    df = load_data()

    # Header
    st.markdown('<div class="main-header">COVID-19 Advanced Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive Data Analysis and Storytelling Platform**")
    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("Analysis Controls")
    st.sidebar.markdown("Select parameters to customize your analysis")

    countries_options = sorted(df['Country/Region'].unique())
    default_countries = [c for c in ["India", "US", "China", "Italy", "Spain"] if c in countries_options]

    countries = st.sidebar.multiselect(
        "Select Countries for Analysis",
        options=countries_options,
        default=default_countries,
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
        filtered_df = df.nlargest(10, 'Confirmed').copy()
        countries = filtered_df['Country/Region'].tolist()
        st.warning(f"No countries selected. Showing top 10 countries by confirmed cases: {', '.join(countries[:5])}...")

    # Key insights section
    create_insights_section()

    # Global metrics
    if analysis_type in ["Complete Analysis", "Basic Overview"]:
        st.markdown('<div class="section-header">Global Impact Overview</div>', unsafe_allow_html=True)
        st.markdown("**The Numbers Tell the Story**: These global metrics represent the cumulative impact of COVID-19 worldwide, highlighting the scale and severity of the pandemic.")

        col1, col2, col3, col4 = st.columns(4)

        total_confirmed = df['Confirmed'].sum()
        total_deaths = df['Deaths'].sum()
        total_recovered = df['Recovered'].sum()
        total_active = df['Active'].sum()
        
        global_mortality = (total_deaths / total_confirmed * 100) if total_confirmed > 0 else 0
        global_recovery = (total_recovered / total_confirmed * 100) if total_confirmed > 0 else 0

        with col1:
            st.metric(label="Total Confirmed Cases", value=f"{total_confirmed:,}")

        with col2:
            st.metric(label="Total Deaths", value=f"{total_deaths:,}", delta=f"Mortality: {global_mortality:.2f}%")

        with col3:
            st.metric(label="Total Recovered", value=f"{total_recovered:,}", delta=f"Recovery: {global_recovery:.1f}%")

        with col4:
            st.metric(label="Total Active Cases", value=f"{total_active:,}")

    # Raw data section
    if show_raw_data:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
        st.markdown("**Raw Data Insights**: Explore the underlying data that powers this analysis.")
        st.dataframe(
            filtered_df.style.format({
                'Confirmed': '{:,.0f}',
                'Deaths': '{:,.0f}',
                'Recovered': '{:,.0f}',
                'Active': '{:,.0f}',
                'Mortality_Rate': '{:.2f}%',
                'Recovery_Rate': '{:.2f}%',
                'Deaths per 100 Cases': '{:.2f}',
                'Recovered per 100 Cases': '{:.2f}'
            }),
            use_container_width=True
        )

    # Basic visualizations
    if analysis_type in ["Complete Analysis", "Basic Overview"]:
        st.markdown('<div class="section-header">Core Visualizations</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Confirmed Cases by Country")
            st.markdown("*This chart shows the total number of confirmed cases, providing a clear picture of the pandemic's reach in each selected country.*")

            fig1, ax1 = plt.subplots(figsize=(10, 6))
            plot_data = filtered_df.sort_values("Confirmed", ascending=False)
            
            if not plot_data.empty:
                bars = sns.barplot(
                    x="Confirmed",
                    y="Country/Region",
                    data=plot_data,
                    palette="viridis",
                    ax=ax1
                )
                ax1.set_title("Confirmed Cases by Country", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Number of Confirmed Cases")
                
                for bar in bars.patches:
                    width = bar.get_width()
                    ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                             f'{int(width):,}', ha='left', va='center', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig1)
            else:
                st.warning("No data to display for Confirmed Cases by Country.")
                plt.close(fig1)

        with col2:
            st.markdown("### Mortality Distribution")
            st.markdown("*This pie chart illustrates the proportional distribution of deaths among selected countries, highlighting the relative impact.*")

            fig2, ax2 = plt.subplots(figsize=(8, 8))
            
            pie_data = filtered_df[filtered_df["Deaths"] > 0].copy()
            
            if not pie_data.empty:
                colors = plt.cm.Set3(np.arange(len(pie_data)))
                total_deaths_pie = pie_data["Deaths"].sum()
                
                wedges, texts, autotexts = ax2.pie(
                    pie_data["Deaths"],
                    labels=pie_data["Country/Region"],
                    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_deaths_pie):,})',
                    colors=colors,
                    startangle=90
                )
                ax2.set_title("Deaths Distribution Among Selected Countries", fontsize=14, fontweight='bold')

                for autotext in autotexts:
                    autotext.set_color('black') 
                    autotext.set_fontweight('bold')

                st.pyplot(fig2)
            else:
                st.warning("No deaths recorded in the selected countries to display in the pie chart.")
                plt.close(fig2)

    # Advanced analytics
    if analysis_type in ["Complete Analysis", "Advanced Analytics Only"]:
        create_advanced_visualizations(df, filtered_df, countries)

    # Comparative analysis
    if analysis_type in ["Complete Analysis", "Comparative Analysis Only"] and len(countries) > 0:
        create_comparative_analysis(df, filtered_df, countries)

    # Footer insights
    st.markdown("---")
    st.markdown('<div class="section-header">Key Takeaways</div>', unsafe_allow_html=True)

    if not filtered_df.empty:
        takeaways = [
            f"**Scale of Impact**: Among selected countries, **{filtered_df.loc[filtered_df['Confirmed'].idxmax(), 'Country/Region']}** has the highest confirmed cases with {filtered_df['Confirmed'].max():,} cases.",
            f"**Mortality Insights**: The country with the highest mortality rate among selections is **{filtered_df.loc[filtered_df['Mortality_Rate'].idxmax(), 'Country/Region']}** at {filtered_df['Mortality_Rate'].max():.2f}%.",
            f"**Recovery Success**: **{filtered_df.loc[filtered_df['Recovery_Rate'].idxmax(), 'Country/Region']}** shows the highest recovery rate at {filtered_df['Recovery_Rate'].max():.1f}%.",
            "**Data Patterns**: The visualizations reveal distinct patterns in how different countries experienced and managed the pandemic, with varying outcomes in mortality and recovery rates."
        ]

        for takeaway in takeaways:
            st.markdown(f'<div class="insight-box">{takeaway}</div>', unsafe_allow_html=True)
    else:
        st.warning("Cannot generate takeaways: Filtered data is empty.")

    # Technical notes
    with st.expander("Technical Notes & Data Sources"):
        st.markdown("""
        **Data Processing Notes**:
        - **Mortality Rate** = (Deaths / Confirmed Cases) × 100
        - **Recovery Rate** = (Recovered / Confirmed Cases) × 100  
        - **Active Rate** = (Active Cases / Confirmed Cases) × 100
        - Rates for 0 confirmed cases are set to 0.
        - **Note on Column Renaming:** The script automatically attempts to correct common column naming issues in the raw data (`Deaths / 100 Cases` or similar) to ensure the application runs smoothly.

        **Visualization Techniques**:
        - **Box plots** show quartiles, medians, and outliers of the rates across the selected countries.
        - **Pair plots** use a **logarithmic scale** for count data (Confirmed, Deaths, Recovered) to better reveal correlations and manage large numerical differences between countries.
        - **Heatmaps** display correlation strength and direction.
        - **Radar charts** provide a multi-dimensional comparison of normalized rates (Recovery is good, Mortality/Active are inverted for "Performance" score).

        **Data Quality**: Please note that COVID-19 data reporting standards vary by country and may affect comparability.
        """)

if __name__ == "__main__":
    main()