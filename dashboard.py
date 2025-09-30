import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("country_wise_latest.csv")
    return df

df = load_data()

st.title("🌍 COVID-19 Dashboard")
st.markdown("Data Source: Country wise latest COVID dataset")

st.sidebar.header("Filters")
countries = st.sidebar.multiselect(
    "Select Countries",
    options=df['Country/Region'].unique(),
    default=["India", "US", "Brazil"]
)

if countries:
    filtered_df = df[df['Country/Region'].isin(countries)]
else:
    filtered_df = df.copy()

st.subheader("🌐 Key Global Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Confirmed", f"{df['Confirmed'].sum():,}")
col2.metric("Total Deaths", f"{df['Deaths'].sum():,}")
col3.metric("Total Recovered", f"{df['Recovered'].sum():,}")
col4.metric("Total Active", f"{df['Active'].sum():,}")

st.subheader("📋 Selected Countries Data")
st.dataframe(filtered_df)

st.subheader("📊 Visualizations")

st.markdown("**Confirmed Cases by Country**")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(
    x="Confirmed",
    y="Country/Region",
    data=filtered_df.sort_values("Confirmed", ascending=False),
    ax=ax1
)
ax1.set_title("Confirmed Cases by Country")
st.pyplot(fig1)

st.markdown("**Deaths Share Among Selected Countries**")
fig2, ax2 = plt.subplots()
ax2.pie(
    filtered_df["Deaths"],
    labels=filtered_df["Country/Region"],
    autopct="%1.1f%%"
)
ax2.set_title("Deaths Share")
st.pyplot(fig2)

st.markdown("**Correlation Between Variables**")
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(
    df[["Confirmed", "Deaths", "Recovered", "Active"]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax3
)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)

st.markdown("**Country-wise Comparison (Confirmed, Deaths, Recovered, Active)**")
fig4, ax4 = plt.subplots(figsize=(10, 5))
for country in countries:
    cdata = df[df["Country/Region"] == country]
    ax4.plot(
        ["Confirmed", "Deaths", "Recovered", "Active"],
        [
            cdata["Confirmed"].values[0],
            cdata["Deaths"].values[0],
            cdata["Recovered"].values[0],
            cdata["Active"].values[0]
        ],
        marker="o",
        label=country
    )
ax4.legend()
ax4.set_title("Comparison Across Metrics")
st.pyplot(fig4)

