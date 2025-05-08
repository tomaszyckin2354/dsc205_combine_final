import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

st.title("NFL Combine: Metric vs. Draft Pick with Regression Lines")
# Group by Year and Pos_group
avg_pick = df.groupby(['Year', 'Pos_group'])['Pick'].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=avg_pick, x='Year', y='Pick', hue='Pos_group', marker='o', ax=ax)
ax.set_title("Average Draft Pick by Position Group Over Time")
ax.set_ylabel("Average Draft Pick")
ax.set_xlabel("Year")
ax.invert_yaxis()
ax.legend(title="Position Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

st.pyplot(fig)

st.write("Explore how combine metrics influence draft position — with trendlines for each position group.")

# Metric options
metric_options = {
    "40-Yard Dash": "40yd",
    "Vertical Jump": "Vertical",
    "Broad Jump": "Broad Jump",
    "3-Cone Drill": "3Cone",
    "Shuttle Run": "Shuttle",
    "Bench Press Reps": "Bench"
}
selected_label = st.selectbox("Choose Y-axis Combine Metric:", list(metric_options.keys()))
selected_column = metric_options[selected_label]

# Filters
years = sorted(df['Year'].dropna().unique())
positions = sorted(df['Pos_group'].dropna().unique())

selected_years = st.multiselect("Select Year(s):", options=years, default=years)
selected_positions = st.multiselect("Select Position Group(s):", options=positions, default=positions)

# Filtered data
filtered_df = df[
    (df['Year'].isin(selected_years)) &
    (df['Pos_group'].isin(selected_positions))
].dropna(subset=['Pick', selected_column, 'Pos_group'])

# Plot with regression lines
st.write("Each line shows the trend between draft pick and performance for a given position group.")
fig = sns.lmplot(
    data=filtered_df,
    x='Pick',
    y=selected_column,
    hue='Pos_group',
    height=6,
    aspect=1.6,
    scatter_kws={'alpha': 0.6, 's': 50},
    line_kws={'linewidth': 2}
)

plt.gca().invert_xaxis()  # Lower pick = better draft slot
plt.xlabel("Draft Pick")
plt.ylabel(selected_label)
plt.title(f"{selected_label} vs. Draft Pick with Trendlines by Position Group")

st.pyplot(fig)

