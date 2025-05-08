import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

st.title("NFL Combine: Metric vs. Draft Pick")
st.write("Use the dropdown and filters to explore how athletic performance influences draft selection by position group.")

# Select Combine Metric (Y-Axis)
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

# Filter by Year and Position Group
years = sorted(df['Year'].dropna().unique())
positions = sorted(df['Pos_group'].dropna().unique())

selected_years = st.multiselect("Select Year(s):", options=years, default=years)
selected_positions = st.multiselect("Select Position Group(s):", options=positions, default=positions)

# Filter the DataFrame
filtered_df = df[
    (df['Year'].isin(selected_years)) &
    (df['Pos_group'].isin(selected_positions))
].dropna(subset=['Pick', selected_column, 'Pos_group'])

# Scatterplot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='Pick', y=selected_column, hue='Pos_group', palette='tab10', ax=ax)
ax.set_title(f"{selected_label} vs. Draft Pick by Position Group")
ax.set_xlabel("Draft Pick")
ax.set_ylabel(selected_label)
ax.invert_xaxis()  # Lower pick = better draft position

st.pyplot(fig)
