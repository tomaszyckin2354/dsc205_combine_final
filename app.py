import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

# Page header
st.title("NFL Combine Metric vs. Draft Pick")
st.subheader("Explore how athletic performance influences draft selection by position group")

# Filter options
years = sorted(df['Year'].unique())
positions = df['Pos_group'].dropna().unique()

selected_years = st.multiselect("Select Year(s):", options=years, default=years)
selected_positions = st.multiselect("Select Position Group(s):", options=positions, default=positions)

# Select y-axis variable
metric_options = {
    "40-Yard Dash": "40yd",
    "Vertical Jump": "Vertical",
    "Broad Jump": "Broad Jump",
    "3-Cone Drill": "3Cone",
    "Shuttle Run": "Shuttle",
    "Bench Press Reps": "Bench"
}
selected_metric_label = st.selectbox("Select Combine Metric (Y-axis):", options=list(metric_options.keys()))
selected_metric = metric_options[selected_metric_label]

# Filter data
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Pos_group'].isin(selected_positions))]
filtered_df = filtered_df.dropna(subset=['Pick', selected_metric, 'Pos_group'])

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='Pick', y=selected_metric, hue='Pos_group', palette='tab10')
plt.xlabel("Draft Pick")
plt.ylabel(selected_metric_label)
plt.title(f"{selected_metric_label} vs. Draft Pick by Position Group")
plt.gca().invert_xaxis()  # Pick 1 = better
st.pyplot(plt)
