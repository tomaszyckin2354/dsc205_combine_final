import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

st.title("Combine Metric vs. Draft Pick")
st.write("Select a combine event to visualize how it influences draft position.")

# Define valid metrics for Y-axis
metric_options = {
    "40-Yard Dash": "40yd",
    "Vertical Jump": "Vertical",
    "Broad Jump": "Broad Jump",
    "3-Cone Drill": "3Cone",
    "Shuttle Run": "Shuttle",
    "Bench Press Reps": "Bench"
}

# Select metric
selected_label = st.selectbox("Choose Y-axis Combine Metric:", list(metric_options.keys()))
selected_column = metric_options[selected_label]

# Filter out rows with missing data for required columns
plot_df = df[['Pick', selected_column, 'Pos_group']].dropna()

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='Pick', y=selected_column, hue='Pos_group', palette='tab10', ax=ax)
ax.set_title(f"{selected_label} vs. Draft Pick by Position Group")
ax.set_xlabel("Draft Pick")
ax.set_ylabel(selected_label)
ax.invert_xaxis()  # Lower picks = higher priority

st.pyplot(fig)
