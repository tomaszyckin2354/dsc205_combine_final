import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 

import seaborn as sns

# Load data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

# Streamlit UI
st.title("NFL Combine: 40-Yard Dash vs. Draft Pick")
st.subheader("How Does Speed Influence Draft Pick by Position Group?")

# Filter options
years = sorted(df['Year'].unique())
positions = df['Pos_group'].dropna().unique()

selected_years = st.multiselect("Select Year(s):", options=years, default=years)
selected_positions = st.multiselect("Select Position Group(s):", options=positions, default=positions)

# Apply filters
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Pos_group'].isin(selected_positions))]
filtered_df = filtered_df.dropna(subset=['40yd', 'Pick', 'Pos_group'])

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_df, x='Pick', y='40yd', hue='Pos_group', palette='tab10')
plt.xlabel("Draft Pick")
plt.ylabel("40-Yard Dash Time (s)")
plt.title("40-Yard Dash vs. Draft Pick by Position Group")
plt.gca().invert_xaxis()  # Lower pick number is better
st.pyplot(plt)
