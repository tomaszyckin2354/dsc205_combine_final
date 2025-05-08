import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 

# Load data
df = pd.read_csv("combine_2009_2019_cleaned.csv")

st.title("NFL Combine: Metric vs. Draft Pick with Regression Lines")

# Get list of unique position groups
position_groups = sorted(df['Pos_group'].unique())

# Add a multiselect widget
selected_positions = st.multiselect(
    "Select Position Group(s) to View:",
    options=position_groups,
    default=position_groups[:1]  # Default to first one
)

# Filter data based on selection
filtered_df = df[df['Pos_group'].isin(selected_positions)]

# Group and calculate average pick
avg_pick = filtered_df.groupby(['Year', 'Pos_group'])['Pick'].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=avg_pick, x='Year', y='Pick', hue='Pos_group', marker='o', ax=ax)
ax.set_title("Average Draft Pick by Position Group Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Average Draft Pick")
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

#linear regression
combine_metrics = ['40yd', 'Vertical', 'Broad Jump', '3Cone', 'Shuttle', 'Bench']
df = df.dropna(subset=['Pick', 'Pos_group'] + combine_metrics)

# Position group selector
position_groups = sorted(df['Pos_group'].unique())
selected_position = st.selectbox("Select a Position Group:", position_groups)

# Filter for selected position group
group_df = df[df['Pos_group'] == selected_position]

# Define X and y
X = group_df[combine_metrics]
y = group_df['Pick']

# Add constant for statsmodels
X_sm = sm.add_constant(X)

# Run regression
model = sm.OLS(y, X_sm).fit()

# Show table of regression results
st.subheader(f"Linear Regression for {selected_position}")
st.write("**Target**: Draft Pick")
st.write("**Features**: Combine Metrics")


# Predict draft picks using the model
y_pred = model.predict(X_sm)

r_squared = model.rsquared
st.markdown(f"**R² (Coefficient of Determination):** {r_squared:.3f}")

# Plot: Actual vs Predicted
st.subheader("Actual vs. Predicted Draft Picks")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred)
ax.set_xlabel("Actual Draft Pick")
ax.set_ylabel("Predicted Draft Pick")
ax.set_title(f"{selected_position}: Actual vs. Predicted Draft Picks")
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Reference line
plt.tight_layout()

st.pyplot(fig)
