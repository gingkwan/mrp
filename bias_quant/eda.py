import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the original dataset
file_path = "profession_counts_with_bias.csv"
df = pd.read_csv(file_path)

# Ensure numeric values for selected columns
cols_to_convert = ["Count", "avg_bias"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in avg_bias
df = df.dropna(subset=["avg_bias"])

# Load occupation tags with percentage of women
occupation_file = "occupation_tags_with_count.csv"
occ_df = pd.read_csv(occupation_file, header=None, names=["Profession", "Women_pct"])
occ_df["Women_pct"] = pd.to_numeric(occ_df["Women_pct"], errors='coerce')

# Merge the datasets on "Profession". Keep rows even if Women_pct is missing.
df = df.merge(occ_df, on="Profession", how="left")

# Define classification function using threshold=0.3.
# If the value is missing, returns "No Information".
def classify_women_pct(women_pct, lower=0.33):
    if pd.isna(women_pct):
        return "No Information"
    if women_pct < lower:
        return "Male dominated"
    elif women_pct > (1 - lower):
        return "Female dominated"
    else:
        return "Relatively equal"

df["bias_category"] = df["Women_pct"].apply(classify_women_pct)

# Select the top 50 professions with the highest absolute avg_bias and sort by avg_bias descending
df_top = df.loc[df["Count"].abs().nlargest(50).index]
df_top = df_top.sort_values(by="avg_bias", ascending=True)

# Define colors based on Women_pct bias category (add a color for "No Information")
colors = {
    "Male dominated": "lightblue",
    "Female dominated": "lightcoral",
    "Relatively equal": "gray",
    "No Information": "lightgray"
}
df_top["color"] = df_top["bias_category"].map(colors)

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 12))

# Create horizontal bar chart using avg_bias for bar lengths and color based on Women_pct classification
ax.barh(df_top["Profession"], df_top["avg_bias"], color=df_top["color"])
ax.axvline(x=0, color="black", linestyle="dashed", linewidth=1)  # Reference line at 0

# Customize labels and title
ax.set_xlabel("Average Bias", fontsize=12)
ax.set_ylabel("Professions", fontsize=12)
ax.set_title("Bias Representation of the Top 50 Occupations", fontsize=14)

# Annotate each bar using the avg_bias information.
# For positive values, add the label to the right; for negative values, to the left.
for i, patch in enumerate(ax.patches):
    avg_bias_val = df_top.iloc[i]["avg_bias"]
    width = patch.get_width()
    y = patch.get_y() + patch.get_height() / 2
    label_text = f"{avg_bias_val:.2f}"
    if width >= 0:
        x_pos = width + 0.005
        ha = "left"
    else:
        x_pos = width - 0.005
        ha = "right"
    ax.text(x_pos, y, label_text, va="center", ha=ha, fontsize=8)

# Add legend in the bottom right corner
legend_labels = [plt.Line2D([0], [0], color=colors[key], lw=4) for key in colors.keys()]
ax.legend(legend_labels, colors.keys(), loc="lower right")
ax.tick_params(axis="y", labelsize=8)

# Improve spacing by adjusting margins
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

plt.savefig("bias_representation.png")
plt.show()
plt.close()

# Aggregate data for histogram generation
histogram_data = []
for i in range(1):  # Adjust the range if you have more batches
    bias_metadata_csv = f"bias_metadata_{i}.csv"
    df_batch = pd.read_csv(bias_metadata_csv)
    # Select columns useful for plotting histograms, e.g., caption text, text_bias, and occupations.
    histogram_data.append(df_batch[['text_bias']])

# Combine all histogram data into a single DataFrame
combined_histogram_data = pd.concat(histogram_data, ignore_index=True)
combined_histogram_data = combined_histogram_data.dropna(subset=['text_bias'])

# Plot histogram of the text_bias distribution
plt.figure(figsize=(8, 6))
plt.hist(combined_histogram_data['text_bias'], bins=100, color='skyblue', edgecolor='black')
plt.xlabel("Text Bias")
plt.ylabel("Count")
plt.title("Distribution of Text Bias")
plt.savefig("bias_histogram.png")
plt.show()
plt.close()
        