import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load file_summary.csv
file_summary = pd.read_csv("file_summary.csv")

# Rename columns for clarity
file_summary.columns = ["file_id", "total_entries", "occupation_related_entries", "professions_count"]

# Convert professions_count column from string to dictionary
file_summary["professions_count"] = file_summary["professions_count"].apply(ast.literal_eval)

# Compute the percentage of occupation-related entries
file_summary["occupation_percentage"] = (file_summary["occupation_related_entries"] / file_summary["total_entries"]) * 100

# Print overall statistics
print("===== Summary Statistics =====")
print(f"Total Batches Processed: {len(file_summary)}")
print(f"Average % of occupation-related entries: {file_summary['occupation_percentage'].mean():.2f}%")
print(f"Maximum % of occupation-related entries: {file_summary['occupation_percentage'].max():.2f}%")
print(f"Minimum % of occupation-related entries: {file_summary['occupation_percentage'].min():.2f}%")

# Aggregate profession counts across all batches
total_profession_counts = {}
for prof_dict in file_summary["professions_count"]:
    for prof, count in prof_dict.items():
        total_profession_counts[prof] = total_profession_counts.get(prof, 0) + count

# Convert to DataFrame for visualization
profession_df = pd.DataFrame(list(total_profession_counts.items()), columns=["Profession", "Count"])
profession_df = profession_df.sort_values(by="Count", ascending=False)

print(f"Total distinct professions identified: {len(total_profession_counts)}")

# Display the top 50 professions
print(profession_df.head(50))

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 12))

# Create horizontal bar chart using counts for bar lengths and color gray
bars = ax.barh(profession_df["Profession"].head(50)[::-1], profession_df["Count"].head(50)[::-1], color="gray")

# Customize labels and title
ax.set_xlabel("Number of Mentions", fontsize=12)
ax.set_ylabel("Profession", fontsize=12)
ax.set_title("Top 50 Most Frequent Professions in Dataset", fontsize=14)
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{int(bar.get_width())}", va="center", ha="left", fontsize=8)

# Improve spacing by adjusting margins
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)

plt.show()
plt.savefig('top_professions.png', bbox_inches='tight', transparent=False, facecolor='white')

# Save profession counts to CSV
profession_df.to_csv('profession_counts.csv', index=False)


