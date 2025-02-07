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

# Display the top 30 professions
print(profession_df.head(30))

# Plot: Top 30 most common professions
plt.figure(figsize=(12, 6))
plt.barh(profession_df["Profession"][:30], profession_df["Count"][:30])
plt.xlabel("Number of Mentions")
plt.ylabel("Profession")
plt.title("Top 30 Most Frequent Professions in Dataset")
plt.gca().invert_yaxis()  # Invert for better readability
plt.tight_layout()  # Adjust layout to prevent label cutoff
#plt.show()
plt.savefig('top_professions.png', bbox_inches='tight', transparent=False, facecolor='white')

# Save profession counts to CSV
profession_df.to_csv('profession_counts.csv', index=False)