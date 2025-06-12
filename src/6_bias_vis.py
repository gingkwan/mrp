import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

global_occ_dict = {}

def aggregate_by_occupation(df_input):
    """
    Aggregate the text_bias values by occupation.

    Parameters:
    df_input (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary where the keys are occupations and the values are dictionaries
    with the sum of text_bias and the count of occurrences.
    """
    for _, row in df_input.iterrows():
        # Only aggregate if text_bias is not zero
        if row["bias_score_pred"] != 0:
            occ_value = row.get("identified_occupations")
            occupations = ast.literal_eval(occ_value)
            for occ in occupations:
                if occ in global_occ_dict:
                    global_occ_dict[occ]["sum"] += row["bias_score_pred"]
                    global_occ_dict[occ]["count"] += 1
                else:
                    global_occ_dict[occ] = {"sum": row["bias_score_pred"], "count": 1}
    return global_occ_dict

def get_avg_bias(prof):
    """
    Get the average bias score for each profession in the counts file.
    
    Parameters:
    prof (str): The profession to look up.
    
    Returns:
    float: The average bias score for the profession, or None if not available.
    """
    if prof in global_occ_dict:
        vals = global_occ_dict[prof]
        return vals["sum"] / vals["count"]
    else:
        return None
    
def plot_graph(df):
    # Ensure numeric values for selected columns
    cols_to_convert = ["Count", "avg_bias"]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in avg_bias
    df = df.dropna(subset=["avg_bias"])

    # Load occupation tags with percentage of women
    occupation_file = "../data/occupation_tags_with_count.csv"
    occ_df = pd.read_csv(occupation_file, header=None, names=["Profession", "Women_pct"])
    occ_df["Women_pct"] = pd.to_numeric(occ_df["Women_pct"], errors='coerce')

    # Merge the datasets on "Profession". Keep rows even if Women_pct is missing.
    df = df.merge(occ_df, on="Profession", how="left")

    # Define classification function using threshold=0.3.
    # If the value is missing, returns "No Information".
    def classify_women_pct(women_pct, lower=0.5):
        if pd.isna(women_pct):
            return "No Information"
        if women_pct < lower:
            return "Male dominated"
        else:
            return "Female dominated"

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

    plt.savefig("../output/bias_representation.png")
    plt.show()
    plt.close()

    # Plot histogram of the text_bias distribution
    plt.figure(figsize=(8, 6))
    plt.hist(df['bias_score_pred'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Text Bias")
    plt.ylabel("Count")
    plt.title("Distribution of Text Bias")
    plt.savefig("../output/bias_histogram.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    metadata = pd.DataFrame()
    # Load the bias metadata file
    for i in range(9):
        metadata_csv = f"../data/1_5_bias_metadata/bias_metadata_{i}.csv"
        print(f"Loading metadata batch {i}...")
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        metadata = pd.concat([metadata, df], ignore_index=True)

    occ_dict = aggregate_by_occupation(metadata)   

    # Read the profession_counts CSV to get the list of possible professions
    prof_counts_path = "../output/profession_counts.csv"
    prof_df = pd.read_csv(prof_counts_path)

    # Use occ_dict from aggregate_by_occupation
    prof_df["avg_bias"] = prof_df['Profession'].apply(get_avg_bias)

    # Write updated profession counts to a new CSV file
    output_prof_csv = "../output/profession_counts_with_bias.csv"
    print(f"Writing updated profession counts to {output_prof_csv}...")
    prof_df.to_csv(output_prof_csv, index=False)

    plot_graph(prof_df)
    print("Processing complete.") 
        