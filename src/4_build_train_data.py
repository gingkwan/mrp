import pandas as pd
import numpy as np
import ast

def filter_entries():
    all_filtered = []  # List to collect filtered dataframes
    filtered_text_emb = []  # List to collect filtered text embeddings
    filtered_image_emb = []  # List to collect filtered image embeddings
    indices = {}  # List to collect indices of filtered entries

    # Loop through files count_metadata_0.csv to count_metadata_8.csv
    for i in range(9):
        print(f"Processing file count_metadata_{i}.csv")
        filename = f"../data/1_3_count_metadata/count_metadata_{i}.csv"
        df = pd.read_csv(filename)

        target_masc = {'he', 'him', 'his', 'men', 'man'}
        target_fem = {'she', 'her', 'hers', 'women', 'woman'}

        filtered_df = df[
            (
            df['masculine_words'].apply(lambda x: any(word in target_masc for word in ast.literal_eval(x))) |
            df['feminine_words'].apply(lambda x: any(word in target_fem for word in ast.literal_eval(x)))
            )
        ].copy()

        all_filtered.append(filtered_df)
        # Get the indices of the filtered rows
        indices[i] = filtered_df.index.tolist()

        # Load the embedding files for the current index
        text_emb = np.load(f"../data/2_1_filtered_text_emb/filtered_text_emb_{i}.npy")[indices[i]]
        image_emb = np.load(f"../data/3_1_filtered_img_emb/filtered_img_emb_{i}.npy")[indices[i]]
        filtered_text_emb.append(text_emb)
        filtered_image_emb.append(image_emb)

    # Combine all filtered dataframes into one (if any)
    combined_df = pd.concat(all_filtered, ignore_index=True)
    combined_text_emb = np.concatenate(filtered_text_emb, axis=0)
    combined_image_emb = np.concatenate(filtered_image_emb, axis=0)
    print("\nShape of the filtered DataFrame:")
    print(combined_df.shape)
    print("\nShape of the filtered text embeddings:")
    print(combined_text_emb.shape)
    print("\nShape of the filtered image embeddings:")
    print(combined_image_emb.shape)

    # Save the combined dataframes and embeddings
    combined_df.to_csv("../data/4_train_data/train_metadata.csv", index=False)
    np.save("../data/4_train_data/train_text_emb.npy", combined_text_emb)
    np.save("../data/4_train_data/train_img_emb.npy", combined_image_emb)

    print("Filtered data saved.")

    # Save the filtered indices to a CSV file
    combined_indices = pd.DataFrame.from_dict(indices, orient='index').transpose()
    combined_indices.to_csv("../data/4_train_data/train_indices.csv", index=False)

    print("Filtered indices saved to train_indices.csv")

    return indices

if __name__ == '__main__':  
    indices = filter_entries()