import pandas as pd
import re
import fnmatch
from multiprocessing import Pool, cpu_count
import ast
import numpy as np

# Initialize a global dictionary to store the sum of text_bias and count of occurrences for each occupation
global_occ_dict = {}

def load_word_patterns(file_path):
    """
    Load a list of word patterns from a file.
    Each line in the file is a pattern.
    
    Parameters:
    file_path (str): The path to the file containing word patterns.

    Returns:
    list: A list of word patterns.
    """
    #print(f"Loading word patterns from {file_path}...")
    # Read one word per line, drop any empty entries and whitespace
    with open(file_path, "r", encoding="utf8") as f:
        patterns = [w.strip() for w in f if w.strip()]
        expanded = []
        for pattern in patterns:
            if '*' in pattern or '?' in pattern:
                # Expand wildcard pattern using the available embeddings
                for word in word_vectors.keys():
                    if fnmatch.fnmatch(word, pattern):
                        expanded.append(word)
            else:
                expanded.append(pattern)
    return expanded

def tokenize(text):
    """
    Tokenize a string into words.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    list: A list of words.
    """
    return text.split()

def load_glove_embeddings(file_path):
    """Load GloVe embeddings from a text file into a dictionary."""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

# Load GloVe embeddings
glove_path = "glove.6B.300d.txt"  # Change this path based on your embeddings
word_vectors = load_glove_embeddings(glove_path)

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    
    Parameters:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.
    
    Returns:
    float: Cosine similarity between vec1 and vec2.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# File paths
masculine_csv = "masculine_tags.csv"
feminine_csv = "feminine_tags.csv"

# Load word patterns
masculine_patterns = load_word_patterns(masculine_csv)
feminine_patterns = load_word_patterns(feminine_csv)

# Function to compute counts and bias_naive for a sentence
def compute_bias(sentence, lambda_coef=2.0):
    """
    Compute the counts of masculine and feminine words in a sentence,
    and calculate the bias score for the sentence.
    
    S1 is calculated as before based on normalized difference in pattern counts.
    S2 is calculated by:
      - Averaging the word vectors from the text to create V_T.
      - Computing the average cosine similarities between V_T and each word vector in the
        masculine and feminine word lists (only for words available in the embeddings).
      - S2 = (avg similarity for masculine words) - (avg similarity for feminine words)
    
    The final text_bias is S1 + lambda_coef * S2.
    """  
    # Tokenize
    tokens = tokenize(sentence)
    
    # Average cosine over masculine and feminine word list
    count_m = 0
    count_f = 0
    m_token_list = []
    f_token_list = []
    for token in tokens:
        if token in feminine_patterns:
            count_f += 1
            f_token_list.append(token)
            break # Only count once per word
        # Masculine patterns can be subset of feminine patterns (e.g. "*man" and "*women")
        elif token in masculine_patterns:
            count_m += 1
            m_token_list.append(token)
            break # Only count once per word
    
    # Calculate S1
    if count_m == count_f or (count_m == 0 and count_f == 0):
        S1 = 0
    elif count_m > count_f:
        S1 = (count_m - count_f) / count_m
    else:
        S1 = - (count_f - count_m) / count_f

    # Calculate S2
    # Compute text vector V_T: average of word vectors for tokens in the sentence
    token_vectors = [word_vectors[token] for token in tokens if token in word_vectors]
    V_T = np.mean(token_vectors, axis=0)

    sim_sum_m = 0
    sim_sum_f = 0

    for word in masculine_patterns:
        if word in word_vectors:
            sim_sum_m += cosine_similarity(V_T, word_vectors[word])
    for word in feminine_patterns:
        if word in word_vectors:
            sim_sum_f += cosine_similarity(V_T, word_vectors[word])

    avg_m = sim_sum_m / len(masculine_patterns)
    avg_f = sim_sum_f / len(feminine_patterns)
    
    S2 = avg_m - avg_f
    
    text_bias = S1 + lambda_coef * S2
    return pd.Series([count_m, m_token_list, count_f, f_token_list, text_bias])

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
        if row["text_bias"] != 0:
            occ_value = row.get("identified_occupations")
            occupations = ast.literal_eval(occ_value)
            for occ in occupations:
                if occ in global_occ_dict:
                    global_occ_dict[occ]["sum"] += row["text_bias"]
                    global_occ_dict[occ]["count"] += 1
                else:
                    global_occ_dict[occ] = {"sum": row["text_bias"], "count": 1}
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

if __name__ == '__main__':
    # Process each CSV file
    for i in range(9):
        metadata_csv = f"../1_Data_Exploration/filtered_metadata_{i}.csv"
        print(f"Loading metadata batch {i}...")
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        
        print(f"Processing captions for batch {i}...")
        with Pool(cpu_count()) as pool:
            results = pool.map(compute_bias, df['cleaned_caption'])
    
        df[['masculine_count', 'masculine_words', 'feminine_count', 'feminine_words', 'text_bias']] = pd.DataFrame(results, index=df.index)
        occ_dict = aggregate_by_occupation(df)

        output_csv = f"bias_metadata_{i}.csv"
        print(f"Saving updated CSV to: {output_csv}.")
        df.to_csv(output_csv, index=False)

    # Read the profession_counts CSV to get the list of possible professions
    prof_counts_path = "../1_Data_Exploration/profession_counts.csv"
    prof_df = pd.read_csv(prof_counts_path)

    # Use occ_dict from aggregate_by_occupation
    prof_df["avg_bias"] = prof_df['Profession'].apply(get_avg_bias)

    # Write updated profession counts to a new CSV file
    output_prof_csv = "profession_counts_with_bias.csv"
    print(f"Writing updated profession counts to {output_prof_csv}...")
    prof_df.to_csv(output_prof_csv, index=False)

    print("Processing complete.")