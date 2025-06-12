import pandas as pd
import fnmatch
from multiprocessing import Pool, cpu_count
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
            if '*' in pattern:
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
glove_path = "../data/glove.6B.50d.txt"  # Change this path based on your embeddings
#print(f"Loading GloVe corpus from {glove_path}...")
word_vectors = load_glove_embeddings(glove_path)

# File paths
masculine_csv = "../data/masculine_tags.csv"
feminine_csv = "../data/feminine_tags.csv"

# Load word patterns
masculine_patterns = load_word_patterns(masculine_csv)
feminine_patterns = load_word_patterns(feminine_csv)

# Function to compute counts and bias_naive for a sentence
def word_count(sentence):
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
        # Masculine patterns can be subset of feminine patterns (e.g. "*man" and "*women")
        elif token in masculine_patterns:
            count_m += 1
            m_token_list.append(token)
    
    # Calculate S1
    if count_m == count_f or (count_m == 0 and count_f == 0):
        count_score = 0
    elif count_m > count_f:
        count_score = (count_m - count_f) / count_m
    else:
        count_score = - (count_f - count_m) / count_f

    return pd.Series([count_m, m_token_list, count_f, f_token_list, count_score])

if __name__ == '__main__':
    # Process each CSV file
    for i in range(9):
        metadata_csv = f"../data/1_1_filtered_metadata/filtered_metadata_{i}.csv"
        
        # Load metadata
        df = pd.read_csv(metadata_csv)        
        print(f"Processing captions for batch {i}...")
        with Pool(cpu_count()) as pool:
            results = pool.map(word_count, df['cleaned_caption'])
    
        df[['masculine_count', 'masculine_words', 'feminine_count', 'feminine_words', 'count_score']] = pd.DataFrame(results, index=df.index)

        output_csv = f"../data/1_3_count_metadata/count_metadata_{i}.csv"
        print(f"Saving updated CSV to: {output_csv}.")
        df.to_csv(output_csv, index=False)
    
    print("Processing complete.")