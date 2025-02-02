import pandas as pd
import numpy as np
import os
import sys
import re
import nltk
import json
import torch
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import subprocess
from multiprocessing import Pool, cpu_count

# Download required NLTK datasets
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(f"Using device: {device}")

# Define valid POS tags (expandable)
VALID_NOUN_TAGS = {"NN", "NNS", "NNP"}
VALID_ADJ_TAGS = {"JJ", "VBG"}  # Allow adjectives and gerunds like "nursing", "teaching"

# Dictionary to store summary of processing
file_summary = {}
id_counter = 0  # Global counter for entry IDs

# Load the list of professions from `occupation_tags.csv`
def load_professions():
    """
    Loads the list of professions from the `occupation_tags.csv` file.
    
    Returns:
    set: Set of profession tags loaded from the file

    The function:
    1. Reads the `occupation_tags.csv` file
    2. Extracts the profession tags from the first column
    3. Returns the set of profession tags

    Examples:
    >>> load_professions()
    Loaded 1000 profession tags.
    """
    try:
        df = pd.read_csv("occupation_tags.csv")
        profession_list = set(df.iloc[:, 0])  # Read first column
        #print(f"Loaded {len(profession_list)} profession tags.")
        return profession_list
    except Exception as e:
        print(f"Error loading occupation_tags.csv: {e}")
        return set()

PROFESSIONS = load_professions()

# Function to clean text (lowercase, remove punctuation, numbers)
def clean_text(text):
    """
    Cleans the input text by converting to lowercase, removing punctuation and numbers.

    Parameters:
    text (str): Input text to clean

    Returns:
    str: Cleaned text with punctuation, numbers removed and lowercase

    The function:
    1. Converts text to lowercase
    2. Removes numbers
    3. Removes punctuation
    4. Tokenizes words and joins them back into a string

    Examples:
    >>> clean_text("Hello, World! 123")
    'hello world'
    >>> clean_text("This is a test.")
    'this is a test'
    >>> clean_text("The quick brown fox jumps over the lazy dog.")
    'the quick brown fox jumps over the lazy dog'
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    return ' '.join(words)  # Return cleaned text as a string

# Function to check if a caption contains an occupation (using NLTK POS tagging)
def extract_professions(text):
    """
    Extracts professions from a given text using NLTK POS tagging.

    Parameters:
    text (str): Input caption to analyze for profession mentions

    Returns:
    list or None: List of detected professions if found, None otherwise

    The function:
    1. Cleans and tokenizes the input text
    2. Applies POS tagging to identify nouns and adjectives
    3. Matches words against known profession list
    4. Validates matches using POS tags to reduce false positives
    5. Returns a list of detected professions or None if no professions are found

    Notes:
    - The function uses a predefined list of professions (PROFESSIONS)
    - The function uses NLTK POS tagging for word classification
    - The function allows for multi-word profession detection using regex
    
    Examples:
    >>> extract_professions("She works as a nursing assistant in a hospital")
    ['nursing assistant']
    >>> extract_professions("He is a teaching assistant at the university.")
    ['teaching assistant']
    >>> extract_professions("She is a software engineer at Microsoft.")
    ['software engineer']
    >>> extract_professions("He is a police officer and a teacher.")
    ['police officer', 'teacher']
    >>> extract_professions("The bank teller helped the customer.")
    ['bank teller']
    >>> extract_professions("He has a great teaching style.")
    None
    >>> extract_professions("John is learning nursing skills.")
    None
    """
    text = clean_text(text)  # Ensure text is preprocessed before matching
    words = word_tokenize(text)  # Tokenize words
    tagged_words = pos_tag(words)  # Apply POS tagging
    
    detected_professions = set()  # Use a set to avoid duplicates
    
    # Convert tagged words to a dictionary for easy lookup
    word_pos_dict = {word: tag for word, tag in tagged_words}

    # Regex-based multi-word detection
    for profession in PROFESSIONS:
        profession_regex = r"\b" + re.sub(r"\s+", r"\\s+", profession) + r"\w*\b"  # Keeps sequence & allows plural forms
        if re.search(profession_regex, text):  # Search for profession in caption
            profession_words = profession.split()
            
            # Check if at least one word is a noun and allow some adjectives or gerunds
            if any(word in word_pos_dict and word_pos_dict[word] in VALID_NOUN_TAGS for word in profession_words):
                if all(word in word_pos_dict and word_pos_dict[word] in (VALID_NOUN_TAGS | VALID_ADJ_TAGS) for word in profession_words):
                    detected_professions.add(profession)

    return list(detected_professions) if detected_professions else None

# Function to download, process, and clean files in a single step
def process_file(file_number, merged_data):
    """
    Downloads, processes, and cleans a single batch of metadata and embeddings.
    
    Parameters:
    file_number (int or str): The batch number/identifier to process
    merged_data (pandas.DataFrame): The existing merged dataset to append processed data to

    Returns:
    pandas.DataFrame: Updated merged dataset containing the processed batch data

    Global Variables:
    id_counter (int): Global counter for assigning unique IDs to entries

    The function:
    This function handles the complete pipeline for processing a single batch of data:
    1. Downloads metadata and embedding files
    2. Cleans and filters captions containing occupation mentions
    3. Matches embeddings with filtered captions
    4. Merges the processed data into the main dataset

    Notes:
    - Requires external files: metadata_{file_number}.parquet, img_emb_{file_number}.npy, 
      text_emb_{file_number}.npy
    - Creates temporary files during processing
    - Updates global file_summary dictionary with batch statistics
    - Automatically cleans up temporary files after processing
    """
    global id_counter  # Ensure global ID is correctly updated across files

    str_i = str(file_number)
    
    # Download the batch
    print(f"Downloading data for batch {str_i}...")
    subprocess.run(["bash", "./download.sh", str_i])  # Run the download script

    metadata_file = f"metadata_{str_i}.parquet"

    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Skipping batch {str_i}.")
        return merged_data  # Skip if the metadata file is missing

    # Load metadata
    data = pd.read_parquet(metadata_file)
    data["entry_id"] = data.index  # Preserve original row index

    # Print processing start message
    print(f"Preprocessing file {str_i} with {len(data)} entries...")

    # this small hack is needed becase caption sometimes contains all kind of quotes, from ClickHouse
    data["caption"] = data["caption"].apply(lambda x: x.replace("'", " ").replace('"', " "))

    # Preserve original captions before cleaning
    data["original_caption"] = data["caption"]

    # Clean captions
    print(f"Cleaning captions...")
    
    # Use parallel processing for cleaning & filtering
    with Pool(cpu_count()) as pool:
        data["caption"] = pool.map(clean_text, data["caption"])
        data["identified_occupations"] = pool.map(extract_professions, data["caption"])

    # Remove duplicated entries
    #data["identified_occupations"] = data["identified_occupations"].apply(lambda x: list(set(x)) if x else None)
    
    filtered_data = data[data["identified_occupations"].notnull()]

    # Count each detected profession in the dataset
    profession_count_dict = {}
    for professions in filtered_data["identified_occupations"]:
        for profession in professions:
            profession_count_dict[profession] = profession_count_dict.get(profession, 0) + 1  # Efficient counting

    # Store summary statistics
    file_summary[str_i] = {
        "file_id": str_i,
        "total_entries": len(data),
        "occupation_related_entries": len(filtered_data),
        "professions_count": profession_count_dict  # Counting for multiple professions per caption
    }

    # Get the original indices of the selected captions
    valid_indices = filtered_data["entry_id"].tolist()

    # Remove metadata file after processing
    os.remove(metadata_file)

    print(f"Filtered {len(filtered_data)} occupation-related captions from file {str_i}.")

    # Load image and text embeddings
    print(f"Loading image and text embeddings...")
    img_emb_file = f"img_emb_{str_i}.npy"
    text_emb_file = f"text_emb_{str_i}.npy"

    if not os.path.exists(img_emb_file) or not os.path.exists(text_emb_file):
        print(f"Missing embedding files for batch {str_i}. Skipping merging step.")
        return merged_data

    img_emb = np.load(img_emb_file)
    text_emb = np.load(text_emb_file)

    # Select only embeddings corresponding to filtered captions using valid_indices
    print(f"Filtering embeddings...")
    img_emb = img_emb[valid_indices]
    text_emb = text_emb[valid_indices]

    # Convert embeddings to list format
    img_emb_list = list(img_emb)
    text_emb_list = list(text_emb)

    # Ensure dimensions match (truncate if needed)
    min_len = min(len(filtered_data), len(img_emb_list), len(text_emb_list))
    filtered_data = filtered_data.iloc[:min_len]

    # Convert embeddings to JSON strings
    filtered_data["image_embedding"] = [json.dumps(emb.tolist()) for emb in img_emb[:min_len]]
    filtered_data["text_embedding"] = [json.dumps(emb.tolist()) for emb in text_emb[:min_len]]

    # Add overall entry ID
    filtered_data.insert(0, "id", range(id_counter, id_counter + len(filtered_data)))
    id_counter += len(filtered_data)  # Update global counter

    # Add file ID as a column
    filtered_data.insert(1, "file_id", str_i)

    # Keep only the required columns
    print(f"Creating filtered dataset...")
    final_data = filtered_data[["id", "file_id", "entry_id", "original_caption", 
                                "identified_occupations", "similarity", "image_embedding", "text_embedding"]]

    # Append to global DataFrame
    merged_data = pd.concat([merged_data, final_data], ignore_index=True)

    print(f"Merged {len(final_data)} records from file {str_i}.")
    
    # Cleanup intermediate files (optional)
    os.remove(img_emb_file)
    os.remove(text_emb_file)

    return merged_data

# Main execution
if __name__ == "__main__":
    merged_data = pd.DataFrame()  # Initialize an empty DataFrame for merging

    # Define the number of files (start from 1)
    total_files = 2

    for file_id in range(total_files):
        print(f"Processing batch {file_id}...")
        merged_data = process_file(file_id, merged_data)

    # Save final merged dataset
    final_output_file = "merged_dataset.csv"
    merged_data.to_csv(final_output_file, index=False)

    # Save file summary to CSV
    summary_df = pd.DataFrame.from_dict(file_summary, orient="index")
    summary_df.to_csv("file_summary.csv", index=False)

    print(f"Final merged dataset saved as {final_output_file} with {len(merged_data)} records.")