import pandas as pd
import numpy as np
import os
import re
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

# Initialize variables for processing
id_counter = 0  # Global counter for entry IDs
batch_file_number = 0  # Global counter for batch files
merged_metadata = pd.DataFrame()  # Initialize an empty DataFrame for merging
merged_text_emb = np.empty((0, 512))  # Initialize empty arrays for embeddings
merged_img_emb = np.empty((0, 512))
file_summary = {}  # Initialize file summary dictionary

# Global variable to store profession tags
_profession_cache = None

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
    """
    # Use a global variable to cache the profession tags
    global _profession_cache
    if _profession_cache is None:
        try:
            df = pd.read_csv("../data/occupation_tags.csv")
            _profession_cache = set(df.iloc[:, 0])  # Read first column
            #print(f"Loaded {len(_profession_cache)} profession tags.")
        except Exception as e:
            print(f"Error loading occupation_tags.csv: {e}")
            _profession_cache = set()
    return _profession_cache

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
    # Define valid POS tags (expandable)
    VALID_NOUN_TAGS = {"NN", "NNS", "NNP"}
    VALID_ADJ_TAGS = {"JJ", "VBG"}  # Allow adjectives and gerunds like "nursing", "teaching"
        
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

def download_batch(file_number):
    """
    Downloads a single batch of data files.
    
    Parameters:
    file_number (int): Batch number to download
    
    Returns:
    tuple: (metadata, text_emb, img_emb) containing loaded data or None if files are missing

    The function:
    1. Downloads metadata, text embeddings and image embeddings for a batch
    2. Loads the metadata, text embeddings and image embeddings
    3. Returns the loaded data or None if any files are missing
    """
    str_i = str(file_number)

    print(f"Downloading data for batch {str_i}...")
    # Download batch data with retry mechanism
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = subprocess.run(["bash", "0_download.sh", str_i], timeout=300)
            if result.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            print(f"Attempt {attempt + 1} timed out, retrying...")
        if attempt == max_retries - 1:
            print(f"Failed to download batch {str_i} after {max_retries} attempts")
            return None, None, None
    
    metadata_file = f"metadata_{str_i}.parquet"
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} not found. Skipping batch {str_i}.")
        return None, None, None  # Skip if the metadata file is missing
    # Load metadata
    metadata = pd.read_parquet(metadata_file)
    
    text_emb_file = f"text_emb_{str_i}.npy"
    if not os.path.exists(text_emb_file):
        print(f"Text embeddings file {text_emb_file} not found. Skipping batch {str_i}.")
        return None, None, None  # Skip if the metadata file is missing
    # Load text embeddings
    text_emb = np.load(text_emb_file)
    
    img_emb_file = f"img_emb_{str_i}.npy"
    if not os.path.exists(img_emb_file):
        print(f"Image embeddings file {img_emb_file} not found. Skipping batch {str_i}.")
        return None, None, None  # Skip if the metadata file is missing
    # Load image embeddings
    img_emb = np.load(img_emb_file)
        
    return metadata, text_emb, img_emb

def filter_captions(data, file_id):
    """
    Filters and processes captions to identify occupations.

    Parameters:
    data (pandas.DataFrame): Input DataFrame containing captions
    file_id (str): ID of the current file being processed

    Returns:
    pandas.DataFrame: Filtered DataFrame containing only entries with identified occupations 

    The function:
    1. Adds entry IDs to the data
    2. Cleans caption text
    3. Identifies occupations in captions using parallel processing
    4. Filters to keep only entries with identified occupations
    5. Updates file summary with statistics
    """
    global file_summary
    
    # Print processing start message
    print(f"Preprocessing batch {file_id} with {len(data)} entries...")

    # Preserve original row index
    data["entry_id"] = data.index

    # this small hack is needed becase caption sometimes contains all kind of quotes, from ClickHouse
    data["caption"] = data["caption"].apply(lambda x: x.replace("'", " ").replace('"', " "))

    # Preserve original captions before cleaning
    data["original_caption"] = data["caption"]
    
    # Print cleaning message
    print(f"Cleaning and filtering captions...")

    # Use parallel processing for cleaning & filtering
    with Pool(cpu_count()) as pool:
        data["cleaned_caption"] = pool.map(clean_text, data["caption"])
        data["identified_occupations"] = pool.map(extract_professions, data["cleaned_caption"])
    
    # Filter out entries without identified occupations
    filtered_data = data[data["identified_occupations"].notnull()]
    
    # Count each detected profession in the dataset
    profession_count_dict = {}
    for professions in filtered_data["identified_occupations"]:
        for profession in professions:
            profession_count_dict[profession] = profession_count_dict.get(profession, 0) + 1 # Increment count
    
    # Store summary statistics
    file_summary[file_id] = {
        "file_id": file_id,
        "total_entries": len(data),
        "occupation_related_entries": len(filtered_data),
        "professions_count": profession_count_dict
    }

    # Print filtering result message
    print(f"Filtered {len(filtered_data)} occupation-related captions from batch {file_id}.")
    
    return filtered_data, file_summary

def load_img_emb(args):
    """
    Helper function to load image embeddings using parallel processing.
    """
    file, indices = args
    return np.load(file)[indices]

def load_text_emb(args):
    """
    Helper function to load text embeddings using parallel processing.
    """
    file, indices = args
    return np.load(file)[indices]

def filter_embeddings(filtered_data, file_number):
    """
    Loads and filters embeddings based on filtered captions.

    Parameters:
    filtered_data (pandas.DataFrame): DataFrame containing filtered caption data
    file_number (int): Current file number being processed

    Returns:
    tuple: (final_data, text_emb, img_emb) containing processed data and embeddings

    The function:
    1. Gets valid indices from filtered data
    2. Loads and filters embedding files
    3. Assigns unique IDs to entries
    4. Returns processed data and embeddings
    """
    global id_counter
    
    str_i = str(file_number)
    # Get valid indices from filtered data
    valid_indices = filtered_data["entry_id"].tolist()
    
    img_emb_file = f"img_emb_{str_i}.npy"
    text_emb_file = f"text_emb_{str_i}.npy"
    
    if not os.path.exists(img_emb_file) or not os.path.exists(text_emb_file):
        print(f"Missing embedding files for batch {str_i}. Skipping merging step.")
        return None, None, None  # Skip if the embedding files are missing
        
    # Load image and text embeddings
    print(f"Filtering image and text embeddings...")
    # Load embeddings using parallel processing
    with Pool(cpu_count()) as pool:
        # Load embeddings in parallel
        img_emb = pool.map(load_img_emb, [(img_emb_file, valid_indices)])[0]
        text_emb = pool.map(load_text_emb, [(text_emb_file, valid_indices)])[0]
    
    # Add overall entry IDs
    filtered_data.insert(0, "id", range(id_counter, id_counter + len(filtered_data)))
    id_counter += len(filtered_data) # Update global counter
    
    # Select final columns for output
    final_data = filtered_data[["id", "original_caption", "cleaned_caption", "identified_occupations", "similarity"]]
    
    return final_data, text_emb, img_emb

def remove_batch(file_number):
    """
    Removes temporary batch files.
    
    Parameters:
    file_number (int): Batch number to remove
    
    The function:
    1. Removes temporary files for a given batch
    """
    print(f"Removing temporary files for batch {file_number}...")
    try:
        str_i = str(file_number)
        for file in [f"metadata_{str_i}.parquet", f"img_emb_{str_i}.npy", f"text_emb_{str_i}.npy"]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except OSError as e:
                    print(f"Error removing {file}: {e}")
    except Exception as e:
        print(f"Error in remove_batch: {e}")
    str_i = str(file_number)
    for file in [f"metadata_{str_i}.parquet", f"img_emb_{str_i}.npy", f"text_emb_{str_i}.npy"]:
        if os.path.exists(file):
            os.remove(file)

def process_file(file_number, merged_metadata, merged_text_emb, merged_img_emb):
    """
    Main processing function that orchestrates the pipeline.

    Parameters:
    file_number (int): Current file number to process
    merged_metadata (pandas.DataFrame): Accumulated metadata
    merged_text_emb (numpy.ndarray): Accumulated text embeddings
    merged_img_emb (numpy.ndarray): Accumulated image embeddings

    Returns:
    tuple: (merged_metadata, merged_text_emb, merged_img_emb) with updated data
    """
    # Download batch data
    metadata, text_emb, img_emb = download_batch(file_number)
    if metadata is None:
        return merged_metadata, merged_text_emb, merged_img_emb, file_summary
        
    # Filter captions and embeddings
    filtered_data, file_summary = filter_captions(metadata, str(file_number))
    final_data, text_emb, img_emb = filter_embeddings(filtered_data, file_number)
    
    # Merge data with accumulated data
    if final_data is not None:
        merged_metadata = pd.concat([merged_metadata, final_data], ignore_index=True)
        merged_text_emb = np.concatenate((merged_text_emb, text_emb), axis=0)
        merged_img_emb = np.concatenate((merged_img_emb, img_emb), axis=0)

        # Check if we've exceeded 1M entries
        if len(merged_metadata) >= 1_000_000:
            # Save the current batch and get the next file number
            batch_file_number = save_batch(merged_metadata, merged_text_emb, merged_img_emb)
            print(f"Saved file {batch_file_number - 1} with {len(merged_metadata)} records")
            # Reset the merged data structures
            merged_metadata = pd.DataFrame()
            merged_text_emb = np.empty((0, 512))
            merged_img_emb = np.empty((0, 512))
    
    # Remove temporary batch files
    remove_batch(file_number)
    return merged_metadata, merged_text_emb, merged_img_emb, file_summary

def save_batch(metadata, text_emb, img_emb):
    """
    Saves a batch of data when it reaches 1M entries.
    
    Parameters:
    metadata (pandas.DataFrame): Current metadata batch
    text_emb (numpy.ndarray): Current text embeddings batch
    img_emb (numpy.ndarray): Current image embeddings batch
    
    Returns:
    int: Next file number to use
    """
    # Use global variable for file number tracking
    global batch_file_number
    if 'batch_file_number' not in globals():
        batch_file_number = 0
    next_num = batch_file_number
    batch_file_number += 1
    
    # Save the current batch
    metadata.to_csv(f"../data/1_1_filtered_metadata/filtered_metadata_{next_num}.csv", index=False)
    np.save(f"../data/2_1_filtered_text_emb/filtered_text_emb_{next_num}.npy", text_emb)
    np.save(f"../data/3_1_filtered_img_emb_{next_num}.npy", img_emb)

    return batch_file_number

def save_file(merged_metadata, merged_text_emb, merged_img_emb):
    """
    Saves any remaining data that didn't make it to a full 1M batch.

    Parameters:
    merged_metadata (pandas.DataFrame): Accumulated metadata
    merged_text_emb (numpy.ndarray): Accumulated text embeddings
    merged_img_emb (numpy.ndarray): Accumulated image embeddings

    The function:
    1. Saves the final merged dataset
    2. Saves the file summary to a CSV file
    """
    global batch_file_number

    if len(merged_metadata) > 0:
        batch_file_number = save_batch(merged_metadata, merged_text_emb, merged_img_emb)
    
    # Save file summary to CSV
    summary_df = pd.DataFrame.from_dict(file_summary, orient="index")
    summary_df.to_csv("../output/file_summary.csv", index=False)
    total_entries = summary_df["occupation_related_entries"].sum()
    
    print(f"Processing complete with {batch_file_number} files and {total_entries} entries filtered.")

# Main execution
if __name__ == "__main__":
    if torch.device("mps"):
        print(f"Using device: {device}")

    # Define the number of files (start from 1)
    total_files = 410

    # Process each file
    for file_id in range(total_files):
        print(f"Processing batch {file_id}...")
        merged_metadata, merged_text_emb, merged_img_emb, file_summary = process_file(file_id, merged_metadata, merged_text_emb, merged_img_emb)

    # Save the final merged dataset
    save_file(merged_metadata, merged_text_emb, merged_img_emb)
