import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz
import os
from tqdm import tqdm

# Initialize vectorizers
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

# Function to process and save chunks
def process_chunk(df_chunk, output_folder, index):
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df_chunk = df_chunk.copy()
        df_chunk['text'] = df_chunk['text'].fillna('')
        
        # Bag of Words transformation
        X_bow = bow_vectorizer.transform(df_chunk['text'])
        print(f"Bag of Words transformation complete for chunk {index}.")
        
        # TF-IDF transformation
        X_tfidf = tfidf_vectorizer.transform(df_chunk['text'])
        print(f"TF-IDF transformation complete for chunk {index}.")
        
        # Save sparse matrices to file
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        save_npz(f"{output_folder}/bow_chunk_{index}.npz", X_bow)
        save_npz(f"{output_folder}/tfidf_chunk_{index}.npz", X_tfidf)
        
        print(f"Chunk {index} saved successfully.")
    
    except Exception as e:
        print(f"An error occurred for chunk {index}: {e}")

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Process chunks with progress bar
def process_chunks(file_path, output_folder, chunk_size=100000):
    df = load_data(file_path)
    num_chunks = (len(df) // chunk_size) + 1

    # Fit vectorizers on the whole dataset (if you need consistent vocabularies)
    df['text'] = df['text'].fillna('')
    bow_vectorizer.fit(df['text'])
    tfidf_vectorizer.fit(df['text'])
    print("Vectorizers fitted on the whole dataset.")

    for i in tqdm(range(num_chunks), desc="Processing Chunks"):
        chunk_start = i * chunk_size
        chunk_end = chunk_start + chunk_size
        df_chunk = df[chunk_start:chunk_end]

        # Check if chunk is empty
        if df_chunk.empty:
            print(f"Chunk {i+1} is empty and will be skipped.")
            continue

        print(f"Processing chunk {i+1}...")
        print(f"Chunk {i+1} range: {chunk_start} to {chunk_end}")
        print(f"Number of samples in chunk: {len(df_chunk)}")

        # Process the chunk
        process_chunk(df_chunk, output_folder, i+1)
        print(f"Chunk {i+1} processed.")

# Example usage
file_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/sample_preprocessed_tweets.csv'
output_folder = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/processed_chunks'
process_chunks(file_path, output_folder)
