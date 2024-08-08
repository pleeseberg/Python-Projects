import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download NLTK resources (if not already downloaded)
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
print("NLTK resources downloaded.")

# Initialize SpaCy model
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")
print("SpaCy model loaded.")

# Define a function for text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    # Tokenization using NLTK
    tokens = word_tokenize(text)

    # Normalize text: Convert to lowercase and remove punctuation
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Remove stop words using NLTK
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization using SpaCy
    doc = nlp(' '.join(tokens))
    tokens = [token.lemma_ for token in doc]
    
    return ' '.join(tokens)

# Path to the CSV file
input_csv_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/sample_preprocessed_tweets.csv'
output_csv_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/preprocessed_tweets.csv'

# Load the CSV file
print(f"Loading CSV file from {input_csv_path}...")
df = pd.read_csv(input_csv_path, encoding='latin1')
print("CSV file loaded.")

# Rename columns if necessary
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
print("Columns renamed.")

# Print the number of rows to confirm the size of the dataset
print(f"Number of rows in the dataset: {len(df)}")

# Initialize progress bar
print("Preprocessing text data...")
tqdm.pandas(desc="Processing Tweets")  # Set description for the progress bar

# Preprocess the 'text' column with progress bar
df['text'] = df['text'].progress_apply(preprocess_text)
print("Text data preprocessing completed.")

# Save the preprocessed data to a new CSV file
print(f"Saving preprocessed data to {output_csv_path}...")
df.to_csv(output_csv_path, index=False)
print(f"Preprocessed data saved to {output_csv_path}")
