import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm  # Progress bar library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import save_npz, hstack
import joblib
import gzip
import shutil
import os

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
input_csv_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/training.1600000.processed.noemoticon.csv'
output_csv_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/preprocessed_tweets.csv'
sample_csv_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/sample_preprocessed_tweets.csv'

# Load the CSV file
print(f"Loading CSV file from {input_csv_path}...")
df = pd.read_csv(input_csv_path, header=None, encoding='latin1')
print("CSV file loaded.")

# Assuming the tweet text is in the 5th column (index 4) and there is no header
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']  # Update column names if necessary
print("Columns renamed.")

# Reduce the sample size
sample_df = df.sample(n=10000, random_state=42)  # Adjust the sample size as needed
print("Sample data created.")

# Initialize progress bar
print("Preprocessing text data...")
tqdm.pandas(desc="Processing Tweets")  # Set description for the progress bar

# Preprocess the 'text' column with progress bar
sample_df['text'] = sample_df['text'].progress_apply(preprocess_text)
print("Text data preprocessing completed.")

# Save the preprocessed sample data to a new CSV file
print(f"Saving preprocessed sample data to {sample_csv_path}...")
sample_df.to_csv(sample_csv_path, index=False)
print(f"Preprocessed sample data saved to {sample_csv_path}")

# Vectorize the text data
print("Vectorizing text data...")
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

X_bow = bow_vectorizer.fit_transform(sample_df['text'])
X_tfidf = tfidf_vectorizer.fit_transform(sample_df['text'])

# Combine the features into one sparse matrix
X_combined = hstack([X_bow, X_tfidf])

# Save combined matrix to a single file
combined_features_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/combined_features.npz'
save_npz(combined_features_path, X_combined)
print(f"Combined features saved to {combined_features_path}")

# Save vectorizers
joblib.dump(bow_vectorizer, '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/bow_vectorizer.joblib')
joblib.dump(tfidf_vectorizer, '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/tfidf_vectorizer.joblib')
print("Vectorizers saved.")

# Compress the files
with open(combined_features_path, 'rb') as f_in:
    with gzip.open(combined_features_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(combined_features_path)

joblib_files = ['/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/bow_vectorizer.joblib',
                '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/tfidf_vectorizer.joblib']

for file in joblib_files:
    with open(file, 'rb') as f_in:
        with gzip.open(file + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file)

print("Files compressed.")
