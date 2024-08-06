import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Define file paths
data_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/sample_preprocessed_tweets.csv'
vectorizer_path = 'vectorizer.joblib.gz'

# Load data
print(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Print the first few rows and class distribution
print("\nFirst few rows of the dataset:")
print(data.head())

print("\nClass distribution in the dataset:")
print(data['target'].value_counts())

# Check for class diversity
class_counts = data['target'].value_counts()
if len(class_counts) < 2:
    print("\nInsufficient class diversity in the dataset. Model evaluation will be skipped.")
else:
    # Handle missing values
    print("\nHandling missing values...")
    data = data.dropna()

    # Split data into features and target
    X = data['text']  # Assuming the text column is named 'text'
    y = data['target']
    
    # Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split completed. Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    # Extract features from text using TF-IDF
    print("\nExtracting features from text using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)
    print(f"Feature extraction completed. Number of features: {X_train_vec.shape[1]}")
    
    # Save the vectorizer
    if not os.path.exists(vectorizer_path):
        joblib.dump(tfidf_vectorizer, vectorizer_path)
    else:
        print(f"File {vectorizer_path} already exists. Model not saved.")

    # Define models and parameter grids
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Linear SVC': LinearSVC(max_iter=1000)
    }
    
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10]
        },
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0]
        },
        'Linear SVC': {
            'C': [0.1, 1, 10]
        }
    }
    
    # Evaluate and tune models
    for name, model in models.items():
        print(f"\nCross-validation for model: {name}")
        if len(class_counts) < 2:
            print(f"Skipping model evaluation for {name} due to insufficient class diversity.")
            continue

        try:
            scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
            print(f"Mean cross-validation accuracy: {scores.mean():.4f}")
        except ValueError as e:
            print(f"Cross-validation failed: {e}")

        print(f"Tuning model: {name}")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train_vec, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        # Evaluate on the test set
        y_pred = best_model.predict(X_test_vec)
        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy:.4f}")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save the model
        model_filename = f"{name.replace(' ', '_')}_tuned.joblib.gz"
        joblib.dump(best_model, model_filename)
        print(f"Model saved to {model_filename}")

        # Load and validate the saved model
        loaded_model = joblib.load(model_filename)
        loaded_accuracy = (loaded_model.predict(X_test_vec) == y_test).mean()
        print(f"Model loaded from {model_filename} accuracy: {loaded_accuracy:.4f}")

        # Show top features for Naive Bayes
        if name == 'Naive Bayes':
            feature_names = tfidf_vectorizer.get_feature_names_out()
            top10 = np.argsort(loaded_model.feature_log_prob_[1])[-10:]
            print("\nTop 10 features for each class:")
            for i in top10:
                print(f"{feature_names[i]}: {loaded_model.feature_log_prob_[1][i]:.4f}")

print("Hyperparameter tuning and model evaluation completed.")
