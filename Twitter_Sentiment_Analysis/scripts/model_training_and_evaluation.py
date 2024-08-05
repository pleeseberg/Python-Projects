import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def save_model(model, filename):
    if not os.path.exists(filename):
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    else:
        print(f"File {filename} already exists. Model not saved.")

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def process_and_evaluate_model(name, model, param_grid, X_train_vec, y_train, X_test_vec, y_test):
    print(f"\nCross-validation for model: {name}")
    scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    print(f"Mean cross-validation accuracy: {scores.mean():.4f}")

    print(f"\nTuning model: {name}")
    search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    search.fit(X_train_vec, y_train)
    print(f"Best Parameters: {search.best_params_}")

    # Evaluate the best model
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, labels=np.unique(y_test))

    # Save the model
    model_filename = f"{name.replace(' ', '_')}_tuned.joblib"
    save_model(best_model, model_filename)
    
    # Load and validate the model
    loaded_model = joblib.load(model_filename)
    loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test_vec))
    print(f"Model loaded from {model_filename} accuracy: {loaded_accuracy:.4f}")

    # Extract and print top features if applicable
    feature_names = vectorizer.get_feature_names_out()
    if name in ['Logistic Regression', 'Linear SVC']:
        coef = best_model.coef_.flatten()
        top_indices = np.argsort(coef)[::-1][:10]
        top_features = [(feature_names[i], coef[i]) for i in top_indices]
        print("Top 10 features:")
        for feature, score in top_features:
            print(f"{feature}: {score:.4f}")
    elif name == 'Naive Bayes':
        log_prob = best_model.feature_log_prob_
        top_features = []
        for i in range(log_prob.shape[0]):
            top_indices = np.argsort(log_prob[i])[::-1][:10]
            top_features.append((f'Class {i}', [(feature_names[j], log_prob[i][j]) for j in top_indices]))
        print("Top 10 features for each class:")
        for class_label, features in top_features:
            print(f"\n{class_label}:")
            for feature, score in features:
                print(f"{feature}: {score:.4f}")

# Load data
data_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/preprocessed_tweets.csv'
print(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)
print("Data loaded successfully.")

# Handle missing values
print("\nHandling missing values...")
data['text'] = data['text'].fillna('unknown')

# Split data
print("\nSplitting data into training and testing sets...")
X = data['text']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split completed. Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# Feature extraction
print("\nExtracting features from text using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature extraction completed. Number of features: {X_train_vec.shape[1]}")

# Save vectorizer
save_model(vectorizer, 'vectorizer.joblib')

# Initialize models and hyperparameters for tuning
models = {
    'Logistic Regression': LogisticRegression(max_iter=100),
    'Naive Bayes': MultinomialNB(),
    'Linear SVC': LinearSVC()
}

param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'Naive Bayes': {
        'alpha': [0.5, 1.0, 2.0]
    },
    'Linear SVC': {
        'C': [0.1, 1, 10],
        'max_iter': [1000, 2000]
    }
}

# Process and evaluate each model
for name, model in models.items():
    process_and_evaluate_model(name, model, param_grids[name], X_train_vec, y_train, X_test_vec, y_test)

print("\nHyperparameter tuning and model evaluation completed.")
