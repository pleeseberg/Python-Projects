import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Save the vectorizer
vectorizer_filename = 'vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer saved to {vectorizer_filename}")

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

# Cross-validation and hyperparameter tuning
for name, model in models.items():
    print(f"\nCross-validation for model: {name}")
    scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='accuracy')
    print(f"Mean cross-validation accuracy: {scores.mean():.4f}")

    print(f"\nTuning model: {name}")
    param_grid = param_grids[name]
    
    if name == 'Naive Bayes':
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)

    search.fit(X_train_vec, y_train)
    best_model = search.best_estimator_
    best_params = search.best_params_
    print(f"Best Parameters: {best_params}")

    # Train and evaluate the best model
    y_pred = best_model.predict(X_test_vec)
    print(f"\nEvaluating model: {name}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_filename = f'{name.replace(" ", "_")}_tuned.joblib'
    joblib.dump(best_model, model_filename)
    print(f"{name} tuned model saved to {model_filename}")

    # Load and test the saved model
    loaded_model = joblib.load(model_filename)
    y_pred_loaded = loaded_model.predict(X_test_vec)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
    print(f"Model loaded from {model_filename} accuracy: {accuracy_loaded:.4f}")

    # Confusion Matrix for the loaded model
    def plot_normalized_confusion_matrix(cm, title='Confusion Matrix'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    cm = confusion_matrix(y_test, y_pred_loaded)
    plot_normalized_confusion_matrix(cm, f'Confusion Matrix ({name}) - Loaded Model')

    # Plot learning curves
    def plot_learning_curve(model, X, y, title):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.title(f"Learning Curves ({title})")
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    plot_learning_curve(best_model, X_train_vec, y_train, name)

    # Feature importance/analysis
    if hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])
        top_features = np.argsort(importance)[-10:]  # Top 10 features
        feature_names = vectorizer.get_feature_names_out()
        print("\nTop 10 features:")
        for i in top_features:
            print(f"{feature_names[i]}: {importance[i]:.4f}")

print("\nHyperparameter tuning and model evaluation completed.")
