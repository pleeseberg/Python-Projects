import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
data_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/preprocessed_tweets.csv'
print(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(data.head())

# Print column names to check available columns
print("\nColumn names in the dataset:", data.columns)

# Check for missing values
print("\nChecking for missing values in the dataset...")
print(data.isnull().sum())

# Handle missing values
print("\nHandling missing values...")
data['text'].fillna('unknown', inplace=True)  # Replace NaNs with 'unknown' or any other placeholder

# Verify no missing values remain
print("\nMissing values after handling:")
print(data.isnull().sum())

# Prepare data
X = data['text']
y = data['target']  # Update this with the correct column name

# Load the vectorizer used during training
vectorizer_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/vectorizer.joblib'
print(f"Loading vectorizer from {vectorizer_path}...")
vectorizer = joblib.load(vectorizer_path)

# Vectorize the text data
print("Vectorizing text data...")
X_vectorized = vectorizer.transform(X)

# Split the data
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Load models
models = {
    'Logistic Regression': '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/Logistic_Regression_tuned.joblib',
    'Naive Bayes': '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/Naive_Bayes_tuned.joblib',
    'Linear SVC': '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/Linear_SVC_tuned.joblib'
}

# Initialize metrics storage
metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

# Define file paths for saving plots
conf_matrix_path = 'confusion_matrix.png'
roc_curve_path = 'roc_curve.png'
feature_importances_path = 'feature_importances.png'
word_cloud_path = 'word_cloud.png'
top_tfidf_scores_path = 'top_tfidf_scores.png'
performance_metrics_path = 'performance_metrics_comparison.png'

# Evaluate models
for name, model_path in models.items():
    print(f"\nEvaluating model {name}...")
    model = joblib.load(model_path)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}")
        
        # Collect metrics
        metrics['Model'].append(name)
        metrics['Accuracy'].append(report['accuracy'])
        metrics['Precision'].append(report['macro avg']['precision'])
        metrics['Recall'].append(report['macro avg']['recall'])
        metrics['F1-Score'].append(report['macro avg']['f1-score'])

        # Print metrics to terminal
        print(f"\nMetrics for {name}:")
        print(f"Accuracy: {report['accuracy']:.2f}")
        print(f"Precision: {report['macro avg']['precision']:.2f}")
        print(f"Recall: {report['macro avg']['recall']:.2f}")
        print(f"F1-Score: {report['macro avg']['f1-score']:.2f}")
        
    except ValueError as e:
        print(f"Error evaluating model {name}: {e}")

    try:
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(conf_matrix_path)  # Save plot to file
        plt.close()
        
        # ROC Curve
        if hasattr(model, "decision_function"):  # For models with decision function
            y_scores = model.decision_function(X_test)
        else:  # For models with predict_proba
            y_scores = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
        auc = roc_auc_score(y_test, y_scores)
        
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(roc_curve_path)  # Save plot to file
        plt.close()
        
        # Print AUC
        print(f"AUC for {name}: {auc:.2f}")
        
    except Exception as e:
        print(f"Error plotting for {name}: {e}")

# Feature Importance (only for models that support it)
for name, model_path in models.items():
    print(f"\nEvaluating Feature Importance for {name}...")
    model = joblib.load(model_path)
    
    if hasattr(model, "coef_"):
        importances = model.coef_.flatten()
        features = vectorizer.get_feature_names_out()
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances for {name}')
        plt.bar(range(10), importances[indices][:10], align='center')
        plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.savefig(feature_importances_path)  # Save plot to file
        plt.close()
        
        # Print top features
        print(f"Top features for {name}:")
        for i in range(10):
            print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")

# Word Cloud
print("\nGenerating word cloud...")
text = ' '.join(data['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Tweets')
plt.savefig(word_cloud_path)  # Save plot to file
plt.close()

# Top TF-IDF Scores
print("\nTop TF-IDF features:")
tfidf_scores = np.mean(X_vectorized.toarray(), axis=0)
top_indices = np.argsort(tfidf_scores)[::-1]
top_features = [vectorizer.get_feature_names_out()[i] for i in top_indices[:10]]
top_scores = tfidf_scores[top_indices[:10]]

plt.figure(figsize=(12, 8))
plt.barh(top_features, top_scores)
plt.xlabel('TF-IDF Score')
plt.title('Top TF-IDF Scores')
plt.savefig(top_tfidf_scores_path)  # Save plot to file
plt.close()

# Print top TF-IDF features
for feature, score in zip(top_features, top_scores):
    print(f"{feature}: {score:.4f}")

# Performance Metrics Comparison
metrics_df = pd.DataFrame(metrics)
plt.figure(figsize=(12, 8))
metrics_df.set_index('Model').plot(kind='bar')
plt.title('Performance Metrics Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.savefig(performance_metrics_path)  # Save plot to file
plt.close()

# Print performance metrics comparison
print("\nPerformance Metrics Comparison:")
print(metrics_df)

print("\nModel evaluation and visualization completed.")
