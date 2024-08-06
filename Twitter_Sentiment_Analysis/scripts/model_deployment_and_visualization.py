import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load dataset
data_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/data/preprocessed_tweets.csv'
print(f"Loading data from {data_path}...")
data = pd.read_csv(data_path)

# Display first few rows and columns
print("First few rows of the dataset:")
print(data.head())
print("\nColumn names in the dataset:", data.columns)

# Check for missing values
print("\nChecking for missing values in the dataset...")
missing_values = data.isnull().sum()
print(missing_values)

# Handle missing values
print("\nHandling missing values...")
data['text'].fillna('unknown', inplace=True)  # Replace NaNs with 'unknown' or any other placeholder

# Verify missing values are handled
missing_values_after = data.isnull().sum()
print("Missing values after handling:")
print(missing_values_after)

# Vectorize text data
print("\nLoading vectorizer from /Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/vectorizer.joblib...")
vectorizer = joblib.load('/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/scripts/vectorizer.joblib')
X = vectorizer.transform(data['text'])
y = data['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
model_logistic = LogisticRegression(max_iter=1000)
model_nb = MultinomialNB()
model_svc = LinearSVC()

def train_and_evaluate(model, model_name):
    print(f"\nEvaluating model {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # ROC Curve and AUC
    if hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    elif hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=4)
    auc_score = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, auc_score, model_name)

def plot_roc_curve(fpr, tpr, auc_score, model_name):
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

# Train and evaluate models
train_and_evaluate(model_logistic, 'Logistic Regression')
train_and_evaluate(model_nb, 'Naive Bayes')
train_and_evaluate(model_svc, 'Linear SVC')

# Plotting ROC curves
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Save the ROC curve plot
roc_curve_path = '/Users/paigeleeseberg/Downloads/Python-Projects/Twitter_Sentiment_Analysis/roc_curve.png'
plt.savefig(roc_curve_path)
print(f"ROC curve saved to {roc_curve_path}")

plt.show()

# Evaluate feature importance
print("\nEvaluating Feature Importance for Logistic Regression...")
coef_logistic = model_logistic.coef_.flatten()
feature_names = vectorizer.get_feature_names_out()
top_features_logistic = sorted(zip(coef_logistic, feature_names), reverse=True)[:10]
print("Top features for Logistic Regression:")
for feature, coef in top_features_logistic:
    print(f"{feature}: {coef:.4f}")

print("\nEvaluating Feature Importance for Naive Bayes...")
# Naive Bayes does not have a straightforward way to get feature importances like Logistic Regression

print("\nEvaluating Feature Importance for Linear SVC...")
coef_svc = model_svc.coef_.flatten()
top_features_svc = sorted(zip(coef_svc, feature_names), reverse=True)[:10]
print("Top features for Linear SVC:")
for feature, coef in top_features_svc:
    print(f"{feature}: {coef:.4f}")
