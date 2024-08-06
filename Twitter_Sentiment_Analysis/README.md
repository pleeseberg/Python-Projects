**Confusion Matrices:**

> <img src="results/confusion_matrix.png" alt="Confusion Matrix" width="300" height="300">  
> Confusion matrices for each model, showing true positives, true negatives, false positives, and false negatives.

**ROC Curves:**

> <img src="results/roc_curve.png" alt="ROC Curve" width="300" height="300">  
> ROC curves for each model, demonstrating their ability to distinguish between classes. AUC scores indicate model performance.

### Model Performance Summary

- **Logistic Regression:**  
  - Cross-Validation Accuracy: 0.7563  
  - Tuned Accuracy: 0.7560  
  - AUC: 0.84

- **Naive Bayes:**  
  - Cross-Validation Accuracy: 0.7444  
  - Tuned Accuracy: 0.7462  
  - AUC: 0.83

- **Linear SVC:**  
  - Cross-Validation Accuracy: 0.7518  
  - Tuned Accuracy: 0.7553  
  - AUC: 0.84

## Feature Analysis

**Feature Importances:**

> **Logistic Regression Top Features:**  
> <img src="results/feature_importances.png" alt="Feature Importances" width="300" height="300">  
> Bar charts showing the top 10 features for Logistic Regression.

> **Naive Bayes Top Features:**  
> Top features with negative coefficients.

> **Linear SVC Top Features:**  
> Top features showing their importance in the Linear SVC model.

## Visualization and Interpretation

**Word Cloud:**

> <img src="results/word_cloud.png" alt="Word Cloud" width="300" height="300">  
> Visualization of the most frequent words in the tweet text.

**Top TF-IDF Scores:**

> <img src="results/top_tfidf_scores.png" alt="Top TF-IDF Scores" width="300" height="300">  
> Bar chart showing the top TF-IDF scores for terms in the dataset.

**Performance Metrics Comparison:**

> <img src="results/performance_metrics_comparison.png" alt="Performance Metrics Comparison" width="300" height="300">  
> Comparison of accuracy, precision, recall, and F1-score across the models.
