# Sentiment Analysis Project

## Overview

This project focuses on developing a model to classify the sentiment of text data, such as reviews or tweets. The objective is to classify text into sentiment categories (positive, negative, or neutral) using machine learning techniques. The project involves data collection, preprocessing, feature extraction, model training, evaluation, and visualization of results.

## Requirements

1. **Data Collection and Preprocessing:**
   - **Dataset:** Sentiment140 in CSV format.
   - **Text Preprocessing:** Use NLTK and SpaCy for tasks including tokenization, normalization, stop word removal, and stemming/lemmatization.

2. **Feature Extraction:**
   - **Convert Text to Numerical Features:** Implement techniques like Bag of Words or TF-IDF using Python libraries.

## 3. Model Evaluation

**Confusion Matrices:**

> ![Confusion Matrix](Twitter_Sentiment_Analysis/results/confusion_matrix.png)  
> Confusion matrices for each model, showing true positives, true negatives, false positives, and false negatives.

**ROC Curves:**

> ![ROC Curve](Twitter_Sentiment_Analysis/results/roc_curve.png)  
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

## 4. Feature Analysis

**Feature Importances:**

> **Logistic Regression Top Features:**  
> ![Feature Importances](Twitter_Sentiment_Analysis/results/feature_importances.png)  
> Bar charts showing the top 10 features for Logistic Regression.

> **Naive Bayes Top Features:**  
> Top features with negative coefficients.

> **Linear SVC Top Features:**  
> Top features showing their importance in the Linear SVC model.

## 5. Visualization and Interpretation

**Word Cloud:**

> ![Word Cloud](Twitter_Sentiment_Analysis/results/word_cloud.png)  
> Visualization of the most frequent words in the tweet text.

**Top TF-IDF Scores:**

> ![Top TF-IDF Scores](Twitter_Sentiment_Analysis/results/top_tfidf_scores.png)  
> Bar chart showing the top TF-IDF scores for terms in the dataset.

**Performance Metrics Comparison:**

> ![Performance Metrics Comparison](Twitter_Sentiment_Analysis/results/performance_metrics_comparison.png)  
> Comparison of accuracy, precision, recall, and F1-score across the models.

## 6. Documentation and Reporting

The following visualizations and figures provide insights into the performance and analysis of the sentiment analysis models:

- **Word Cloud:**  
  ![Word Cloud](Twitter_Sentiment_Analysis/results/word_cloud.png)  
  This word cloud visualizes the most frequent words in the tweet text, highlighting common terms associated with sentiment.

- **Top TF-IDF Scores:**  
  ![Top TF-IDF Scores](Twitter_Sentiment_Analysis/results/top_tfidf_scores.png)  
  A bar chart displaying the top TF-IDF scores for terms in the dataset, showing which terms have the highest importance.

- **ROC Curves:**  
  ![ROC Curve](Twitter_Sentiment_Analysis/results/roc_curve.png)  
  ROC curves for each model, illustrating their performance in distinguishing between classes, with AUC scores indicating model accuracy.

- **Performance Metrics Comparison:**  
  ![Performance Metrics Comparison](Twitter_Sentiment_Analysis/results/performance_metrics_comparison.png)  
  A comparison of accuracy, precision, recall, and F1-score across Logistic Regression, Naive Bayes, and Linear SVC models.

- **Feature Importances:**  
  ![Feature Importances](Twitter_Sentiment_Analysis/results/feature_importances.png)  
  Bar charts showing the importance of the top features for each model, helping to identify key terms influencing sentiment.

- **Confusion Matrix:**  
  ![Confusion Matrix](Twitter_Sentiment_Analysis/results/confusion_matrix.png)  
  Heatmaps for each model’s confusion matrix, providing a detailed view of the classification results and misclassifications.

## 7. Future Work and Improvements

Future work could include:

- Exploring additional features or models to improve classification performance.
- Conducting further hyperparameter tuning to optimize model performance.
- Analyzing sentiment trends over time or across different contexts.

## Summary

This sentiment analysis project involves loading and processing tweet data, training and evaluating various classification models, and analyzing the results through visualizations and feature importance. The models show competitive performance with opportunities for further enhancements.
