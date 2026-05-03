# Customer Support Flow Analysis

## 1. Data Collection
- **Dataset Acquisition:**  
  The dataset used in this analysis was sourced from [Capterra Reviews](https://www.capterra.com/). It includes review-level information and customer support ticket data across various software vendors. Key columns include:
  - `ticket_id`, `created_time`, `closed_time`, `assigned_agent`, `escalated`, `resolved_within_sla`
  - Rating features: `ease_of_use`, `customer_service`, `value_for_money`, `features`, `overall_rating`
  - Feedback text fields: `pros_text`, `cons_text`
  - Target: `likelihood_to_recommend`

## 2. Data Preparation
### Execution Summary:

- **Initial Cleaning and Handling Missing Values:**
  - All numeric fields such as `ease_of_use`, `customer_service`, `value_for_money`, and `features` were filled with their respective column means to ensure model input consistency.
  - Binary indicators (columns 11 onward) had `-1` values replaced with `0` to standardize binary representation.
  - Missing text feedback fields were filled with the placeholder `"No feedback"` to avoid dropping rows unnecessarily.
  - Records with missing `likelihood_to_recommend` (target) were excluded to avoid introducing target bias.

- **Feature Engineering:**
  - A binary classification target `recommend_flag` was created:
    - `recommend_flag = 1` if `likelihood_to_recommend ≥ 4`
    - `recommend_flag = 0` otherwise
  - This binarization helps to distinguish between promoters and detractors in a simplified way.

- **Class Imbalance Handling with SMOTE:**
  - The initial dataset showed a slight class imbalance between recommenders and non-recommenders.
  - To prevent biased model learning, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied **before train-test splitting** to balance class distributions.
  - The resampled dataset showed near-equal representation across both classes.

- **Feature Selection:**
  - Selected features included all numeric ratings and binary feedback columns:  
    `['overall_rating', 'ease_of_use', 'customer_service', 'value_for_money', 'features'] + binary_indicators`

- **Train-Test Split:**
  - 80% of the balanced data was used for training and 20% for testing using `train_test_split`.

## 3. Model Implementation
### Model Training Overview:

- **Logistic Regression:**
  - Trained with `max_iter=1000` to ensure convergence.
  - Evaluated using cross-validation (5-fold) and test set performance.
  - AUC-ROC of **0.84** on test data confirms reliable, interpretable performance for baseline classification.

- **Random Forest Classifier:**
  - Hyperparameter tuning performed using **RandomizedSearchCV** over:
    - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`
  - Final model achieved **ROC AUC of 0.98**, indicating high discriminative power.

- **XGBoost Classifier:**
  - Also tuned via **RandomizedSearchCV** with parameters like:
    - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
  - Achieved **ROC AUC of 0.98**, showcasing superior performance, especially in handling complex patterns.

### Model Evaluation:

| Model               | Precision | Recall | F1-Score | ROC AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression | 0.75      | 0.75   | 0.75     | 0.84    |
| Random Forest       | 0.93      | 0.93   | 0.93     | 0.98    |
| XGBoost             | 0.93      | 0.94   | 0.93     | 0.98    |

## 4. Data Visualization
### Plot Overview and Discussion:

1. **Combined ROC Curves:**
   - Plots for all three models show that **Random Forest** and **XGBoost** outperform **Logistic Regression**.
   - A diagonal reference line helps visualize performance over random guessing.
   - ![ROC Curves](plots/roc_curves.png)

2. **SHAP Summary Plot:**
   - Generated using SHAP’s TreeExplainer on XGBoost.
   - Illustrates the magnitude and direction of each feature's impact.
   - Highlights `customer_service`, `ease_of_use`, and binary flags as most influential.
   - ![SHAP Summary](plots/shap_summary_plot.png)

3. **Feature Correlation Heatmap:**
   - Visualizes relationships among numeric and binary variables.
   - Useful for checking multicollinearity and uncovering related features.
   - ![Correlation Heatmap](plots/correlation_heatmap.png)

4. **Sentiment Analysis Boxplots:**
   - **Pros Sentiment:** Customers who recommended the product had higher sentiment scores in `pros_text`.
   - **Cons Sentiment:** Negative sentiment scores are more pronounced for non-recommenders.
   - ![Pros Sentiment](plots/pros_sentiment_boxplot.png)
   - ![Cons Sentiment](plots/cons_sentiment_boxplot.png)

5. **Random Forest Feature Importance:**
   - Shows ranking of features based on mean decrease in impurity.
   - Top features: `customer_service`, `ease_of_use`, and `overall_rating`.
   - ![RF Importance](plots/random_forest_feature_importance.png)

6. **XGBoost Feature Importance:**
   - Similar ranking to Random Forest, but highlights additional binary indicators.
   - Built-in importance and SHAP were both used for interpretation.
   - ![XGBoost Importance](plots/xgboost_feature_importance.png)

7. **XGBoost Confusion Matrix:**
   - High number of true positives and true negatives.
   - Very few false positives/negatives, further reinforcing model reliability.
   - ![Confusion Matrix](plots/confusion_matrix_xgb.png)

## 5. Conclusion

This **Customer Support Flow Analysis** project used structured customer data to predict the likelihood of customer recommendation using three classification models:

- **Logistic Regression** served as a strong baseline with solid interpretability.
- **Random Forest** and **XGBoost** achieved outstanding performance (**ROC AUC 0.98**) and outperformed baseline methods in both recall and precision.
- **SMOTE** successfully addressed the class imbalance issue, ensuring the model performance was not skewed toward the majority class.

Through the use of **SHAP values**, **sentiment analysis**, and **feature importance plots**, the study revealed that factors such as `customer_service`, `ease_of_use`, and `value_for_money` are key drivers in customer satisfaction and recommendation.

### Key Insights:
- **Customer Service and Ease of Use** are the most influential factors in driving product recommendation.
- **Text sentiment** in `pros_text` and `cons_text` aligns strongly with customer decisions.
- **Balanced datasets (via SMOTE)** lead to more reliable model evaluation metrics.

## 6. Future Enhancements

- **Process Mining:**  
  Leverage timestamped ticket data (`created_time`, `closed_time`) to uncover inefficiencies and bottlenecks in support workflows.

- **Social Network Analysis:**  
  Use escalation paths and `assigned_agent` relationships to map influence and workload distribution across agents.

- **Text Embedding Techniques:**  
  Enhance sentiment analysis by applying advanced NLP methods such as TF-IDF or BERT embeddings instead of basic polarity scores.

- **Deep Learning Models:**  
  Evaluate the effectiveness of neural networks for classifying complex patterns, particularly in free-text fields.

---

**Note:**  
Upon auditing this project, I noticed that **SMOTE was applied *before* the train-test split**, which can lead to **data leakage** and overly optimistic performance metrics. I plan to update the analysis after correcting the pipeline to apply SMOTE **only on the training data** after the split. A revised version of this report will be published once the retraining and validation are complete.
