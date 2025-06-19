# Customer Support Flow Analysis

## 1. Data Collection
- **Dataset Acquisition:** The dataset used in this analysis was sourced from [Capterra Reviews](https://www.capterra.com/). It includes key columns such as `ticket_id`, `created_time`, `closed_time`, `assigned_agent`, `escalated`, `resolved_within_sla`, `issue_type`, and feedback text fields like `pros_text` and `cons_text`.

## 2. Data Preparation
### Execution Summary:
- **Data Loading:**
  - The dataset was successfully loaded, containing essential customer feedback and ticket resolution data.
- **Data Cleaning:**
  - Missing values in numeric columns like `ease_of_use`, `customer_service`, `value_for_money`, and `features` were replaced with mean values to maintain consistency.
  - Empty text feedback fields (`pros_text` and `cons_text`) were populated with "No feedback" to ensure that no feedback data was omitted.
  - Rows with missing values in the target column (`likelihood_to_recommend`) were dropped to avoid bias in model training.
- **Feature Engineering:**
  - A new binary variable (`recommend_flag`) was created to classify whether a customer would recommend the product, based on their `likelihood_to_recommend` score.

## 3. Model Implementation
### Model Training Overview:
- **Logistic Regression:**
  - The Logistic Regression model was trained to predict customer recommendation likelihood based on various features such as `customer_service`, `ease_of_use`, and `value_for_money`.
  - The model achieved a **ROC AUC** score of 0.84, indicating a solid but less complex model that provides reasonably balanced precision and recall for both positive and negative recommendations.

- **Random Forest Classifier:**
  - The Random Forest Classifier was optimized using cross-validation for better hyperparameter tuning.
  - The model achieved an impressive **ROC AUC** score of 0.98, reflecting superior performance with high precision and recall.

- **XGBoost Classifier:**
  - Similar to the Random Forest model, XGBoost also reached a **ROC AUC** of 0.98. It showed excellent performance in handling large, complex datasets.

### Model Evaluation:
- **Logistic Regression:**
  - **Precision:** 0.75
  - **Recall:** 0.75
  - **F1-Score:** 0.75
  - **ROC AUC:** 0.84

- **Random Forest:**
  - **Precision:** 0.93
  - **Recall:** 0.93
  - **F1-Score:** 0.93
  - **ROC AUC:** 0.98

- **XGBoost:**
  - **Precision:** 0.93
  - **Recall:** 0.94
  - **F1-Score:** 0.93
  - **ROC AUC:** 0.98

## 4. Data Visualization
### Plot Overview and Discussion:

1. **Logistic Regression ROC Curve:**
   - **Description:** The ROC curve for the Logistic Regression model illustrates the model's performance in distinguishing between positive and negative recommendations. A **ROC AUC** of 0.84 shows a decent model, suitable for simpler binary classification tasks.
   - ![Logistic Regression ROC Curve](plots/roc_curves.png)

2. **Random Forest ROC Curve:**
   - **Description:** This ROC curve shows the performance of the Random Forest model. With a **ROC AUC** of 0.98, the model demonstrates exceptional performance in classifying recommendations.
   - ![Random Forest ROC Curve](plots/roc_curves.png)

3. **XGBoost ROC Curve:**
   - **Description:** The ROC curve for the XGBoost model demonstrates its ability to separate the classes, achieving a **ROC AUC** of 0.98.
   - ![XGBoost ROC Curve](plots/roc_curves.png)

4. **SHAP Summary Plot:**
   - **Description:** This plot summarizes feature importance based on SHAP (Shapley Additive exPlanations) values. It shows how each feature influences the model's prediction.
   - ![SHAP Summary Plot](plots/shap_summary_plot.png)

5. **Feature Correlation Heatmap:**
   - **Description:** The heatmap shows the correlations between features in the dataset. It helps to identify relationships between features like `customer_service` and `ease_of_use`.
   - ![Feature Correlation Heatmap](plots/correlation_heatmap.png)

6. **Sentiment Analysis Boxplots:**
   - **Description:** These boxplots display sentiment analysis for both `pros_text` and `cons_text` fields, broken down by customer recommendation status. Positive sentiment in the `pros_text` is often linked to higher likelihood to recommend the product.
   - ![Pros Sentiment Boxplot](plots/pros_sentiment_boxplot.png)
   - ![Cons Sentiment Boxplot](plots/cons_sentiment_boxplot.png)

7. **Feature Importance Plot for Random Forest:**
   - **Description:** This plot visualizes the feature importance scores based on the Random Forest model, showing which features are most influential in predicting whether a customer will recommend the product.
   - ![Random Forest Feature Importance](plots/random_forest_feature_importance.png)

8. **Feature Importance Plot for XGBoost:**
   - **Description:** Similar to the Random Forest plot, this one visualizes feature importance for XGBoost, highlighting the most influential features.
   - ![XGBoost Feature Importance](plots/xgboost_feature_importance.png)

9. **Confusion Matrix for XGBoost:**
   - **Description:** The confusion matrix shows how well the XGBoost model predicts customer recommendations by displaying true positives, false positives, and other metrics.
   - ![Confusion Matrix](plots/confusion_matrix_xgb.png)

## 5. Conclusion
The **Customer Support Flow Analysis** project applied machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost** to predict whether a customer would recommend a product. The **Random Forest** and **XGBoost** models performed exceptionally well, with **ROC AUC scores above 0.97**. 

- **Logistic Regression** provides a simpler, more interpretable model, achieving a **ROC AUC** of 0.84, suitable for less complex applications.
- **Random Forest** and **XGBoost** are more advanced models that offer superior predictive accuracy with **ROC AUC scores of 0.98**.

By leveraging **SHAP values**, we gained insights into the features that most influence customer recommendations, and the visualizations helped further interpret the models' decision-making processes.

### Key Insights:
- **Model Performance:** Random Forest and XGBoost demonstrated superior classification capabilities with high ROC AUC scores.
- **Feature Importance:** SHAP values revealed the most influential features for predicting recommendations.
- **Sentiment Analysis:** Revealed a clear connection between sentiment in customer feedback and the likelihood of recommending a product.

## 6. Future Enhancements
- **Process Mining:** To identify inefficiencies in the ticket resolution process and optimize workflows.
- **Social Network Analysis:** To investigate agent escalation paths and identify key influencers within the customer support process.
- **Deep Learning Models:** To explore the application of deep learning techniques for more accurate predictions.

---

The dataset used in this analysis is available for download from [this link](https://www.capterra.com/) for those who wish to explore it further.
