# Customer Support Flow Analysis

## 1. Data Collection
- **Dataset Acquisition:** Obtain customer support ticket data from a reliable source. The dataset should include relevant columns like `ticket_id`, `created_time`, `closed_time`, `assigned_agent`, `escalated`, `resolved_within_sla`, `issue_type`, and feedback columns such as `pros_text` and `cons_text`.

## 2. Data Preparation
### Execution Summary:
- **Data Loading:**
  - The dataset was successfully loaded, containing both customer feedback and ticket resolution information.
- **Data Cleaning:**
  - Missing values in numeric columns like `ease_of_use`, `customer_service`, `value_for_money`, and `features` were replaced with mean values.
  - Empty text feedback fields (`pros_text` and `cons_text`) were populated with the placeholder "No feedback".
  - Rows with missing values in the target column (`likelihood_to_recommend`) were dropped to ensure model reliability.
- **Feature Engineering:**
  - A new binary variable (`recommend_flag`) was created to indicate whether a customer would recommend the product, based on the `likelihood_to_recommend` score.

## 3. Model Implementation
### Model Training Overview:
- **Logistic Regression:**
  - This model aimed to predict customer recommendation likelihood based on various features like customer service, ease of use, etc.
  - The model achieved a **ROC AUC** score of 0.84, with balanced precision and recall for both positive and negative classes.

- **Random Forest Classifier:**
  - Hyperparameters were optimized using cross-validation.
  - The model achieved a **ROC AUC** score of 0.98, reflecting excellent performance with high precision and recall across both classes.

- **XGBoost Classifier:**
  - Similar to the Random Forest model, XGBoost achieved a **ROC AUC** score of 0.98, proving its ability to handle large datasets effectively while maintaining strong performance.

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
### Plot Overview:

- **Logistic Regression ROC Curve:**
  - **Description:** The ROC curve for the Logistic Regression model illustrates its performance in distinguishing between positive and negative classes.
  - **Purpose:** Evaluates the overall model performance.
  - ![Logistic Regression ROC Curve](plots/roc_curves.png)

- **Random Forest ROC Curve:**
  - **Description:** The ROC curve for the Random Forest model.
  - **Purpose:** Enables performance comparison with Logistic Regression.
  - ![Random Forest ROC Curve](plots/roc_curves.png)

- **XGBoost ROC Curve:**
  - **Description:** The ROC curve for the XGBoost model.
  - **Purpose:** Assesses the ability of XGBoost to separate classes effectively.
  - ![XGBoost ROC Curve](plots/roc_curves.png)

- **SHAP Summary Plot:**
  - **Description:** The SHAP summary plot shows the contribution of each feature to the model's predictions.
  - **Purpose:** Provides valuable insights into the model’s decision-making process.
  - ![SHAP Summary Plot](plots/shap_summary_plot.png)

- **Feature Correlation Heatmap:**
  - **Description:** The heatmap displays the correlation between features in the dataset.
  - **Purpose:** Helps identify strong feature relationships that may impact model performance.
  - ![Feature Correlation Heatmap](plots/correlation_heatmap.png)

- **Sentiment Analysis Boxplots:**
  - **Description:** Boxplots display sentiment scores from `pros_text` and `cons_text`, segmented by recommendation status.
  - **Purpose:** Analyzes how customer sentiment influences their likelihood to recommend the product.
  - ![Pros Sentiment Boxplot](plots/pros_sentiment_boxplot.png)
  - ![Cons Sentiment Boxplot](plots/cons_sentiment_boxplot.png)

- **Feature Importance Plot for Random Forest:**
  - **Description:** This plot visualizes the relative importance of each feature in the Random Forest model.
  - **Purpose:** Highlights which features are most influential in prediction.
  - ![Random Forest Feature Importance](plots/random_forest_feature_importance.png)

- **Feature Importance Plot for XGBoost:**
  - **Description:** This plot shows the feature importance for the XGBoost model.
  - **Purpose:** Identifies the most critical features for the model's predictions.
  - ![XGBoost Feature Importance](plots/xgboost_feature_importance.png)

- **Confusion Matrix for XGBoost:**
  - **Description:** The confusion matrix for the XGBoost model, showing true positives, true negatives, false positives, and false negatives.
  - **Purpose:** Assesses the accuracy of the model's predictions.
  - ![Confusion Matrix](plots/confusion_matrix_xgb.png)

## 5. Conclusion
The **Customer Support Flow Analysis** project applied several machine learning techniques to predict the likelihood of customers recommending a product based on their feedback. We leveraged **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers, achieving strong results with **ROC AUC scores above 0.84**.

- **Logistic Regression** offers a simple, interpretable model with adequate performance.
- **Random Forest** and **XGBoost** are more complex, providing higher performance and **ROC AUC scores above 0.97**.

The analysis also utilized **SHAP values** to interpret model predictions, offering insights into the most important features. Visualizations, including **ROC curves** and **feature importance plots**, further enhanced understanding of model decisions.

### Key Insights:
- **Model Performance:** Random Forest and XGBoost demonstrated excellent performance.
- **Feature Importance:** SHAP values helped identify key predictors in the models.
- **Sentiment Analysis:** Revealed how sentiment influences customer recommendations.

For future work, **social network analysis** and **process mining** can deepen insights into support workflows and agent interactions during ticket resolution.

## 6. Future Enhancements
- **Process Mining:** Discover ticket resolution workflows and pinpoint inefficiencies.
- **Social Network Analysis:** Investigate agent escalation paths and identify key influencers in customer support.
- **Deep Learning Models:** Explore deep learning techniques for even more accurate predictions.
