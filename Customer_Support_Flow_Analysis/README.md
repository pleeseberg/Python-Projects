{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Customer Support Flow Analysis\
\
## 1. Data Collection\
- **Obtain the Dataset:** Download customer support ticket data from a reliable source. Ensure the dataset includes relevant columns such as `ticket_id`, `created_time`, `closed_time`, `assigned_agent`, `escalated`, `resolved_within_sla`, `issue_type`, and feedback text columns like `pros_text` and `cons_text`.\
\
## 2. Data Preparation\
### Summary of Execution:\
- **Initial Data Load:**\
  - Successfully loaded the dataset containing customer feedback and ticket resolution data.\
- **Data Cleaning:**\
  - Replaced missing values in numeric columns like `ease_of_use`, `customer_service`, `value_for_money`, and `features` with mean values.\
  - Filled missing text feedback fields (`pros_text` and `cons_text`) with "No feedback".\
  - Dropped rows with missing values in the target column (`likelihood_to_recommend`).\
- **Feature Engineering:**\
  - Created a binary target variable (`recommend_flag`) to indicate whether the user recommends the product (based on `likelihood_to_recommend`).\
\
## 3. Model Implementation\
### Summary of Model Training:\
- **Logistic Regression:** \
  - Aimed to predict the likelihood of a recommendation based on features such as customer service, ease of use, and more.\
  - Achieved a **ROC AUC** of 0.84 with reasonable precision and recall for both classes.\
  \
- **Random Forest Classifier:** \
  - Tuned using cross-validation to optimize hyperparameters.\
  - Achieved an **ROC AUC** of 0.98, demonstrating strong classification performance with high precision and recall for both classes.\
\
- **XGBoost Classifier:**\
  - Achieved a **ROC AUC** of 0.98, similar to the Random Forest model. It offers high performance in handling large, complex datasets.\
\
### Model Evaluation:\
- **Logistic Regression:**\
  - **Precision:** 0.75\
  - **Recall:** 0.75\
  - **F1-Score:** 0.75\
  - **ROC AUC:** 0.84\
  \
- **Random Forest:**\
  - **Precision:** 0.93\
  - **Recall:** 0.93\
  - **F1-Score:** 0.93\
  - **ROC AUC:** 0.98\
\
- **XGBoost:**\
  - **Precision:** 0.93\
  - **Recall:** 0.94\
  - **F1-Score:** 0.93\
  - **ROC AUC:** 0.98\
\
## 4. Data Visualization\
### Summary of Each Plot\
\
- **Logistic Regression ROC Curve:**\
  - **Description:** Displays the ROC curve for the Logistic Regression model, showing how well it differentiates between the positive and negative classes.\
  - **Purpose:** Helps evaluate the overall performance of the model.\
  - ![Logistic Regression ROC Curve](plots/roc_curves.png)\
\
- **Random Forest ROC Curve:**\
  - **Description:** Displays the ROC curve for the Random Forest model.\
  - **Purpose:** Similar to the Logistic Regression curve, it allows for performance comparison.\
  - ![Random Forest ROC Curve](plots/roc_curves.png)\
\
- **XGBoost ROC Curve:**\
  - **Description:** Displays the ROC curve for the XGBoost model.\
  - **Purpose:** Evaluates the ability of the XGBoost model to separate classes.\
  - ![XGBoost ROC Curve](plots/roc_curves.png)\
\
- **SHAP Summary Plot:**\
  - **Description:** Provides a summary of the feature importances according to SHAP values.\
  - **Purpose:** Helps interpret the model by showing which features contribute most to the predictions.\
  - ![SHAP Summary Plot](plots/shap_summary_plot.png)\
\
- **Feature Correlation Heatmap:**\
  - **Description:** Displays correlations between the features in the dataset.\
  - **Purpose:** Helps identify strong relationships between the features.\
  - ![Feature Correlation Heatmap](plots/correlation_heatmap.png)\
\
- **Sentiment Analysis Boxplots:**\
  - **Description:** Shows how sentiment from the `pros_text` and `cons_text` fields varies by recommendation.\
  - **Purpose:** Provides insights into how positive or negative sentiments correlate with customer recommendations.\
  - ![Pros Sentiment Boxplot](plots/pros_sentiment_boxplot.png)\
  - ![Cons Sentiment Boxplot](plots/cons_sentiment_boxplot.png)\
\
- **Feature Importance Plot for Random Forest:**\
  - **Description:** Displays the feature importance scores for the Random Forest model, showing the relative importance of each feature in the prediction task.\
  - **Purpose:** Helps understand which features the Random Forest model relies on most.\
  - ![Random Forest Feature Importance](plots/random_forest_feature_importance.png)\
\
- **Feature Importance Plot for XGBoost:**\
  - **Description:** Displays the feature importance scores for the XGBoost model, providing insight into which features were most influential for the model's predictions.\
  - **Purpose:** Helps identify the most important features in the XGBoost model.\
  - ![XGBoost Feature Importance](plots/xgboost_feature_importance.png)\
\
- **Confusion Matrix for XGBoost:**\
  - **Description:** Displays the confusion matrix for the XGBoost model, showing the true positives, true negatives, false positives, and false negatives.\
  - **Purpose:** Helps evaluate the accuracy of the model's predictions.\
  - ![Confusion Matrix](plots/confusion_matrix_xgb.png)\
\
## 5. Conclusion\
The **Customer Support Flow Analysis** project utilizes various machine learning techniques to predict the likelihood of a customer recommending a product based on their feedback. We used **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers to build predictive models, achieving strong results with **ROC AUC scores above 0.84**.\
\
- **Logistic Regression** provides a simple, interpretable model with reasonable performance.\
- **Random Forest** and **XGBoost** offer more powerful, complex models with higher performance and **ROC AUC scores above 0.97**.\
\
The project also leverages **SHAP values** to interpret model predictions, providing insights into the most important features for classification. Visualizations such as **ROC curves** and **feature importance plots** further help to understand the model's decision-making process.\
\
### Key Outcomes:\
- **Model performance:** Random Forest and XGBoost performed exceptionally well.\
- **Feature importance:** SHAP values helped us identify key drivers of the model's predictions.\
- **Sentiment analysis:** Provided insights into how customer sentiments influence recommendations.\
\
Moving forward, further exploration into **social network analysis** and **process mining** can provide a deeper understanding of the support ticket workflow and the relationships between agents in resolving tickets.\
\
## 6. Future Enhancements\
- **Process Mining:** To discover ticket resolution workflows and identify inefficiencies.\
- **Social Network Analysis:** To examine agent escalation paths and identify key influencers in customer support.\
- **Deep Learning Models:** Exploring deep learning-based models for improved prediction accuracy.\
}