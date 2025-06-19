# phase 1: data preprocessing and feature engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os
from collections import Counter
from sklearn.model_selection import GridSearchCV

# define file path and load data
file_path = '/Users/paigeleeseberg/Downloads/Python-Projects/customer-support-flow-analysis/data/capterra_reviews.csv'
df = pd.read_csv(file_path)

# replace -1 with 0 for binary features
binary_features = df.columns[11:]
df[binary_features] = df[binary_features].replace(-1, 0)

# fill missing values: numeric columns with mean and text columns with 'no feedback'
df[['ease_of_use', 'customer_service', 'value_for_money', 'features']] = df[['ease_of_use', 'customer_service', 'value_for_money', 'features']].fillna(df[['ease_of_use', 'customer_service', 'value_for_money', 'features']].mean())
df.fillna({'pros_text': 'No feedback', 'cons_text': 'No feedback'}, inplace=True)

# create recommendation flag (binary)
df['recommend_flag'] = (df['likelihood_to_recommend'] >= 4).astype(int)

print("target class balance:")
print(df['recommend_flag'].value_counts(normalize=True))

# select features and target variable
features = ['overall_rating', 'ease_of_use', 'customer_service', 'value_for_money', 'features'] + list(binary_features)
X = df[features]
y = df['recommend_flag']

# handle class imbalance using smote
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nresampled class balance:")
print(y_resampled.value_counts(normalize=True))

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# phase 2: model training and evaluation
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print("\nlogistic regression report:")
print(classification_report(y_test, lr_preds))
print("logistic regression roc auc:", roc_auc_score(y_test, lr_probs))

# perform cross-validation to improve model evaluation
lr_cv_score = cross_val_score(lr, X_resampled, y_resampled, cv=5, scoring='roc_auc')
print(f"logistic regression cv roc auc: {lr_cv_score.mean()}")

# random forest classifier with hyperparameter tuning using randomizedsearchcv
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
rf_random_search.fit(X_train, y_train)

print(f"best random forest parameters: {rf_random_search.best_params_}")

rf = RandomForestClassifier(**rf_random_search.best_params_, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\nrandom forest report:")
print(classification_report(y_test, rf_preds))
print("random forest roc auc:", roc_auc_score(y_test, rf_probs))

# xgboost classifier with hyperparameter tuning using randomizedsearchcv
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_random_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), xgb_param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
xgb_random_search.fit(X_train, y_train)

print(f"best xgboost parameters: {xgb_random_search.best_params_}")

xgb_model = XGBClassifier(**xgb_random_search.best_params_, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\nxgboost report:")
print(classification_report(y_test, xgb_preds))
print("xgboost roc auc:", roc_auc_score(y_test, xgb_probs))

# phase 3: shap values for model interpretation
explainer = shap.Explainer(xgb_model, X_train, feature_names=features)
shap_values = explainer(X_test)

# save shap summary plot
output_dir = '/Users/paigeleeseberg/Downloads/Python-Projects/customer-support-flow-analysis/plots'
os.makedirs(output_dir, exist_ok=True)

shap.summary_plot(shap_values.values, X_test, feature_names=features, show=False)
plt.savefig(f"{output_dir}/shap_summary_plot.png", bbox_inches="tight")
plt.show()  # show the plot for debugging before saving
plt.close()

# save shap values and xgboost model
output_model_path = '/Users/paigeleeseberg/Downloads/Python-Projects/customer-support-flow-analysis/outputs'
joblib.dump(xgb_model, os.path.join(output_model_path, 'xgboost_model.joblib'))
np.save(os.path.join(output_model_path, 'shap_values.npy'), shap_values.values)

# phase 4: sentiment analysis (boxplots)
df['pros_sentiment'] = df['pros_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['cons_sentiment'] = df['cons_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# sentiment analysis plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='recommend_flag', y='pros_sentiment')
plt.title("pros sentiment by recommendation")
plt.xticks([0, 1], ['not recommend', 'recommend'])
plt.xlabel("recommendation")
plt.ylabel("pros sentiment")
plt.tight_layout()
plt.savefig(f"{output_dir}/pros_sentiment_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='recommend_flag', y='cons_sentiment')
plt.title("cons sentiment by recommendation")
plt.xticks([0, 1], ['not recommend', 'recommend'])
plt.xlabel("recommendation")
plt.ylabel("cons sentiment")
plt.tight_layout()
plt.savefig(f"{output_dir}/cons_sentiment_boxplot.png")
plt.close()

# correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("feature correlation heatmap - customer support dataset")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# statistics related to the correlation heatmap
corr_matrix = df[features].corr()
print("\ncorrelation matrix:\n", corr_matrix)

# --- feature importance visualization ---
# for random forest
rf_importances = rf.feature_importances_
rf_feature_names = features  # use the same features list as before

# sort the features by importance
rf_sorted_idx = np.argsort(rf_importances)[::-1]
rf_importances_sorted = rf_importances[rf_sorted_idx]
rf_feature_names_sorted = [rf_feature_names[i] for i in rf_sorted_idx]

# plot the random forest feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=rf_importances_sorted, y=rf_feature_names_sorted, palette="viridis")
plt.title("random forest feature importance")
plt.xlabel("importance")
plt.ylabel("features")
plt.tight_layout()
plt.savefig(f"{output_dir}/random_forest_feature_importance.png")
plt.close()

# for xgboost
xgb_importances = xgb_model.feature_importances_

# sort the features by importance for xgboost
xgb_sorted_idx = np.argsort(xgb_importances)[::-1]
xgb_importances_sorted = xgb_importances[xgb_sorted_idx]
xgb_feature_names_sorted = [features[i] for i in xgb_sorted_idx]

# plot the xgboost feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x=xgb_importances_sorted, y=xgb_feature_names_sorted, palette="viridis")
plt.title("xgboost feature importance")
plt.xlabel("importance")
plt.ylabel("features")
plt.tight_layout()
plt.savefig(f"{output_dir}/xgboost_feature_importance.png")
plt.close()

# alternatively, plot xgboost's built-in importance plot (using xgboost's plot_importance method)
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10, title="xgboost feature importance")
plt.tight_layout()
plt.savefig(f"{output_dir}/xgboost_feature_importance_plot.png")
plt.close()

# roc curve for all models
plt.figure(figsize=(10, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.plot(fpr_lr, tpr_lr, color='blue', label='logistic regression')
plt.plot(fpr_rf, tpr_rf, color='green', label='random forest')
plt.plot(fpr_xgb, tpr_xgb, color='red', label='xgboost')

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('roc curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curves.png")
plt.close()

# confusion matrix for xgboost
cm = confusion_matrix(y_test, xgb_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['not recommend', 'recommend'], yticklabels=['not recommend', 'recommend'])
plt.title('confusion matrix for xgboost')
plt.xlabel('predicted')
plt.ylabel('true')
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix_xgb.png")
plt.close()

# print confusion matrix stats
print("\nconfusion matrix for xgboost:")
print(cm)
print(f"true positives: {cm[1, 1]}")
print(f"true negatives: {cm[0, 0]}")
print(f"false positives: {cm[0, 1]}")
print(f"false negatives: {cm[1, 0]}")
