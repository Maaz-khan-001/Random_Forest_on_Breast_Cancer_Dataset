
# Random Forest Project 


# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# 1: Load Dataset

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("Dataset Shape:", df.shape)
print(df.head())


# 2: Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3: Random Forest Model

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)


# 4: Prediction
y_pred = rf.predict(X_test)

# 5: Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 6: Confusion Matrix Visualization

labels = np.array([
    ["TN", "FP"],
    ["FN", "TP"]
])

# Combine label + value
annot = np.empty_like(cm).astype(str)

for i in range(2):
    for j in range(2):
        annot[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=annot,
    fmt="",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with TP TN FP FN")

plt.show()

# 7: Cross Validation

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())

# 8: Hyperparameter Tuning

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_


# 9: Feature Importance

importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(importance_df.head(10))

# 10: Feature Importance Visualization

plt.figure(figsize=(8,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(10)
)

plt.title("Top 10 Feature Importance")
plt.show()
