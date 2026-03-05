# Random Forest Project

This project demonstrates a **Random Forest Classification model** using the built-in Breast Cancer dataset from scikit-learn.

The goal is to train a machine learning model, evaluate its performance, and visualize important metrics such as the confusion matrix and feature importance.

## Dataset

The dataset used is the **Breast Cancer Wisconsin Dataset** from scikit-learn.

It contains features computed from digitized images of breast mass cells.  
The task is to classify tumors as **malignant or benign**.


## Libraries Used

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn


## Project Steps

1. Load dataset using `sklearn.datasets`
2. Split data into training and testing sets
3. Train a **RandomForestClassifier**
4. Evaluate the model using:

   * Accuracy
   * Classification Report
   * Confusion Matrix
5. Visualize confusion matrix with **TP, TN, FP, FN**
6. Perform **Cross Validation**
7. Apply **Hyperparameter Tuning (GridSearchCV)**
8. Analyze **Feature Importance**
9. Visualize top important features


## Evaluation Metrics

The project uses several evaluation techniques:

* Accuracy Score
* Classification Report
* Confusion Matrix
* Cross Validation

## Feature Importance

Random Forest provides **feature importance scores**, which show which variables contribute the most to the model’s predictions.

This helps understand the most influential features in the dataset.


**Why Random Forest is better:**

Random Forest combines multiple decision trees and averages their predictions.
This reduces variance, improves generalization, and usually produces **better accuracy than a single Decision Tree**.


## Run the Project

python random_forest.py


## Output

The program prints:

* Accuracy score
* Classification report
* Confusion matrix
* Cross-validation scores
* Best hyperparameters
* Top important features

It also displays:

* Confusion Matrix Heatmap
* Feature Importance Graph

