# %% [markdown]
# # Different approaches to anomaly detection

# %% [markdown]
# This Python notebook proposes three different approaches for anomaly detection in network flows. In addition, their advantages and disadvantages are detailed.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from river import anomaly
from river import preprocessing
from river import optim
from river import sketch
import re

# %%
# Training flows for the online learning algorithm
FLOWS_TRAIN_SCALER = 1000
FLOWS_TRAIN_OML = 100000


# %% [markdown]
# This dataset was created based on one of the datasets provided by Proactivanet. Subsequently, the following anomalies were introduced, with 2,500 flows of each type:
# 
# - Benign IP connecting to an anomalous IP during working hours
# - Benign IP connecting to a benign IP at an anomalous time
# - Benign IP connecting to an anomalous domain during working hours
# - Benign IP connecting to an anomalous domain at an anomalous time
# - Anomalous IP connecting to an anomalous IP during working hours

# %% [markdown]
# ## Exact Match Detector

# %%
DATASET_PATH_DATASET2 = './dataset/'
DATASET_NAME_DATASET2 = 'dataset_anonymized.csv'

# %%
# Read CSV
dataset = pd.read_csv(DATASET_PATH_DATASET2 + DATASET_NAME_DATASET2, sep=',')

# Use the following features
dataset = dataset[['FIRST_SWITCHED', 'IPV_SRC_ADDR', 'L_SRC_PORT', 'IPV_DST_ADDR', 'DIRECTION', 'L_DST_PORT', 'IN_BYTES', 'OUT_BYTES', 'Label']]

# The FIRST_SWITCHED feature is converted into the hour of the day
dataset['FIRST_SWITCHED'] = pd.to_datetime(dataset['FIRST_SWITCHED']).dt.hour

# A PORT feature is created that is L_SRC_PORT if DIRECTION is 0 and L_DST_PORT if DIRECTION is 1
dataset['PORT'] = np.where(dataset['DIRECTION'] == 0, dataset['L_SRC_PORT'], dataset['L_DST_PORT'])

# Remove the columns L_SRC_PORT, L_DST_PORT, and DIRECTION
dataset = dataset.drop(columns=['L_SRC_PORT', 'L_DST_PORT', 'DIRECTION'])

# Preprocess IN_BYTES and OUT_BYTES by rounding to the nearest multiple of 100
dataset['IN_BYTES'] = dataset['IN_BYTES'].apply(lambda x: round(x, -2))
dataset['OUT_BYTES'] = dataset['OUT_BYTES'].apply(lambda x: round(x, -2))

# Remove label
dataset_sin_etiqueta = dataset.drop(columns=['Label'])



# %%
# Create X_train and X_test for the unsupervised learning algorithms
X_train = dataset_sin_etiqueta[dataset['Label'] == 0].iloc[:100000]
X_test = dataset_sin_etiqueta[dataset['Label'] == 1].iloc[:12500]
X_test = pd.concat([X_test, dataset_sin_etiqueta[dataset['Label'] == 0].iloc[100000:112500]], ignore_index=True)
y_test = np.ones(12500)
y_test = np.concatenate([y_test, np.zeros(12500)])
y_train = np.zeros(100000)

# Print the number of samples in the training and test datasets
print("Number of samples in the training dataset: ", X_train.shape[0])
print("Number of samples in the test dataset: ", X_test.shape[0])
print("Number of anomalies in the test dataset: ", y_test[y_test == 1].shape[0])
print("Number of normal samples in the test dataset: ", y_test[y_test == 0].shape[0])



# %% [markdown]
# ## Unsupervised learning - OCSVM

# %%
# One-Class SVM - Grid search for hyperparameter optimization
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler

# Scale the data using MaxAbsScaler (better for SVM than StandardScaler)
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ | Data scaled using MaxAbsScaler")

# Create the One-Class SVM classifier
clf = OneClassSVM()

# Expanded parameter grid for One-Class SVM
param_grid = {
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    'nu': [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15],
    'degree': [2, 3],
    'coef0': [0.0, 0.1],
    'tol': [1e-4, 1e-3, 1e-2],
}

# Grid search with cross-validation
print("Starting grid search for One-Class SVM...")
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=100,verbose=4)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
print("Best parameters: ", best_params)

# Train final model with best parameters
clf = OneClassSVM(**best_params)
clf.fit(X_train_scaled)

# %%
# Evaluate the One-Class SVM classifier
# One-Class SVM returns: -1 for anomalies (outliers), 1 for inliers (normal)
# Convert to: 1 for anomalies, 0 for normal (benign)
y_pred_raw = clf.predict(X_test_scaled)
y_pred = np.where(y_pred_raw == -1, 1, 0)  # -1 (anomaly) -> 1, 1 (inlier) -> 0

# Calculate metrics
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
false_positive = fp / (fp + tn) if (fp + tn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("False Positive Rate: ", false_positive)
print("TP: ", tp, "TN: ", tn, "FP: ", fp, "FN: ", fn)

