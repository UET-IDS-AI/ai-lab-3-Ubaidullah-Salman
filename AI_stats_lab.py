"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():

    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions & Metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6: Top 3 features by absolute coefficient magnitude
    coefficients = model.coef_
    abs_coefficients = np.abs(coefficients)
    top_3_feature_indices = np.argsort(abs_coefficients)[-3:][::-1].tolist()

    return (
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices
    )


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: 5-fold cross-validation
    model = LinearRegression()
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    # STEP 4: Mean & Std
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions & Metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Confusion matrix (computed but not returned)
    cm = confusion_matrix(y_test, y_test_pred)

    return (
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    )


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train for different C values
    C_values = [0.01, 0.1, 1, 10, 100]
    results_dictionary = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        results_dictionary[C] = (train_accuracy, test_accuracy)

    return results_dictionary


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: 5-fold CV
    model = LogisticRegression(C=1, max_iter=5000)
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    # STEP 4: Mean & Std
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    return mean_accuracy, std_accuracy
