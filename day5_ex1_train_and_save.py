"""
Day 5 (Merged) - Example 1: Model Benchmarking & Artifact Serialization
File: day5_ex1_train_and_save.py

Compares LogisticRegression vs. RandomForestClassifier on the trainer dataset
and all three student group datasets, selects the best-performing model, and exports:
1. `<prefix>_model.pkl`      (Winning trained classifier)
2. `<prefix>_columns.pkl`    (Exact feature column order after one-hot encoding)
3. `<prefix>_importance.pkl` (Feature importance / coefficient weights for dashboard display)

Run:
    python day5_ex1_train_and_save.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


def benchmark_and_export_model(name, csv_path, id_col, cat_cols, target_col, model_prefix, max_iter=500):
    if not os.path.exists(csv_path):
        print(f"[Skipped] Dataset not found: {csv_path}")
        return

    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(csv_path).drop(columns=[id_col])
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    X = encoded.drop(columns=[target_col])
    y = encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=max_iter, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    best_name = None
    best_model = None
    best_score = -1.0

    print(f"=== Benchmarking: {name} ===")
    for algo_name, clf in candidates.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        print(f"   - {algo_name:<20} | Accuracy: {acc:.2%} | Recall: {rec:.2%}")
        if acc > best_score:
            best_score = acc
            best_name = algo_name
            best_model = clf

    # Extract feature importance scores for explainability
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    else:
        importances = pd.Series(np.abs(best_model.coef_[0]), index=X_train.columns).sort_values(ascending=False)

    model_file = f"models/{model_prefix}_model.pkl"
    cols_file = f"models/{model_prefix}_columns.pkl"
    imp_file = f"models/{model_prefix}_importance.pkl"

    joblib.dump(best_model, model_file)
    joblib.dump(X_train.columns.tolist(), cols_file)
    joblib.dump(importances, imp_file)

    print(f"   => Winner: {best_name} ({best_score:.2%}) -> Saved to {model_file}\n")


if __name__ == "__main__":
    # 1. Trainer Live Demo Dataset (Telecom Customer Churn)
    benchmark_and_export_model(
        name="Trainer Model - Telecom Customer Churn",
        csv_path="customer_churn.csv",
        id_col="customer_id",
        cat_cols=["city", "contract_type"],
        target_col="churn",
        model_prefix="churn",
        max_iter=500,
    )

    # # 2. Group A Dataset (E-Commerce Order Returns)
    # benchmark_and_export_model(
    #     name="Group A - E-Commerce Order Returns",
    #     csv_path="project_ecommerce_orders.csv",
    #     id_col="order_id",
    #     cat_cols=["city", "category", "payment_method"],
    #     target_col="is_returned",
    #     model_prefix="ecom",
    #     max_iter=500,
    # )
    #
    # # 3. Group B Dataset (HR Employee Attrition)
    # benchmark_and_export_model(
    #     name="Group B - HR Employee Attrition",
    #     csv_path="project_hr_attrition.csv",
    #     id_col="employee_id",
    #     cat_cols=["department", "education_level"],
    #     target_col="left_company",
    #     model_prefix="hr",
    #     max_iter=1000,
    # )
    #
    # # 4. Group C Dataset (Medical Clinic Appointment No-Show)
    # benchmark_and_export_model(
    #     name="Group C - Medical Clinic No-Show",
    #     csv_path="project_medical_appointments.csv",
    #     id_col="appointment_id",
    #     cat_cols=["clinic_department", "gender"],
    #     target_col="no_show",
    #     model_prefix="med",
    #     max_iter=500,
    # )
