"""
modelling.py (MLProject version)
=================================
Pelatihan model Machine Learning untuk MLflow Project.
Dijalankan melalui `mlflow run` dalam workflow CI.

Author: Sony Alfauzan
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import json
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_path: str) -> tuple:
    """Memuat data preprocessing dan memisahkan fitur dan target."""
    df = pd.read_csv(data_path)
    df = df.dropna().reset_index(drop=True)
    X = df.drop('quality_category', axis=1)
    y = df['quality_category'].astype(int)
    return X, y


def create_confusion_matrix_plot(y_true, y_pred, labels, save_path):
    """Membuat confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_feature_importance_plot(model, feature_names, save_path):
    """Membuat feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(feature_names)), importances[indices], color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Wine Quality Model')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    # Memuat data
    data_path = os.path.join(os.path.dirname(__file__), 'wine_quality_preprocessing.csv')
    X, y = load_preprocessed_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    feature_names = X_train.columns.tolist()
    class_labels = ['Low', 'Medium', 'High']

    print("=" * 60)
    print("PELATIHAN MODEL - MLflow Project (CI)")
    print("=" * 60)

    # Aktifkan MLflow autolog - mencatat model, parameter, metrik, dan artefak otomatis
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
    )
    print("MLflow autolog aktif.")

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artifacts
        artifacts_dir = "mlflow_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        # Confusion matrix
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        create_confusion_matrix_plot(y_test, y_pred, class_labels, cm_path)
        mlflow.log_artifact(cm_path, "plots")

        # Feature importance
        fi_path = os.path.join(artifacts_dir, "feature_importance.png")
        create_feature_importance_plot(model, feature_names, fi_path)
        mlflow.log_artifact(fi_path, "plots")

        # Classification report
        report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
        report_path = os.path.join(artifacts_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, "reports")

        # Tags
        mlflow.set_tag("author", "Sony Alfauzan")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "Wine Quality")

        print(f"\nAccuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"\nRun ID: {run.info.run_id}")
        print("Training selesai!")


if __name__ == "__main__":
    main()
