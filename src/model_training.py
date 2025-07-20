# src/model_training.py

import pandas as pd
import numpy as np
import os
import joblib # For saving/loading models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, confusion_matrix,
    roc_auc_score, precision_score, recall_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE # For handling class imbalance

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src.utils is importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data, save_data, clean_dataframe_columns # Import the new function

def prepare_data_for_modeling(df, target_column):
    """
    Separates features (X) and target (y) from the DataFrame.
    Also cleans column names for LightGBM compatibility.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        tuple: (X, y) where X is the features DataFrame and y is the target Series.
    """
    print(f"--- Preparing data for modeling. Target: '{target_column}' ---")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    
    # Clean column names before separating X and y
    df_cleaned_cols = clean_dataframe_columns(df.copy())
    
    # Ensure target column name is also cleaned if it contained special characters
    cleaned_target_column = clean_dataframe_columns(pd.DataFrame(columns=[target_column])).columns[0]
    
    X = df_cleaned_cols.drop(columns=[cleaned_target_column])
    y = df_cleaned_cols[cleaned_target_column]
    print(f"Features shape (X): {X.shape}, Target shape (y): {y.shape}")
    return X, y

def perform_train_test_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Performs a train-test split on the dataset.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generator for reproducibility.
        stratify (bool): If True, data is split in a stratified fashion, using y as the class labels.
                         Recommended for imbalanced datasets.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"--- Performing Train-Test Split (test_size={test_size}, stratify={stratify}) ---")
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"y_train class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"y_test class distribution:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test

def handle_class_imbalance_smote(X_train, y_train, random_state=42, sampling_strategy='auto'):
    """
    Applies SMOTE to the training data to handle class imbalance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        random_state (int): Seed for random number generator for reproducibility.
        sampling_strategy (str or float or dict): Sampling strategy for SMOTE.
                                                  'auto' balances classes.

    Returns:
        tuple: (X_train_resampled, y_train_resampled)
    """
    print(f"--- Handling Class Imbalance with SMOTE (sampling_strategy='{sampling_strategy}') ---")
    print(f"Original y_train distribution:\n{y_train.value_counts()}")
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled X_train shape: {X_train_resampled.shape}, y_train shape: {y_train_resampled.shape}")
    print(f"Resampled y_train distribution:\n{y_train_resampled.value_counts()}")
    return X_train_resampled, y_train_resampled

def train_model(model, X_train, y_train, model_name="Model"):
    """
    Trains a given machine learning model.

    Args:
        model: The scikit-learn compatible model object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_name (str): Name of the model for printing.

    Returns:
        The trained model object.
    """
    print(f"--- Training {model_name} ---")
    model.fit(X_train, y_train)
    print(f"{model_name} training complete.")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model using various metrics suitable for imbalanced data.

    Args:
        model: The trained scikit-learn compatible model object.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for printing results.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print(f"--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_prob) if y_prob is not None else np.nan
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'AUC-PR': auc_pr,
        'ROC-AUC': roc_auc,
        'Confusion Matrix': cm
    }

    print(f"\n{model_name} Evaluation Metrics:")
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"- {metric}: {value:.4f}")
    print(f"- Confusion Matrix:\n{cm}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    plt.show()

    # Plot Precision-Recall Curve
    if y_prob is not None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        disp_pr = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax)
        ax.set_title(f'{model_name} Precision-Recall Curve (AUC-PR: {auc_pr:.2f})')
        plt.show()
    
    return metrics

def save_model(model, model_path):
    """
    Saves a trained model to a file using joblib.

    Args:
        model: The trained model object.
        model_path (str): The path where the model should be saved.
    """
    try:
        joblib.dump(model, model_path)
        print(f"Model saved successfully to '{model_path}'.")
    except Exception as e:
        print(f"Error saving model to '{model_path}': {e}")
        raise

def load_trained_model(model_path):
    """
    Loads a trained model from a file using joblib.

    Args:
        model_path (str): The path from which the model should be loaded.

    Returns:
        The loaded model object.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return None
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        raise

if __name__ == '__main__':
    # This block demonstrates the full modeling pipeline for one dataset.
    # For a full project, you'd run this for both datasets in the notebook.

    data_dir = '../data'
    processed_data_dir = os.path.join(data_dir, 'processed')
    
    fraud_data_path = os.path.join(processed_data_dir, 'fraud_data_final_engineered_scaled.csv')
    
    # Load the processed data
    try:
        fraud_df = load_data(fraud_data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run 02_Feature_Engineering.ipynb to generate processed data.")
        sys.exit(1) # Exit if data not found

    # Drop non-feature columns that are not needed for modeling
    # Ensure these columns exist before dropping
    columns_to_drop_fraud = ['user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
    fraud_df_cleaned_for_model = fraud_df.drop(columns=[col for col in columns_to_drop_fraud if col in fraud_df.columns], errors='ignore')

    # 1. Prepare Data (includes column cleaning now)
    X_fraud, y_fraud = prepare_data_for_modeling(fraud_df_cleaned_for_model, target_column='class')

    # 2. Perform Train-Test Split
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = perform_train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=True
    )

    # 3. Handle Class Imbalance (SMOTE) on training data
    X_train_fraud_resampled, y_train_fraud_resampled = handle_class_imbalance_smote(
        X_train_fraud, y_train_fraud, random_state=42
    )

    # 4. Train Logistic Regression Model
    lr_model_fraud = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000) # Use class_weight='balanced' is an option, but SMOTE is applied
    lr_model_fraud = train_model(lr_model_fraud, X_train_fraud_resampled, y_train_fraud_resampled, model_name="Logistic Regression (Fraud Data)")

    # 5. Evaluate Logistic Regression Model
    lr_metrics_fraud = evaluate_model(lr_model_fraud, X_test_fraud, y_test_fraud, model_name="Logistic Regression (Fraud Data)")

    # 6. Train LightGBM Model
    lgbm_model_fraud = LGBMClassifier(random_state=42, n_estimators=1000, learning_rate=0.05, num_leaves=31,
                                      objective='binary', metric='aucpr', is_unbalance=True,
                                      colsample_bytree=0.7, subsample=0.7) # Added some regularization
    lgbm_model_fraud = train_model(lgbm_model_fraud, X_train_fraud_resampled, y_train_fraud_resampled, model_name="LightGBM (Fraud Data)")

    # 7. Evaluate LightGBM Model
    lgbm_metrics_fraud = evaluate_model(lgbm_model_fraud, X_test_fraud, y_test_fraud, model_name="LightGBM (Fraud Data)")

    # 8. Save Models (Example)
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    save_model(lr_model_fraud, os.path.join(models_dir, 'lr_fraud_model.joblib'))
    save_model(lgbm_model_fraud, os.path.join(models_dir, 'lgbm_fraud_model.joblib'))

    print("\n--- Modeling Pipeline Demonstration Complete for Fraud Data ---")
    print("Check the 'models/' directory for saved models.")