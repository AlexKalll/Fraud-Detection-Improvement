# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_time_features_fraud(df):
    """
    Creates time-based features for the Fraud_Data dataset.

    Features include:
    - hour_of_day: Hour of the purchase transaction.
    - day_of_week: Day of the week of the purchase transaction.
    - time_since_signup: Duration in seconds between signup_time and purchase_time.

    Args:
        df (pandas.DataFrame): The preprocessed Fraud_Data DataFrame,
                               with 'purchase_time' and 'signup_time' as datetime objects.

    Returns:
        pandas.DataFrame: DataFrame with new time-based features.
    """
    print("--- Creating Time-Based Features for E-commerce Fraud Data ---")
    
    # Ensure time columns are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df['purchase_time']):
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['signup_time']):
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')

    # Drop rows where datetime conversion might have failed
    df.dropna(subset=['purchase_time', 'signup_time'], inplace=True)

    # hour_of_day
    df['hour_of_day'] = df['purchase_time'].dt.hour
    print("Created 'hour_of_day'.")

    # day_of_week (Monday=0, Sunday=6)
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    print("Created 'day_of_week'.")

    # time_since_signup (in seconds)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    # Handle cases where signup_time is after purchase_time (e.g., data errors)
    df['time_since_signup'] = df['time_since_signup'].apply(lambda x: max(0, x))
    print("Created 'time_since_signup' (in seconds).")

    print("Time-based feature creation complete.")
    return df


def create_frequency_velocity_features_fraud(df, time_window_seconds=3600):
    """
    Creates transaction frequency and velocity features for the Fraud_Data dataset.

    Features include:
    - user_transaction_count: Total number of transactions by a user up to the current transaction.
    - device_transaction_count: Total number of transactions by a device up to the current transaction.
    - user_velocity: Number of transactions by a user within a specified time window (default 1 hour)
                     prior to the current transaction.
    - device_velocity: Number of transactions by a device within a specified time window (default 1 hour)
                       prior to the current transaction.

    Args:
        df (pandas.DataFrame): The preprocessed Fraud_Data DataFrame,
                               with 'purchase_time' as datetime objects.
        time_window_seconds (int): The time window in seconds for velocity calculation.
                                   Default is 3600 seconds (1 hour).

    Returns:
        pandas.DataFrame: DataFrame with new frequency and velocity features.
    """
    print(f"--- Creating Frequency and Velocity Features for E-commerce Fraud Data (window={time_window_seconds/3600} hours) ---")

    # Ensure 'purchase_time' is datetime and sort by it for accurate cumulative counts
    if not pd.api.types.is_datetime64_any_dtype(df['purchase_time']):
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df.dropna(subset=['purchase_time'], inplace=True) # Drop if conversion failed

    df_sorted = df.sort_values(by='purchase_time').reset_index(drop=False) # Keep original index

    # User transaction count (cumulative count)
    df_sorted['user_transaction_count'] = df_sorted.groupby('user_id').cumcount() + 1
    print("Created 'user_transaction_count'.")

    # Device transaction count (cumulative count)
    df_sorted['device_transaction_count'] = df_sorted.groupby('device_id').cumcount() + 1
    print("Created 'device_transaction_count'.")

    # User velocity (transactions in last `time_window_seconds`)
    def calculate_velocity(group, time_col, window_seconds):
        group = group.sort_values(by=time_col)
        velocities = []
        for i, row in group.iterrows():
            current_time = row[time_col]
            # Count transactions within the window_seconds prior to current_time
            # Exclude the current transaction itself if it's the first in the window
            past_transactions = group[(group[time_col] < current_time) & 
                                      (group[time_col] >= current_time - pd.Timedelta(seconds=window_seconds))]
            velocities.append(len(past_transactions))
        return pd.Series(velocities, index=group.index)

    # Apply to user_id
    # Use transform to ensure alignment with original DataFrame index
    df_sorted['user_velocity'] = df_sorted.groupby('user_id', group_keys=False).apply(
        lambda x: calculate_velocity(x, 'purchase_time', time_window_seconds)
    )
    print("Created 'user_velocity'.")

    # Apply to device_id
    df_sorted['device_velocity'] = df_sorted.groupby('device_id', group_keys=False).apply(
        lambda x: calculate_velocity(x, 'purchase_time', time_window_seconds)
    )
    print("Created 'device_velocity'.")

    # Realign to original index order before returning
    df_sorted = df_sorted.set_index('index').loc[df.index]
    
    print("Frequency and velocity feature creation complete.")
    return df_sorted


def encode_categorical_features(df, categorical_cols, strategy='onehot'):
    """
    Encodes categorical features using One-Hot Encoding.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        categorical_cols (list): A list of column names to encode.
        strategy (str): The encoding strategy. Currently only 'onehot' is supported.

    Returns:
        pandas.DataFrame: DataFrame with categorical features encoded.
        list: List of new column names created by one-hot encoding.
    """
    print(f"--- Encoding Categorical Features using {strategy.upper()} Encoding ---")
    if strategy == 'onehot':
        # Identify columns that actually exist in the DataFrame
        cols_to_encode = [col for col in categorical_cols if col in df.columns]
        if not cols_to_encode:
            print("No specified categorical columns found in DataFrame for encoding.")
            return df, []

        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True) # drop_first to avoid multicollinearity
        # Get the new column names created by get_dummies
        original_cols = set(df.columns)
        new_encoded_cols = [col for col in df_encoded.columns if col not in original_cols]
        print(f"Encoded columns: {cols_to_encode}")
        print(f"New columns created: {len(new_encoded_cols)}")
        return df_encoded, new_encoded_cols
    else:
        print(f"Encoding strategy '{strategy}' not supported.")
        return df, []

def scale_numerical_features(df, numerical_cols, scaler_type='standard'):
    """
    Scales numerical features using StandardScaler or MinMaxScaler.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        numerical_cols (list): A list of column names to scale.
        scaler_type (str): Type of scaler to use ('standard' for StandardScaler,
                           'minmax' for MinMaxScaler).

    Returns:
        pandas.DataFrame: DataFrame with numerical features scaled.
        sklearn.preprocessing.Scaler: The fitted scaler object.
    """
    print(f"--- Scaling Numerical Features using {scaler_type.capitalize()}Scaler ---")

    # Identify columns that actually exist in the DataFrame and are numeric
    cols_to_scale = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not cols_to_scale:
        print("No specified numerical columns found or are not numeric in DataFrame for scaling.")
        return df, None

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        print(f"Scaler type '{scaler_type}' not supported. Returning original DataFrame.")
        return df, None

    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    print(f"Scaled columns: {cols_to_scale}")
    return df_scaled, scaler

if __name__ == '__main__':
    # This block will only run when feature_engineering.py is executed directly.
    # It demonstrates how to use the functions with the cleaned data.
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src')))
    from utils import load_data, save_data
    from data_preprocessing import preprocess_fraud_data, preprocess_creditcard_data, preprocess_ip_to_country_data, merge_fraud_and_ip_data

    data_dir = '../data'
    fraud_cleaned_data_path = os.path.join(data_dir, 'Fraud_Data_Cleaned.csv')
    creditcard_cleaned_data_path = os.path.join(data_dir, 'creditcard_Cleaned.csv')
    ip_to_country_data_path = os.path.join(data_dir, 'IpAddress_to_Country.csv') # This one is still raw

    print("\n--- Loading Cleaned and Raw Data for Feature Engineering Test ---")
    try:
        fraud_cleaned = load_data(fraud_cleaned_data_path)
        creditcard_cleaned = load_data(creditcard_cleaned_data_path)
        ip_country_raw = load_data(ip_to_country_data_path)
    except FileNotFoundError as e:
        print(f"Please ensure all data files are in the '{data_dir}' directory, including the cleaned ones.")
        exit() # Exit if data files are not found

    print("\n--- Preprocessing IP to Country Data (if not already cleaned) ---")
    ip_country_processed = preprocess_ip_to_country_data(ip_country_raw.copy())

    print("\n--- Merging Fraud Data with Country Info ---")
    fraud_merged = merge_fraud_and_ip_data(fraud_cleaned.copy(), ip_country_processed.copy())

    print("\n--- Applying Feature Engineering ---")
    fraud_engineered = create_time_features_fraud(fraud_merged.copy())
    fraud_engineered = create_frequency_velocity_features_fraud(fraud_engineered.copy(), time_window_seconds=3600)

    print("\n--- Feature Engineering Results (Fraud Data) ---")
    print(fraud_engineered.head())
    print(fraud_engineered.info())

    print("\n--- Applying Encoding and Scaling (Demonstration) ---")
    # Example for encoding categorical features
    categorical_cols_to_encode = ['source', 'browser', 'sex', 'country']
    fraud_encoded, new_cols = encode_categorical_features(fraud_engineered.copy(), categorical_cols_to_encode)
    print("\nFraud Data after One-Hot Encoding:")
    print(fraud_encoded.head())
    print(f"New encoded columns: {new_cols}")

    # Example for scaling numerical features
    numerical_cols_to_scale = ['purchase_value', 'age', 'time_since_signup', 
                               'user_transaction_count', 'device_transaction_count', 
                               'user_velocity', 'device_velocity']
    
    numerical_cols_to_scale_existing = [col for col in numerical_cols_to_scale if col in fraud_encoded.columns]
    fraud_scaled, scaler = scale_numerical_features(fraud_encoded.copy(), numerical_cols_to_scale_existing, scaler_type='standard')
    print("\nFraud Data after Standard Scaling (sample of scaled columns):")
    print(fraud_scaled[numerical_cols_to_scale_existing].head())
    print(f"Scaler used: {type(scaler).__name__}")

    # For Credit Card Data, just apply scaling as it's already numerical
    numerical_cols_creditcard = [col for col in creditcard_cleaned.columns if col.startswith('V') or col == 'Amount' or col == 'Time']
    creditcard_scaled, cc_scaler = scale_numerical_features(creditcard_cleaned.copy(), numerical_cols_creditcard, scaler_type='standard')
    print("\nCredit Card Data after Standard Scaling (sample of scaled columns):")
    print(creditcard_scaled[numerical_cols_creditcard].head())
    print(f"Scaler used: {type(cc_scaler).__name__}")