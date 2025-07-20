# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from utils import load_data # Import the utility function

def preprocess_fraud_data(df):
    """
    Performs preprocessing steps specific to the Fraud_Data dataset.
    
    Steps include:
    1. Handling missing values (dropping rows for now, can be extended to imputation).
    2. Removing duplicate rows.
    3. Converting 'signup_time' and 'purchase_time' to datetime objects.
    4. Converting 'ip_address' to integer type.

    Args:
        df (pandas.DataFrame): The raw Fraud_Data DataFrame.

    Returns:
        pandas.DataFrame: The cleaned and preprocessed Fraud_Data DataFrame.
    """
    print("--- Preprocessing Fraud Data ---")
    
    # 1. Handle missing values
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows with missing values from Fraud Data.")
    else:
        print("No missing values found or dropped in Fraud Data.")

    # 2. Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows from Fraud Data.")
    else:
        print("No duplicate rows found or removed in Fraud Data.")

    # 3. Convert timestamp columns to datetime objects
    # errors='coerce' will turn unparseable dates into NaT (Not a Time)
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    print("Converted 'signup_time' and 'purchase_time' to datetime.")

    # Drop rows where datetime conversion failed (if any)
    initial_rows = df.shape[0]
    df.dropna(subset=['signup_time', 'purchase_time'], inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows due to invalid datetime conversions.")

    # 4. Convert 'ip_address' to integer type
    # Using errors='ignore' in astype for int conversion is not standard for float to int.
    # It's better to convert to numeric first, then to int, handling potential NaNs created by coerce.
    # For now, assuming ip_address are clean floats that can be cast to int.
    # If there are non-numeric values, this will raise an error.
    # Based on EDA, it seems they are float64, so direct cast is fine.
    df['ip_address'] = df['ip_address'].astype(np.int64)
    print("Converted 'ip_address' to integer type.")

    print(f"Fraud Data preprocessing complete. New shape: {df.shape}")
    return df

def preprocess_creditcard_data(df):
    """
    Performs preprocessing steps specific to the creditcard.csv dataset.

    Steps include:
    1. Handling missing values (dropping rows for now).
    2. Removing duplicate rows.

    Args:
        df (pandas.DataFrame): The raw creditcard.csv DataFrame.

    Returns:
        pandas.DataFrame: The cleaned and preprocessed creditcard.csv DataFrame.
    """
    print("--- Preprocessing Credit Card Data ---")
    
    # 1. Handle missing values
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows with missing values from Credit Card Data.")
    else:
        print("No missing values found or dropped in Credit Card Data.")

    # 2. Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows from Credit Card Data.")
    else:
        print("No duplicate rows found or removed in Credit Card Data.")
    
    print(f"Credit Card Data preprocessing complete. New shape: {df.shape}")
    return df

def preprocess_ip_to_country_data(df):
    """
    Performs preprocessing steps specific to the IpAddress_to_Country.csv dataset.

    Steps include:
    1. Handling missing values (dropping rows for now).
    2. Removing duplicate rows.
    3. Converting IP bound columns to integer type.

    Args:
        df (pandas.DataFrame): The raw IpAddress_to_Country.csv DataFrame.

    Returns:
        pandas.DataFrame: The cleaned and preprocessed IpAddress_to_Country.csv DataFrame.
    """
    print("--- Preprocessing IP to Country Data ---")

    # 1. Handle missing values
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows with missing values from IP to Country Data.")
    else:
        print("No missing values found or dropped in IP to Country Data.")

    # 2. Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows from IP to Country Data.")
    else:
        print("No duplicate rows found or removed in IP to Country Data.")

    # 3. Convert IP bound columns to integer type
    # Ensure they are numeric first, then convert to int
    df['lower_bound_ip_address'] = pd.to_numeric(df['lower_bound_ip_address'], errors='coerce').astype(np.int64)
    df['upper_bound_ip_address'] = pd.to_numeric(df['upper_bound_ip_address'], errors='coerce').astype(np.int64)
    print("Converted 'lower_bound_ip_address' and 'upper_bound_ip_address' to integer type.")

    # Drop rows where IP bound conversion failed (if any)
    initial_rows = df.shape[0]
    df.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows due to invalid IP bound conversions.")

    print(f"IP to Country Data preprocessing complete. New shape: {df.shape}")
    return df

def merge_fraud_and_ip_data(fraud_df, ip_country_df):
    """
    Merges the Fraud_Data DataFrame with the IpAddress_to_Country DataFrame
    to add country information based on IP addresses.

    This function performs a binary search-like merge. For each IP address in
    the fraud data, it finds the corresponding country by checking which IP
    range it falls into in the ip_to_country_data.

    Args:
        fraud_df (pandas.DataFrame): The preprocessed Fraud_Data DataFrame.
        ip_country_df (pandas.DataFrame): The preprocessed IpAddress_to_Country DataFrame.

    Returns:
        pandas.DataFrame: The merged DataFrame with a 'country' column.
    """
    # first check if there is a country column already in the fraud_df
    print(fraud_df.columns.tolist())
    if 'country' in fraud_df.columns.tolist():
        print("Country column already exists in Fraud Data.i.e it has already been merged in the EDA.")
        return fraud_df
    
    print("--- Merging Fraud Data with IP to Country Data ---")

    # Sort ip_country_df by lower_bound_ip_address for efficient merging
    ip_country_df_sorted = ip_country_df.sort_values(by='lower_bound_ip_address').reset_index(drop=True)

    # Use pd.merge_asof for efficient range-based merging
    # This will merge 'fraud_df' with 'ip_country_df_sorted' where 'ip_address'
    # from fraud_df is greater than or equal to 'lower_bound_ip_address'
    # and the nearest 'lower_bound_ip_address' is chosen.
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_address'), # fraud_df must be sorted by the key
        ip_country_df_sorted,
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward' # Finds the last row in ip_country_df_sorted where lower_bound_ip_address <= ip_address
    )

    # Now, filter to ensure ip_address is also within the upper_bound_ip_address
    merged_df = merged_df[
        (merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])
    ]
    
    # Drop the temporary IP bound columns from the merged_df if they are not needed for modeling
    merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True, errors='ignore')

    print(f"Merge complete. Merged Fraud Data shape: {merged_df.shape}")
    print(f"Number of fraud transactions with country information: {merged_df['country'].notna().sum()}")
    
    return merged_df

if __name__ == '__main__':
    # This block will only run when data_preprocessing.py is executed directly.
    # It demonstrates how to use the functions.
    
    # Define paths to your datasets
    data_dir = '../data'
    fraud_data_path = os.path.join(data_dir, 'Fraud_Data.csv')
    creditcard_data_path = os.path.join(data_dir, 'creditcard.csv')
    ip_to_country_data_path = os.path.join(data_dir, 'IpAddress_to_Country.csv')

    print("--- Loading Raw Data ---")
    try:
        fraud_raw = load_data(fraud_data_path)
        creditcard_raw = load_data(creditcard_data_path)
        ip_country_raw = load_data(ip_to_country_data_path)
    except FileNotFoundError as e:
        print(f"Please ensure all data files are in the '{data_dir}' directory.")
        exit() # Exit if data files are not found

    print("\n--- Applying Preprocessing ---")
    fraud_processed = preprocess_fraud_data(fraud_raw.copy())
    creditcard_processed = preprocess_creditcard_data(creditcard_raw.copy())
    ip_country_processed = preprocess_ip_to_country_data(ip_country_raw.copy())

    print("\n--- Merging Datasets ---")
    fraud_merged = merge_fraud_and_ip_data(fraud_processed.copy(), ip_country_processed.copy())

    print("\n--- Preprocessing Summary ---")
    print("\nFraud Data (after preprocessing and merging with IP data):")
    print(fraud_merged.info())
    print(fraud_merged.head())
    print("\nCredit Card Data (after preprocessing):")
    print(creditcard_processed.info())
    print(creditcard_processed.head())

    # Save the preprocessed data
    # print("\n--- Saving Preprocessed Data ---")
    # fraud_merged.to_csv(os.path.join(data_dir, 'Fraud_Data_Cleaned.csv'), index=False)
    # creditcard_processed.to_csv(os.path.join(data_dir, 'creditcard_Cleaned.csv'), index=False)