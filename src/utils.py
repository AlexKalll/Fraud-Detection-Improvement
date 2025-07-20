# src/utils.py

import pandas as pd
import os
import re # Import regex module

def load_data(file_path, **kwargs):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        **kwargs: Additional keyword arguments to pass to pandas.read_csv().

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: For other potential errors during file loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")
    try:
        df = pd.read_csv(file_path, **kwargs)
        print(f"Successfully loaded data from '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        raise

def save_data(df, file_path, index=False, **kwargs):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        file_path (str): The path to save the CSV file.
        index (bool): Whether to write the DataFrame index as a column. Defaults to False.
        **kwargs: Additional keyword arguments to pass to pandas.to_csv().
    """
    try:
        df.to_csv(file_path, index=index, **kwargs)
        print(f"Successfully saved data to '{file_path}'.")
    except Exception as e:
        print(f"Error saving data to '{file_path}': {e}")
        raise

def clean_dataframe_columns(df):
    """
    Cleans column names of a DataFrame by replacing special characters with underscores
    and removing characters not suitable for LightGBM.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    cleaned_df = df.copy()
    new_columns = []
    for col in cleaned_df.columns:
        # Replace problematic characters with underscore
        # LightGBM doesn't like [, ], <, >, :, =
        # Also replace spaces and other non-alphanumeric characters (except underscore)
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        # Handle cases where multiple underscores might appear
        new_col = re.sub(r'_{2,}', '_', new_col)
        new_columns.append(new_col)
    
    cleaned_df.columns = new_columns
    print("DataFrame column names cleaned for LightGBM compatibility.")
    return cleaned_df

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # This block will only run when data_preprocessing.py is executed directly.
    
    # Create dummy data for testing
    dummy_fraud_data = pd.DataFrame({
        'user id': [1, 2, 3],
        'signup time': ['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00'],
        'purchase value ($)': [50, 100, 25],
        'device_id': ['abc', 'def', 'ghi'],
        'source:web': ['SEO', 'Ads', 'Direct'],
        'browser<type>': ['Chrome', 'Firefox', 'Safari'],
        'class': [0, 1, 0]
    })
    
    print("Original columns:", dummy_fraud_data.columns.tolist())
    cleaned_dummy_df = clean_dataframe_columns(dummy_fraud_data)
    print("Cleaned columns:", cleaned_dummy_df.columns.tolist())

    # Ensure data directory exists for testing
    os.makedirs('../data', exist_ok=True)
    
    # Save dummy data
    save_data(cleaned_dummy_df, '../data/dummy_cleaned.csv')

    print("\n--- Testing load_data function ---")
    try:
        loaded_df = load_data('../data/dummy_cleaned.csv')
        print("Loaded data head:\n", loaded_df.head())
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")