import pandas as pd
import os

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
        df.to_csv(file_path, index= False, **kwargs)
        print(f"Successfully saved data to '{file_path}'.")
    except Exception as e:
        print(f"Error saving data to '{file_path}': {e}")
        raise