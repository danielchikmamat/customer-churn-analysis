
import pandas as pd
import os

def load_csv(file_name="telco-customer-churn.csv", data_dir="raw"):
    """
    Load a CSV file from the raw data folder.

    Parameters:
    ----------
    file_name : str
        Name of the CSV file to load
    data_dir : str
        Relative path to the data/raw folder

    Returns:
    -------
    pd.DataFrame
        Loaded dataset
    """
    file_path = os.path.join(data_dir, file_name)
    
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded {file_name} with shape {df.shape}")
    return df

def save_data(df, file_name="cleaned_telco_customer_churn.csv", data_dir="processed"):
    """Save the cleaned DataFrame to a CSV file in the processed data folder."""
    file_path = os.path.join(data_dir, file_name)
    df.to_csv(file_path, index=False)
    print(f"Saved cleaned data to {file_path}")