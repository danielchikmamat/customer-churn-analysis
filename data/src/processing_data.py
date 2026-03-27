
from file import load_csv, save_data
import pandas as pd
import re

def check_churn_class_balance(df):
    """Check the balance of the target variable 'churn' and print the distribution."""
    class_balance = df["churn"].value_counts(normalize=True)
    print("Churn distribution:")
    print(class_balance)


def check_data_types(df):
    """Check the data types of each column and print the result.
    If float64 is present, it indicates that there are missing values in the column, which should be fixed before analysis."""
    print("Data types of each column:")
    print(df.dtypes)


def fix_missing_total_charges(df):
    """replace missing values in total_charges with tenure multiplied by monthly charges.
    customers with 0 tenure have empty total_charges, so we can safely fill missing values with tenure * monthly charges."""
    df["total_charges"].fillna(df["tenure"] * df["monthly_charges"], inplace=True)
    return df


def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame and print the number of duplicates removed."""
    initial_length = len(df)
    df = df.drop_duplicates()
    final_length = len(df)
    print(f"Removed {initial_length - final_length} duplicate rows.")
    return df


def drop_customer_id(df):
    """Drop the customer_id column from the DataFrame. 
    Since customer_id is a unique identifier for each customer, 
    it does not provide any useful information for analysis and can be safely dropped."""
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
        print("Dropped customer_id column.")
    else:
        print("customer_id column not found.")
    return df

def lowercase_string_and_whitespace_columns(df):
    """Convert all string columns in the DataFrame to lowercase to ensure consistency."""
    string_columns = df.select_dtypes(include=["object"]).columns
    for col in string_columns:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(" ", "_")
    print("Converted string columns to lowercase, added underscores, and removed whitespace.")
    return df


def clean_column_names(df):
    """
    Clean dataframe column names:
    - Strip leading/trailing spaces
    - Convert first letter to lowercase
    - Convert CamelCase to snake_case
    - Replace spaces with underscores
    - Make all letters lowercase
    """
    new_columns = []
    for col in df.columns:
        # 1️⃣ strip spaces
        col_clean = col.strip()
        
        # 2️⃣ lowercase first letter
        col_clean = col_clean[0].lower() + col_clean[1:] if col_clean else col_clean
        
        # 3️⃣ convert CamelCase to snake_case
        col_clean = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', col_clean)
        
        # 4️⃣ replace spaces with underscores
        col_clean = col_clean.replace(' ', '_')
        
        # 5️⃣ make everything lowercase
        col_clean = col_clean.lower()
        
        new_columns.append(col_clean)
    
    df.columns = new_columns
    return df


def binary_encoding_columns(df):
    """return columns suitable for binary encoding."""
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]
    if binary_columns:
        return binary_columns
    else:
        return False

def convert_yes_no_to_binary(df, binary_columns=["partner", "dependents", "phone_service", "paperless_billing", "churn"]):
    """Convert the specified columns in the DataFrame to numeric values (0 and 1)."""
    for col in binary_columns:
        df[col] = df[col].map({"yes": 1, "no": 0})
    print(f"Converted {binary_columns} to binary encoding.")
    return df

def convert_gender_to_binary(df):
    """Convert the 'gender' column to binary encoding, where 'female' is mapped to 1 and 'male' is mapped to 0."""
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"female": 1, "male": 0})
        print("Converted 'gender' column to binary encoding.")
    return df

def convert_senior_citizen_to_binary(df):
    """convert dtype from object to int."""
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = df["senior_citizen"].astype(int)
        print(f"Converted 'senior_citizen' column to {df['senior_citizen'].dtype}.")
    return df


def convert_multiple_lines_to_binary(df):
    """Convert the 'multiple_lines' column to binary encoding, where 'no phone service' is mapped to 0 and 'yes' is mapped to 1."""
    if "multiple_lines" in df.columns:
        df["multiple_lines"] = df["multiple_lines"].map({"no phone service": 0, "yes": 1, "no": 0})
        print("Converted 'multiple_lines' column to binary encoding.")
    return df


def convert_column_to_one_hot(df, column_name):
    """Convert the specified column to one-hot encoding with 0/1."""
    if column_name in df.columns:
        df = pd.get_dummies(df, columns=[column_name], prefix=column_name, dtype=int)
        print(f"Converted '{column_name}' column to one-hot encoding with 0/1.")

    return df


def convert_internet_dependent_columns_to_binary(df):
    """Convert the columns that depend on 'internet_service' to binary encoding, where 'no internet service' is mapped to 0 and 'yes' is mapped to 1."""
    internet_dependent_columns = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    for col in internet_dependent_columns:
        if col in df.columns:
            df[col] = df[col].map({"no internet service": 0, "yes": 1, "no": 0})
            print(f"Converted '{col}' column to binary encoding.")
    return df

def check_missing_values(df):
    """check missing values."""
    print("Missing values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:  
        print(missing_values)


# Example usage
if __name__ == "__main__":
    df = load_csv()
    print(f"raw data length: {len(df)}")

    df = clean_column_names(df)
    print(f"Cleaned column names: {df.columns.tolist()}")
    
    #check for missing values, data types, and class balance before any data cleaning
    check_missing_values(df)
    check_data_types(df)
    check_churn_class_balance(df)

    #handle missing values and remove duplicates before any analysis
    df = fix_missing_total_charges(df)
    df = remove_duplicates(df)

    #fix string consistency before binary encoding and one-hot encoding
    df = lowercase_string_and_whitespace_columns(df)

    #convert columns with 2 unique values to binary encoding
    binary_columns = binary_encoding_columns(df)
    print(f"Columns suitable for binary encoding: {binary_columns}")
    df = convert_yes_no_to_binary(df)
    df = convert_gender_to_binary(df)
    df = convert_senior_citizen_to_binary(df)

    #convert 3 options to binary if dependent on internet service or phoneservice
    df = convert_multiple_lines_to_binary(df)
    df = convert_internet_dependent_columns_to_binary(df)

    df = convert_column_to_one_hot(df, "internet_service")
    df = convert_column_to_one_hot(df, "contract")
    df = convert_column_to_one_hot(df, "payment_method")

    #drop uselss columns after encoding
    df = drop_customer_id(df)
    
    print(df.columns.tolist())
    
    save_data(df)


    
    







    

  
 



    

