import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from utils.config import ALL_FILES, RELEVANT_COLUMNS, ELECTRIC_SCHEMA

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'MachineLearningCVE')
DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')

def load_data(files: list) -> pd.DataFrame:
    """Join many datasets in one DataFrame"""
    read_kwargs = {'decimal': '.'}
    dfs = []
    for key in files:
        file = ALL_FILES.get(key)
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, **read_kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='ISO-8859-1', **read_kwargs)
        df['source_file'] = file
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_dataset(path):
    """
    Load a dataset from CSV file.

    Parameters:
        path (str): full file path (e.g. 'data/my_data.csv')

    Returns:
        pd.DataFrame: loaded dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use CSV.")

    print(f"Dataset loaded from: {path} — shape: {df.shape}")
    return df


def select_relevant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Returns columns from RELEVANT_COLUMNS if exist in data"""

    print(f'{RELEVANT_COLUMNS = }')
    available_cols = [col for col in RELEVANT_COLUMNS if col in df.columns]
    missing_cols = [col for col in RELEVANT_COLUMNS if col not in df.columns]

    percent_available = round(len(available_cols) / len(RELEVANT_COLUMNS) * 100, 1)

    if verbose:
        print(f"Using {len(available_cols)} of {len(RELEVANT_COLUMNS)} features ({percent_available}%)")
        if missing_cols:
            print(f"Missing columns: {missing_cols}")

    return df[available_cols]

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df

def remove_duplicates(df, verbose=True):
    """Removes duplicated rows and optionally prints info."""
    before = len(df)
    df_cleaned = df.drop_duplicates()
    after = len(df_cleaned)
    if verbose:
        print(f"Removed {before - after} duplicate rows (remaining: {after})")
    return df_cleaned

def check_dtypes(df):
    print(df.dtypes)
    if df.dtypes.eq('object').any():
        print("Warning: Some columns are still object type.")

def encode_labels(df, label_column='Label'):
    df[label_column] = df[label_column].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    return df

def rename_columns(df):
    return df.rename(columns=ELECTRIC_SCHEMA)

def save_dataset(df, name="cic_prepared", include_timestamp=True, columns=None):
    os.makedirs(DATASETS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
    base_name = f"{name}_{timestamp}" if timestamp else name
    file_path = os.path.join(DATASETS_DIR, base_name + ".csv")

    df.to_csv(file_path, index=False, columns=columns)

    print(f"Dataset saved to: {file_path}")
    return file_path
