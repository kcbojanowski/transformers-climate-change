import os
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Tuple
from config import (
    RAW_PATH,
    PROCESSED_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    YEARS,
    MONTHS,
    VARIABLES,
    AREA,
    TRAIN_RATIO,
    VAL_RATIO
)

# --------------------------------------------------------------------
# Data Loading and Processing
# --------------------------------------------------------------------
def process_climate_data(
    base_path: str,
    years: List[str],
    months: List[str],
    variables: Dict[str, str],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> pd.DataFrame:
    """
    Load NetCDF climate data from multiple years and months, filter by latitude and longitude,
    and convert to a single Pandas DataFrame.

    Parameters:
        base_path (str): Base path to the raw data directory.
        years (List[str]): List of years to process.
        months (List[str]): List of months (e.g. ['01', '02', ...]) to process.
        variables (Dict[str, str]): Mapping of variable keys to NetCDF file names.
        lat_min (float): Minimum latitude filter.
        lat_max (float): Maximum latitude filter.
        lon_min (float): Minimum longitude filter.
        lon_max (float): Maximum longitude filter.

    Returns:
        pd.DataFrame: A long-format DataFrame containing all requested variables,
                      indexed by time, latitude, and longitude.
    """
    rows = []
    file_count = 0

    for year in years:
        for month in months:
            folder_name = f"climate_data_{year}_{month.zfill(2)}"
            folder_path = os.path.join(base_path, folder_name)

            if not os.path.exists(folder_path):
                print(f"[WARNING] Folder not found: {folder_path}")
                continue

            temp_data = {}

            for var_key, file_name in variables.items():
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    try:
                        print(f"[INFO] Loading file: {file_path}")
                        ds = xr.open_dataset(file_path)

                        # Store lat/lon if not already present
                        if "latitude" not in temp_data or "longitude" not in temp_data:
                            temp_data["latitude"] = ds.latitude.values
                            temp_data["longitude"] = ds.longitude.values

                        # Store time if not already present
                        if "time" not in temp_data:
                            if "valid_time" in ds:
                                temp_data["time"] = pd.to_datetime(ds.valid_time.values)
                            elif "time" in ds:
                                temp_data["time"] = pd.to_datetime(ds.time.values)
                            else:
                                print(f"[ERROR] No valid time variable found in {file_path}")
                                continue

                        # Extract variable values
                        temp_data[var_key] = ds[var_key].values

                        # Convert total precipitation from meters to millimeters if units are "m"
                        if var_key == "tp":
                            if "units" in ds[var_key].attrs and ds[var_key].attrs["units"] == "m":
                                temp_data[var_key] = temp_data[var_key] * 1000

                        file_count += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to load file {file_path}: {e}")
                else:
                    print(f"[WARNING] File not found: {file_path}")

            # Build rows if we have time, latitude, longitude
            if (
                temp_data
                and "time" in temp_data
                and "latitude" in temp_data
                and "longitude" in temp_data
            ):
                time_values = temp_data["time"]
                for t_idx, time_val in enumerate(time_values):
                    for lat_idx, lat_val in enumerate(temp_data["latitude"]):
                        for lon_idx, lon_val in enumerate(temp_data["longitude"]):
                            if lat_min <= lat_val <= lat_max and lon_min <= lon_val <= lon_max:
                                row = {
                                    "time": time_val,
                                    "time_idx": t_idx,
                                    "latitude": lat_val,
                                    "longitude": lon_val,
                                    "group_id": f"{lat_val:.1f}_{lon_val:.1f}",
                                }
                                # Populate variables
                                for var_key in variables.keys():
                                    if var_key in temp_data:
                                        row[var_key] = temp_data[var_key][t_idx, lat_idx, lon_idx]
                                    else:
                                        row[var_key] = None
                                rows.append(row)

    if rows:
        data = pd.DataFrame(rows)
        print(f"[INFO] Data processing completed. Processed {file_count} NetCDF files.")
        # Sort by time and reset index
        data = data.sort_values(by="time").reset_index(drop=True)
        return data
    else:
        print("[WARNING] No data processed. Check input parameters.")
        return pd.DataFrame()


# --------------------------------------------------------------------
# Preprocessing Utilities
# --------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates()


def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Fill missing values in the DataFrame using a specified method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): The method to fill missing values. Options are 'ffill', 'bfill', 'mean', or 'median'.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if method in ["ffill", "bfill"]:
        df = df.fillna(method=method)
    elif method == "mean":
        # Only fill numeric columns using their mean values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == "median":
        # Only fill numeric columns using their median values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        raise ValueError("Invalid method for filling missing values. Choose 'ffill', 'bfill', 'mean', or 'median'.")
    return df


def remove_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of column names to check for outliers.
        threshold (float): IQR multiplier to define the outlier bounds.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            lower_bound = Q1 - threshold * iqr
            upper_bound = Q3 + threshold * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def scale_features(df: pd.DataFrame, columns: List[str], method: str = "standard") -> Tuple[pd.DataFrame, object]:
    """
    Scale specified features using either StandardScaler or MinMaxScaler.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (List[str]): List of column names to scale.
        method (str): Scaling method to use. 'standard' for StandardScaler or 'minmax' for MinMaxScaler.

    Returns:
        Tuple[pd.DataFrame, object]: A tuple containing the scaled DataFrame and the scaler object used.
    """
    if not columns:
        return df, None

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")

    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train, validation, and test sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        train_ratio (float): Fraction of data to be used for training.
        val_ratio (float): Fraction of data to be used for validation (after training split).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    total_len = len(df)
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    return train_df, val_df, test_df

def convert_group_id_to_numeric(df: pd.DataFrame, col: str = 'group_id') -> pd.DataFrame:
    """
    Convert a DataFrame column containing group_id strings (formatted as 'latitude_longitude')
    into two numeric columns: 'latitude_numeric' and 'longitude_numeric'.
    If the conversion fails for any value, set the corresponding numeric values to NaN and print an error message.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column name containing group_id strings. Default is 'group_id'.

    Returns:
        pd.DataFrame: The DataFrame with two new columns: 'latitude_numeric' and 'longitude_numeric'.
    """
    latitudes: list = []
    longitudes: list = []

    for value in df[col]:
        # Check if value is a string
        if not isinstance(value, str):
            print(f"[DEBUG] group_id value is not a string: {value}")
            latitudes.append(float('nan'))
            longitudes.append(float('nan'))
            continue

        # Debug: if the value is unusually long
        if len(value) > 20:
            print(f"[DEBUG] group_id value '{value}' is unusually long (length {len(value)}).")

        parts = value.split('_')
        if len(parts) < 2:
            print(f"[WARNING] Unexpected group_id format: {value}")
            latitudes.append(float('nan'))
            longitudes.append(float('nan'))
        elif len(parts) > 2:
            print(f"[DEBUG] group_id value '{value}' has more than 2 parts. Using first two parts.")
            parts = parts[:2]

        try:
            latitudes.append(float(parts[0]))
            longitudes.append(float(parts[1]))
        except Exception as e:
            print(f"[ERROR] Failed to convert group_id '{value}' to numeric: {e}")
            latitudes.append(float('nan'))
            longitudes.append(float('nan'))

    df['latitude_numeric'] = latitudes
    df['longitude_numeric'] = longitudes
    return df


# --------------------------------------------------------------------
# Main Preprocessing Pipeline
# --------------------------------------------------------------------
def main_pipeline() -> None:
    """
    Main function that orchestrates the entire preprocessing pipeline:
    1. Load and process raw NetCDF data.
    2. Remove duplicates.
    3. Fill missing values.
    4. Remove outliers.
    5. Scale features.
    6. Split into train, validation, and test sets.
    7. Save results to CSV files.
    """
    # 1. Load and process raw NetCDF data
    df = process_climate_data(
        base_path=RAW_PATH,
        years=YEARS,
        months=MONTHS,
        variables=VARIABLES,
        lat_min=AREA[0],
        lat_max=AREA[1],
        lon_min=AREA[2],
        lon_max=AREA[3]
    )
    if df.empty:
        print("[ERROR] No data was loaded. Exiting pipeline.")
        return

    print("\n-------------------------")
    print("--- Data Information ---")
    print("-------------------------")
    print(f"[INFO] Loaded data shape: {df.shape}")
    print(f"[INFO] Loaded data columns: {df.columns}")
    print(f"[INFO] Loaded data head: {df.head()}")
    print("-------------------------\n")

    df = convert_group_id_to_numeric(df, col='group_id')


    print(f"[INFO] Removing duplicates from the dataset...")
    df = remove_duplicates(df)

    print(f"[INFO] Filling missing values in the dataset...")
    df = fill_missing_values(df, method="mean")

    print(f"[INFO] Removing outliers from the dataset...")
    df = remove_outliers(df, columns=["t2m", "tp", "d2m"], threshold=1.5)

    print(f"[INFO] Scaling features in the dataset...")
    df, _ = scale_features(df, columns=["t2m", "tp", "d2m"], method="standard")

    print(f"[INFO] Splitting dataset into train, validation, and test sets...")
    train_df, val_df, test_df = train_val_test_split(df, TRAIN_RATIO, VAL_RATIO)

    # 7. Save results to CSV files
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"[INFO] Full dataset saved to {PROCESSED_DATA_PATH}")
    print(f"[INFO] Train set saved to {TRAIN_DATA_PATH}")
    print(f"[INFO] Validation set saved to {VAL_DATA_PATH}")
    print(f"[INFO] Test set saved to {TEST_DATA_PATH}")
    print("[INFO] Preprocessing pipeline completed successfully.")


# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    main_pipeline()