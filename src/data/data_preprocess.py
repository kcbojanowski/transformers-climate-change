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

            # Build rows if we have time, latitude, and longitude
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
        data['time'] = pd.to_datetime(data['time'])
        data = data.sort_values(by=['group_id', 'time']).reset_index(drop=True)

        data['norm_time'] = data['time'].dt.normalize()

        full_date_range = pd.date_range(start=data['norm_time'].min(), end=data['norm_time'].max(), freq='D')
        print(
            f"[INFO] Full date range from {full_date_range[0]} to {full_date_range[-1]}, total {len(full_date_range)} days")

        filled_groups = []
        for group, subdf in data.groupby('group_id'):
            group_lat = subdf['latitude'].iloc[0]
            group_lon = subdf['longitude'].iloc[0]

            subdf = subdf.set_index('norm_time')
            subdf = subdf.reindex(full_date_range)

            subdf['group_id'] = group
            subdf['latitude'] = group_lat
            subdf['longitude'] = group_lon
            filled_groups.append(subdf.reset_index().rename(columns={'index': 'norm_time'}))
        data_full = pd.concat(filled_groups, ignore_index=True)

        date_to_idx = {d: i for i, d in enumerate(full_date_range)}
        data_full['time_idx'] = data_full['norm_time'].map(date_to_idx)

        data_full['time'] = data_full['norm_time']
        data_full.drop(columns=['norm_time'], inplace=True)

        print(f"[INFO] Time Index Range: {data_full['time_idx'].min()} to {data_full['time_idx'].max()}")
        return data_full
    else:
        print("[WARNING] No data processed. Check input parameters.")
        return pd.DataFrame()


# --------------------------------------------------------------------
# Preprocessing Utilities
# --------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def fill_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    if method in ["ffill", "bfill"]:
        df = df.fillna(method=method)
    elif method == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        raise ValueError("Invalid method for filling missing values. Choose 'ffill', 'bfill', 'mean', or 'median'.")
    return df


def remove_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
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
    unique_time_idx = sorted(df['time_idx'].unique())
    total_days = len(unique_time_idx)
    train_count = int(total_days * train_ratio)
    val_count = int(total_days * val_ratio)

    train_days = set(unique_time_idx[:train_count])
    val_days = set(unique_time_idx[train_count:train_count + val_count])
    test_days = set(unique_time_idx[train_count + val_count:])

    train_df = df[df['time_idx'].isin(train_days)].copy()
    val_df = df[df['time_idx'].isin(val_days)].copy()
    test_df = df[df['time_idx'].isin(test_days)].copy()

    return train_df, val_df, test_df


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

    print(f"[INFO] Removing duplicates from the dataset...")
    df = remove_duplicates(df)

    print(f"[INFO] Filling missing values in the dataset...")
    df = fill_missing_values(df, method="mean")

    print(f"[INFO] Removing outliers from the dataset...")
    df = remove_outliers(df, columns=["t2m", "tp", "d2m"], threshold=1.5)

    print(f"[INFO] Scaling features in the dataset...")
    df, _ = scale_features(df, columns=["t2m", "tp", "d2m"], method="standard")

    df = df.sort_values(by='time').reset_index(drop=True)
    print(f"[INFO] Time Index Range: {df['time_idx'].min()} to {df['time_idx'].max()}")

    print(f"[INFO] Splitting dataset into train, validation, and test sets...")
    train_df, val_df, test_df = train_val_test_split(df, TRAIN_RATIO, VAL_RATIO)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"[INFO] Full dataset saved to {PROCESSED_DATA_PATH}")
    print(f"[INFO] Train set saved to {TRAIN_DATA_PATH}")
    print(f"[INFO] Validation set saved to {VAL_DATA_PATH}")
    print(f"[INFO] Test set saved to {TEST_DATA_PATH}")
    print("[INFO] Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main_pipeline()