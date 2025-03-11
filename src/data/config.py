DATASET = "derived-era5-single-levels-daily-statistics"


#########################################################
#                 Download parameters                   #
#########################################################

# Parameter database: https://codes.ecmwf.int/grib/param-db/

# Base request parameters
BASE_REQUEST_PARAMS = {
    "product_type": "reanalysis",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "total_precipitation",
    ],
    "day": [f"{day:02d}" for day in range(1, 32)],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+01:00",
    "frequency": "6_hourly",
    "area": [75, -25, 35, 40]
}

RAW_PATH = "../../data/raw/"
OUTPUT_FILENAME_PATTERN = RAW_PATH + "climate_data_{year}_{month}.zip"

YEARS = ["2022"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

#########################################################
#               Preprocess parameters                   #
#########################################################

# Paths
PROCESSED_PATH = "../../data/processed/"
PROCESSED_DATA_PATH = PROCESSED_PATH + "climate_data.csv"
PROCESSED_TRAIN_DATA_PATH = PROCESSED_PATH + "full/train_climate_data.csv"
PROCESSED_VAL_DATA_PATH = PROCESSED_PATH + "full/val_climate_data.csv"
PROCESSED_TEST_DATA_PATH = PROCESSED_PATH + "full/test_climate_data.csv"

SNIPPET_TRAIN_DATA_PATH = PROCESSED_PATH + "snippet/train_climate_data.csv"
SNIPPET_VAL_DATA_PATH = PROCESSED_PATH + "snippet/val_climate_data.csv"
SNIPPET_TEST_DATA_PATH = PROCESSED_PATH + "snippet/test_climate_data.csv"

USE_SNIPPET = True

if USE_SNIPPET:
    TRAIN_DATA_PATH = SNIPPET_TRAIN_DATA_PATH
    VAL_DATA_PATH = SNIPPET_VAL_DATA_PATH
    TEST_DATA_PATH = SNIPPET_TEST_DATA_PATH
else:
    TRAIN_DATA_PATH = PROCESSED_TRAIN_DATA_PATH
    VAL_DATA_PATH = PROCESSED_VAL_DATA_PATH
    TEST_DATA_PATH = PROCESSED_TEST_DATA_PATH

VARIABLES = {
    "t2m": "2m_temperature_0_daily-mean.nc",          # Temperature
    "tp": "total_precipitation_0_daily-mean.nc",      # Total precipitation
    "d2m": "2m_dewpoint_temperature_0_daily-mean.nc"  # Dewpoint temperature
}

# Ranges
# [lat_min, lat_max, lon_min, lon_max]
AREA = [49.0, 55.0, 14.0, 24.0]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2



