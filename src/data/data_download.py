import cdsapi
from src.config import DATASET, BASE_REQUEST_PARAMS, OUTPUT_FILENAME_PATTERN, YEARS, MONTHS


def download_data(year: str, month: str) -> None:
    """
    Download climate data for a given year and month using the CDS API.
    """
    request_params = BASE_REQUEST_PARAMS.copy()
    request_params["year"] = year
    request_params["month"] = month

    filename = OUTPUT_FILENAME_PATTERN.format(year=year, month=month)

    client = cdsapi.Client()
    client.retrieve(DATASET, request_params).download(filename)
    print(f"Downloaded data for {year}-{month} to {filename}")


def download_all_data() -> None:
    """
    Download climate data for all configured years and months.
    """
    for year in YEARS:
        for month in MONTHS:
            print(f"Starting download for {year}-{month}...")
            download_data(year, month)


if __name__ == "__main__":
    download_all_data()