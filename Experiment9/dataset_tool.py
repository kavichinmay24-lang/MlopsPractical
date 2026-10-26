
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def validate_dataset(df):
    """Display dataset structure and identify data-quality issues."""
    print("\n========== DATASET VALIDATION ==========")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nColumn names:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"Completely empty rows: {df.isnull().all(axis=1).sum()}")

    print("\nFirst five rows:")
    print(df.head())


def preprocess_dataset(df):
    """Remove duplicate/empty rows and handle missing values."""
    cleaned = df.copy()

    # Treat blank strings and whitespace-only cells as missing values.
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    # Remove rows that contain no data.
    empty_rows = int(cleaned.isnull().all(axis=1).sum())
    cleaned = cleaned.dropna(how="all")

    # Remove exact duplicate records.
    duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # Convert columns to numeric when all non-missing values are numeric.
    for column in cleaned.columns:
        converted = pd.to_numeric(cleaned[column], errors="coerce")
        non_missing = cleaned[column].notna()

        if non_missing.any() and converted[non_missing].notna().all():
            cleaned[column] = converted

    # Fill missing numeric values with the column median.
    # Fill missing non-numeric values with the most frequent value.
    for column in cleaned.columns:
        if cleaned[column].isnull().any():
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                median_value = cleaned[column].median()

                if pd.notna(median_value):
                    cleaned[column] = cleaned[column].fillna(median_value)
            else:
                modes = cleaned[column].mode(dropna=True)

                if not modes.empty:
                    cleaned[column] = cleaned[column].fillna(modes.iloc[0])

    return cleaned, duplicates, empty_rows


def main():
    parser = argparse.ArgumentParser(
        description="Validate and preprocess a CSV dataset."
    )
    parser.add_argument(
        "--input",
        default="sample_dataset.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output",
        default="cleaned_dataset.csv",
        help="Path for the cleaned CSV file."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        parser.error(f"Input CSV file not found: {input_path}")

    if input_path.resolve() == output_path.resolve():
        parser.error("Input and output files must be different.")

    try:
        df = pd.read_csv(input_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        parser.error(f"Could not read CSV file: {error}")

    if len(df.columns) == 0:
        parser.error("The CSV file does not contain any columns.")

    if len(df.columns) != len(set(df.columns)):
        parser.error("The CSV contains duplicate column names.")

    # Standardize column names and check for empty names.
    df.columns = df.columns.astype(str).str.strip()

    if any(not column for column in df.columns):
        parser.error("The CSV contains an empty column name.")

    validate_dataset(df)

    cleaned, duplicates_removed, empty_rows_removed = (
        preprocess_dataset(df)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print("\n========== PREPROCESSING SUMMARY ==========")
    print(f"Original rows: {len(df)}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Completely empty rows removed: {empty_rows_removed}")
    print(f"Final rows: {len(cleaned)}")
    print(f"Remaining missing values: {int(cleaned.isnull().sum().sum())}")
    print(f"Cleaned dataset saved to: {output_path.resolve()}")

    if cleaned.isnull().sum().sum() == 0:
        print("\nSUCCESS: Dataset validation and preprocessing completed.")
    else:
        print(
            "\nWARNING: Some missing values remain. "
            "Check columns that contain only missing values."
        )


if __name__ == "__main__":
    main()
