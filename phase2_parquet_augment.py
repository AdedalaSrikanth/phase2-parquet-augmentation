from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


script_dir = Path(__file__).resolve().parent

input_file = "input.parquet"
output_file = "augmented_output.parquet"

target_column = "target"
group_column = "group_id"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"

augment_copies = 2
noise_std = 0.01
random_seed = 42


def get_input_path(filename: str) -> Path:
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory as this software: {script_dir}"
        )
    return file_path


def get_output_path(filename: str) -> Path:
    return script_dir / filename


def load_parquet_file(filename: str) -> pd.DataFrame:
    file_path = get_input_path(filename)
    return pd.read_parquet(file_path)


def ensure_source_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if source_row_id_column not in df.columns:
        df[source_row_id_column] = np.arange(len(df))
    return df


def prepare_original_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[is_augmented_column] = 0
    df[augmentation_round_column] = 0
    return df


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        target_column,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
    }

    if group_column in df.columns:
        excluded_columns.add(group_column)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded_columns]


def augment_row(
    row: pd.Series,
    numeric_columns: list[str],
    rng: np.random.Generator,
    round_number: int,
) -> pd.Series:
    new_row = row.copy()

    for column in numeric_columns:
        value = row[column]
        if pd.notna(value):
            noise = rng.normal(loc=0.0, scale=noise_std)
            new_row[column] = value + noise

    new_row[is_augmented_column] = 1
    new_row[augmentation_round_column] = round_number
    return new_row


def generate_augmented_data(df: pd.DataFrame, copies: int) -> pd.DataFrame:
    numeric_columns = get_numeric_feature_columns(df)

    if not numeric_columns:
        raise ValueError("no numeric feature columns found to augment")

    rng = np.random.default_rng(random_seed)
    augmented_rows = []

    for _, row in df.iterrows():
        for round_number in range(1, copies + 1):
            augmented_rows.append(
                augment_row(
                    row=row,
                    numeric_columns=numeric_columns,
                    rng=rng,
                    round_number=round_number,
                )
            )

    if not augmented_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(augmented_rows, columns=df.columns)


def save_parquet_file(df: pd.DataFrame, filename: str) -> None:
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)
    print(f"saved: {output_path.name}")


def augment_parquet_file(
    input_filename: str = input_file,
    output_filename: str = output_file,
    copies: int = augment_copies,
) -> pd.DataFrame:
    df = load_parquet_file(input_filename)
    df = ensure_source_row_id(df)
    df = prepare_original_rows(df)

    augmented_df = generate_augmented_data(df, copies)
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    save_parquet_file(final_df, output_filename)
    return final_df


def main() -> None:
    print("starting parquet augmentation software")
    print(f"software directory: {script_dir}")

    final_df = augment_parquet_file()

    original_rows = (final_df[is_augmented_column] == 0).sum()
    augmented_rows = (final_df[is_augmented_column] == 1).sum()

    print(f"original rows: {original_rows}")
    print(f"augmented rows: {augmented_rows}")
    print(f"final rows: {len(final_df)}")
    print("process completed successfully")


if __name__ == "__main__":
    main()