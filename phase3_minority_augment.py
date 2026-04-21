# data augmentation for minority class balancing
# reads parquet input and generates augmented output
# focuses only on minority class to balance dataset

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd


# current script directory
script_dir = Path(__file__).resolve().parent

# default input and output files
input_file = "input.parquet"
output_file = "augmented_minority_only.parquet"

# column names used in processing
target_column = "target"
group_column = "group_id"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"
randomization_mode_column = "randomization_mode"

# augmentation settings
noise_std = 0.01
random_seed = 42
randomization_mode = "group_based"


def get_input_path(filename: str) -> Path:
    # build full path for input file
    file_path = script_dir / filename

    # stop if file does not exist
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory as this software: {script_dir}"
        )

    return file_path


def get_output_path(filename: str) -> Path:
    # build output file path
    return script_dir / filename


def load_parquet_file(filename: str) -> pd.DataFrame:
    # load parquet file into dataframe
    return pd.read_parquet(get_input_path(filename))


def ensure_source_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # create unique row id if missing
    if source_row_id_column not in df.columns:
        df[source_row_id_column] = np.arange(len(df))

    return df


def ensure_group_id(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()

    # group-based mode: ensure group column exists
    if mode == "group_based":
        if group_column not in df.columns:
            df[group_column] = np.arange(len(df))

    # normal mode: remove group column if present
    elif mode == "normal":
        if group_column in df.columns:
            df = df.drop(columns=[group_column])

    return df


def prepare_original_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()

    # mark original rows
    df[is_augmented_column] = 0
    df[augmentation_round_column] = 0
    df[randomization_mode_column] = mode

    return df


def get_numeric_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    # exclude metadata and non-feature columns
    skip_columns = {
        target_col,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
        randomization_mode_column,
    }

    if group_column in df.columns:
        skip_columns.add(group_column)

    # select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return [col for col in numeric_columns if col not in skip_columns]


def augment_row(
    row: pd.Series,
    numeric_columns: list[str],
    rng: np.random.Generator,
    round_number: int,
    mode: str,
) -> pd.Series:
    new_row = row.copy()

    # add small noise to numeric columns
    for col in numeric_columns:
        value = row[col]
        if pd.notna(value):
            noise = rng.normal(0.0, noise_std)
            new_row[col] = value + noise

    # mark row as augmented
    new_row[is_augmented_column] = 1
    new_row[augmentation_round_column] = round_number
    new_row[randomization_mode_column] = mode

    return new_row


def generate_minority_augmented_data(
    df: pd.DataFrame,
    mode: str,
    target_col: str,
) -> pd.DataFrame:
    # ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    numeric_columns = get_numeric_feature_columns(df, target_col)

    # stop if no numeric columns found
    if not numeric_columns:
        raise ValueError("no numeric columns found to augment")

    rng = np.random.default_rng(random_seed)

    # count class distribution
    class_counts = df[target_col].value_counts()

    # find minority and majority classes
    minority_value = class_counts.idxmin()
    majority_count = int(class_counts.max())

    minority_rows = df[df[target_col] == minority_value].copy()
    minority_count = len(minority_rows)

    # stop if minority class is empty
    if minority_count == 0:
        raise ValueError("minority class is empty")

    # calculate how many rows to generate
    needed_rows = majority_count - minority_count

    # if already balanced, skip augmentation
    if needed_rows <= 0:
        print("classes are already balanced")
        return pd.DataFrame(columns=df.columns)

    print(f"minority class rows: {minority_count}")
    print(f"majority class rows: {majority_count}")
    print(f"rows to generate: {needed_rows}")

    augmented_rows = []

    # cycle through minority rows to generate new samples
    minority_iter = itertools.cycle(minority_rows.iterrows())

    for round_number, (_, row) in enumerate(
        itertools.islice(minority_iter, needed_rows), 1
    ):
        augmented_rows.append(
            augment_row(
                row=row,
                numeric_columns=numeric_columns,
                rng=rng,
                round_number=round_number,
                mode=mode,
            )
        )

    return pd.DataFrame(augmented_rows, columns=df.columns)


def save_parquet_file(df: pd.DataFrame, filename: str) -> None:
    # save output parquet file
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)

    print(f"saved output to: {output_path.name}")


def augment_minority_only_file(
    input_filename: str = input_file,
    output_filename: str = output_file,
    mode: str = randomization_mode,
    target_col: str = target_column,
) -> pd.DataFrame:
    # validate mode
    if mode not in {"normal", "group_based"}:
        raise ValueError("mode must be either 'normal' or 'group_based'")

    # load and prepare data
    df = load_parquet_file(input_filename)
    df = ensure_source_row_id(df)
    df = ensure_group_id(df, mode)
    df = prepare_original_rows(df, mode)

    # generate minority augmentation
    augmented_df = generate_minority_augmented_data(df, mode, target_col)

    # combine original and augmented rows
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    save_parquet_file(final_df, output_filename)
    return final_df


def parse_arguments():
    # read command line switches from the user
    parser = argparse.ArgumentParser(
        description="minority class augmentation software for tabular parquet datasets"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input parquet file name",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output parquet file name",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=target_column,
        help="target column name",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=randomization_mode,
        choices=["normal", "group_based"],
        help="augmentation mode",
    )

    return parser.parse_args()


def main() -> None:
    # read input values from command line
    args = parse_arguments()

    print("starting minority augmentation")
    print(f"script folder: {script_dir}")
    print(f"input file: {args.input}")
    print(f"output file: {args.output}")
    print(f"target column: {args.target}")
    print(f"randomization mode: {args.mode}")

    final_df = augment_minority_only_file(
        input_filename=args.input,
        output_filename=args.output,
        mode=args.mode,
        target_col=args.target,
    )

    # summary of results
    original_count = int((final_df[is_augmented_column] == 0).sum())
    augmented_count = int((final_df[is_augmented_column] == 1).sum())

    print(f"original rows: {original_count}")
    print(f"augmented rows: {augmented_count}")
    print(f"total rows: {len(final_df)}")
    print("done")


if __name__ == "__main__":
    main()