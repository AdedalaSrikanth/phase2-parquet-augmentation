# data augmentation tool for tabular datasets (parkinsons dataset)
# reads parquet input and generates augmented output
# uses numeric noise-based augmentation with group support

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# current script directory
script_dir = Path(__file__).resolve().parent

# default input and output files
input_file = "parkinsons.parquet"
output_file = "parkinsons_augmented.parquet"

# column names used in processing
target_column = "target"
group_column = "subject#"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"
randomization_mode_column = "randomization_mode"

# augmentation settings
augment_copies = 2
noise_std = 0.01
random_seed = 42
randomization_mode = "group_based"


def get_input_path(filename: str) -> Path:
    # build full path for input file
    file_path = script_dir / filename

    # stop execution if file does not exist
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory as this software: {script_dir}"
        )
    return file_path


def get_output_path(filename: str) -> Path:
    # build full path for output file
    return script_dir / filename


def load_parquet_file(filename: str) -> pd.DataFrame:
    # load parquet file into dataframe
    file_path = get_input_path(filename)
    return pd.read_parquet(file_path)


def ensure_source_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # create unique id for each row if missing
    if source_row_id_column not in df.columns:
        df[source_row_id_column] = np.arange(len(df))

    return df


def ensure_group_id(df: pd.DataFrame, mode: str, group_col: str) -> pd.DataFrame:
    df = df.copy()

    # group-based mode: ensure group column exists
    if mode == "group_based":
        if group_col not in df.columns:
            df[group_col] = np.arange(len(df))

    # normal mode: remove group column if present
    elif mode == "normal":
        if group_col in df.columns:
            df = df.drop(columns=[group_col])

    return df


def prepare_original_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()

    # mark original rows before augmentation
    df[is_augmented_column] = 0
    df[augmentation_round_column] = 0
    df[randomization_mode_column] = mode

    return df


def get_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> list[str]:
    # exclude target and metadata columns
    excluded_columns = {
        target_col,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
    }

    if group_col in df.columns:
        excluded_columns.add(group_col)

    if randomization_mode_column in df.columns:
        excluded_columns.add(randomization_mode_column)

    # select numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # return valid feature columns
    return [column for column in numeric_columns if column not in excluded_columns]


def augment_row(
    row: pd.Series,
    numeric_columns: list[str],
    rng: np.random.Generator,
    round_number: int,
    mode: str,
) -> pd.Series:
    new_row = row.copy()

    # add small noise to numeric columns
    for column in numeric_columns:
        value = row[column]
        if pd.notna(value):
            noise = rng.normal(loc=0.0, scale=noise_std)
            new_row[column] = value + noise

    # mark row as augmented
    new_row[is_augmented_column] = 1
    new_row[augmentation_round_column] = round_number
    new_row[randomization_mode_column] = mode

    return new_row


def generate_augmented_data(
    df: pd.DataFrame,
    copies: int,
    mode: str,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    # get numeric columns eligible for augmentation
    numeric_columns = get_numeric_feature_columns(df, target_col, group_col)

    # stop if no valid columns found
    if not numeric_columns:
        raise ValueError("no numeric feature columns found to augment")

    rng = np.random.default_rng(random_seed)
    augmented_rows = []

    # create augmented copies for each row
    for _, row in df.iterrows():
        for round_number in range(1, copies + 1):
            augmented_rows.append(
                augment_row(
                    row=row,
                    numeric_columns=numeric_columns,
                    rng=rng,
                    round_number=round_number,
                    mode=mode,
                )
            )

    # return empty dataframe if nothing generated
    if not augmented_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(augmented_rows, columns=df.columns)


def save_parquet_file(df: pd.DataFrame, filename: str) -> None:
    # save final dataframe as parquet file
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)

    print(f"saved: {output_path.name}")


def augment_parquet_file(
    input_filename: str = input_file,
    output_filename: str = output_file,
    copies: int = augment_copies,
    mode: str = randomization_mode,
    target_col: str = target_column,
    group_col: str = group_column,
) -> pd.DataFrame:

    # validate mode
    if mode not in {"normal", "group_based"}:
        raise ValueError("randomization mode must be 'normal' or 'group_based'")

    # load and prepare data
    df = load_parquet_file(input_filename)
    df = ensure_source_row_id(df)
    df = ensure_group_id(df, mode, group_col)
    df = prepare_original_rows(df, mode)

    # generate augmented data
    augmented_df = generate_augmented_data(df, copies, mode, target_col, group_col)

    # combine original and augmented rows
    final_df = pd.concat([df, augmented_df], ignore_index=True)

    save_parquet_file(final_df, output_filename)
    return final_df


def parse_arguments():
    # read command line switches from the user
    parser = argparse.ArgumentParser(
        description="parquet augmentation software for parkinsons dataset"
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
        "--group",
        type=str,
        default=group_column,
        help="group column name",
    )

    parser.add_argument(
        "--copies",
        type=int,
        default=augment_copies,
        help="number of augmented copies to create per row",
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
    # entry point of the program
    args = parse_arguments()

    print("starting parquet augmentation software")
    print(f"software directory: {script_dir}")
    print(f"input file: {args.input}")
    print(f"output file: {args.output}")
    print(f"target column: {args.target}")
    print(f"group column: {args.group}")
    print(f"copies: {args.copies}")
    print(f"randomization mode: {args.mode}")

    final_df = augment_parquet_file(
        input_filename=args.input,
        output_filename=args.output,
        copies=args.copies,
        mode=args.mode,
        target_col=args.target,
        group_col=args.group,
    )

    # print summary of results
    original_rows = (final_df[is_augmented_column] == 0).sum()
    augmented_rows = (final_df[is_augmented_column] == 1).sum()

    print(f"original rows: {original_rows}")
    print(f"augmented rows: {augmented_rows}")
    print(f"final rows: {len(final_df)}")
    print("process completed successfully")


if __name__ == "__main__":
    main()