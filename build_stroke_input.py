from pathlib import Path
import argparse
import pandas as pd


# get current script directory
script_dir = Path(__file__).resolve().parent

# default file names
input_file = "stroke.csv"
output_file = "input.parquet"

# default column names
input_target_column = "stroke"
output_target_column = "target"


def get_input_path(filename: str) -> Path:
    # build full path for input file
    file_path = script_dir / filename

    # check if file exists before reading
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory as this software: {script_dir}"
        )

    return file_path


def get_output_path(filename: str) -> Path:
    # build output file path
    return script_dir / filename


def load_csv_file(filename: str) -> pd.DataFrame:
    # load csv file into dataframe
    return pd.read_csv(get_input_path(filename))


def rename_target_column(
    df: pd.DataFrame,
    old_target: str,
    new_target: str,
) -> pd.DataFrame:
    # rename target column to standard name
    df = df.copy()
    return df.rename(columns={old_target: new_target})


def encode_categorical_columns(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    # convert categorical columns to numeric using factorization
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object" and col != target_col:
            df[col] = pd.factorize(df[col])[0]

    # ensure target column is numeric
    if df[target_col].dtype == "object":
        df[target_col] = pd.factorize(df[target_col])[0]

    return df


def save_parquet_file(df: pd.DataFrame, filename: str) -> None:
    # save dataframe as parquet file
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)

    print(f"saved: {output_path.name}")


def build_input_parquet(
    input_filename: str = input_file,
    output_filename: str = output_file,
    input_target: str = input_target_column,
    output_target: str = output_target_column,
) -> pd.DataFrame:
    # load dataset
    df = load_csv_file(input_filename)

    # rename target column
    df = rename_target_column(df, input_target, output_target)

    # encode categorical columns
    df = encode_categorical_columns(df, output_target)

    # save processed dataset
    save_parquet_file(df, output_filename)

    return df


def parse_arguments():
    # define command line arguments
    parser = argparse.ArgumentParser(
        description="prepare stroke dataset and convert to parquet format"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input csv file name",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output parquet file name",
    )

    parser.add_argument(
        "--input-target",
        type=str,
        default=input_target_column,
        help="original target column name",
    )

    parser.add_argument(
        "--output-target",
        type=str,
        default=output_target_column,
        help="target column name in output file",
    )

    return parser.parse_args()


def main() -> None:
    # read command line arguments
    args = parse_arguments()

    print("starting stroke dataset preprocessing")
    print(f"software directory: {script_dir}")
    print(f"input file: {args.input}")
    print(f"output file: {args.output}")
    print(f"input target column: {args.input_target}")
    print(f"output target column: {args.output_target}")

    df = build_input_parquet(
        input_filename=args.input,
        output_filename=args.output,
        input_target=args.input_target,
        output_target=args.output_target,
    )

    print(df.head())
    print("process completed successfully")


if __name__ == "__main__":
    # program entry point
    main()