from pathlib import Path
import argparse
import pandas as pd


# get current script directory (used for input/output files)
script_dir = Path(__file__).resolve().parent

# default file names (used if not overridden)
input_file = "parkinsons_updrs.data"
output_file = "parkinsons.parquet"


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
    # build output file path in the same directory
    return script_dir / filename


def load_csv_file(filename: str) -> pd.DataFrame:
    # read csv or data file into pandas dataframe
    return pd.read_csv(get_input_path(filename))


def save_parquet_file(df: pd.DataFrame, filename: str) -> None:
    # save dataframe as parquet file
    output_path = get_output_path(filename)
    df.to_parquet(output_path, index=False)

    # confirm save to user
    print(f"saved: {output_path.name}")


def convert_to_parquet(
    input_filename: str = input_file,
    output_filename: str = output_file,
) -> pd.DataFrame:
    # load input dataset
    df = load_csv_file(input_filename)

    # print basic dataset info
    print(f"rows: {len(df)}")
    print(f"columns: {df.columns.tolist()}")

    # save dataset as parquet file
    save_parquet_file(df, output_filename)

    return df


def parse_arguments():
    # define command line arguments (cli switches)
    parser = argparse.ArgumentParser(
        description="convert csv/data file to parquet format"
    )

    # input file argument
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input file name (csv or data file)",
    )

    # output file argument
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output parquet file name",
    )

    return parser.parse_args()


def main() -> None:
    # read arguments from command line
    args = parse_arguments()

    # print run details
    print("starting parquet conversion")
    print(f"software directory: {script_dir}")
    print(f"input file: {args.input}")
    print(f"output file: {args.output}")

    # run conversion process
    convert_to_parquet(
        input_filename=args.input,
        output_filename=args.output,
    )

    print("conversion completed successfully")


if __name__ == "__main__":
    # program entry point
    main()