from pathlib import Path
import pandas as pd

script_dir = Path(__file__).resolve().parent

input_file = "input.parquet"
augmented_file = "augmented_output.parquet"
analysis_output_file = "analysis_results.csv"

target_column = "target"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"


def get_file_path(filename: str) -> Path:
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory: {script_dir}"
        )
    return file_path


def load_parquet_file(filename: str) -> pd.DataFrame:
    file_path = get_file_path(filename)
    return pd.read_parquet(file_path)


def analyze_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    distribution = df[target_column].value_counts().reset_index()
    distribution.columns = ["class", "count"]
    return distribution


def analyze_augmentation_by_source(df: pd.DataFrame) -> pd.DataFrame:
    augmented_rows = df[df[is_augmented_column] == 1]

    counts = (
        augmented_rows
        .groupby(source_row_id_column)
        .size()
        .reset_index(name="augmented_samples")
    )

    return counts


def save_results(df: pd.DataFrame, filename: str):
    output_path = script_dir / filename
    df.to_csv(output_path, index=False)
    print(f"saved: {output_path.name}")


def main():

    print("starting phase 3 analysis")

    original_df = load_parquet_file(input_file)
    augmented_df = load_parquet_file(augmented_file)

    print("analyzing class distribution")

    original_distribution = analyze_class_distribution(original_df)
    augmented_distribution = analyze_class_distribution(augmented_df)

    print("analyzing augmentation lineage")

    augmentation_counts = analyze_augmentation_by_source(augmented_df)

    save_results(original_distribution, "original_distribution.csv")
    save_results(augmented_distribution, "augmented_distribution.csv")
    save_results(augmentation_counts, "augmentation_by_source.csv")

    print("analysis completed successfully")


if __name__ == "__main__":
    main()