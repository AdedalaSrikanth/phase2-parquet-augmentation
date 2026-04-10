from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from phase2_parquet_augment import augment_parquet_file

script_dir = Path(__file__).resolve().parent

input_file = "input.parquet"
normal_file = "augmented_normal.parquet"
group_file = "augmented_group_based.parquet"
results_file = "model_comparison_results.csv"

target_column = "target"
group_column = "group_id"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"
randomization_mode_column = "randomization_mode"


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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        target_column,
        group_column,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
        randomization_mode_column,
    }

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric_columns if column not in excluded_columns]


def evaluate_original(df: pd.DataFrame) -> dict:
    feature_columns = get_feature_columns(df)

    x = df[feature_columns]
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return {
        "dataset": "original",
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }


def evaluate_normal(df: pd.DataFrame) -> dict:
    feature_columns = get_feature_columns(df)

    x = df[feature_columns]
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return {
        "dataset": "normal_augmented",
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }


def evaluate_group_based(df: pd.DataFrame) -> dict:
    feature_columns = get_feature_columns(df)

    x = df[feature_columns]
    y = df[target_column]
    groups = df[group_column]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(x, y, groups))

    x_train = x.iloc[train_index]
    x_test = x.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return {
        "dataset": "group_based_augmented",
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }


def save_results(results: list[dict], filename: str) -> None:
    output_path = script_dir / filename
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"saved: {output_path.name}")


def main() -> None:
    print("starting model comparison")

    augment_parquet_file(
        input_filename=input_file,
        output_filename=normal_file,
        copies=2,
        mode="normal",
    )

    augment_parquet_file(
        input_filename=input_file,
        output_filename=group_file,
        copies=2,
        mode="group_based",
    )

    original_df = load_parquet_file(input_file)
    normal_df = load_parquet_file(normal_file)
    group_df = load_parquet_file(group_file)

    results = []
    results.append(evaluate_original(original_df))
    results.append(evaluate_normal(normal_df))
    results.append(evaluate_group_based(group_df))

    save_results(results, results_file)

    print("model comparison completed successfully")


if __name__ == "__main__":
    main()