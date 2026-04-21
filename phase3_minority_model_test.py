# model evaluation for minority augmented dataset
# loads augmented data, trains model, and reports performance

from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


# current script directory
script_dir = Path(__file__).resolve().parent

# input dataset and output results file
input_file = "augmented_minority_only.parquet"
results_file = "minority_model_results.csv"

# column names
target_column = "target"
group_column = "group_id"


def get_file_path(filename: str) -> Path:
    # build full file path and check if it exists
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory as this software: {script_dir}"
        )
    return file_path


def get_output_path(filename: str) -> Path:
    # build output file path
    return script_dir / filename


def load_data(filename: str) -> pd.DataFrame:
    # load parquet dataset
    return pd.read_parquet(get_file_path(filename))


def get_features(df: pd.DataFrame, target_col: str, group_col: str) -> list[str]:
    # exclude non-feature columns
    excluded = {
        target_col,
        group_col,
        "source_row_id",
        "is_augmented",
        "augmentation_round",
        "randomization_mode",
    }

    # select numeric columns only
    numeric = df.select_dtypes(include="number").columns.tolist()

    return [col for col in numeric if col not in excluded]


def fill_missing_values(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # handle missing values before training
    df = df.copy()

    for col in df.columns:
        if col != target_col:
            if df[col].dtype in ["float64", "int64"]:
                # fill numeric columns with mean
                df[col] = df[col].fillna(df[col].mean())
            else:
                # fill non-numeric with 0
                df[col] = df[col].fillna(0)

    return df


def evaluate(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    dataset_name: str,
) -> dict:
    # prepare features, labels, and group ids
    features = get_features(df, target_col, group_col)

    x = df[features]
    y = df[target_col]
    groups = df[group_col]

    # split data using group-aware split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups))

    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # make predictions
    preds = model.predict(x_test)

    # return evaluation metrics
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }


def save_results(result: dict, filename: str) -> None:
    # save results to csv
    result_df = pd.DataFrame([result])
    output_path = get_output_path(filename)
    result_df.to_csv(output_path, index=False)
    print(f"saved: {output_path.name}")


def parse_arguments():
    # read command line switches from the user
    parser = argparse.ArgumentParser(
        description="model evaluation for minority augmented parquet dataset"
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
        help="output csv file name",
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
        "--dataset-name",
        type=str,
        default="minority_balanced",
        help="dataset name to store in results",
    )

    return parser.parse_args()


def main() -> None:
    # entry point of the script
    args = parse_arguments()

    print("testing minority balanced dataset")
    print(f"script folder: {script_dir}")
    print(f"input file: {args.input}")
    print(f"output file: {args.output}")
    print(f"target column: {args.target}")
    print(f"group column: {args.group}")
    print(f"dataset name: {args.dataset_name}")

    df = load_data(args.input)

    # clean missing values
    df = fill_missing_values(df, args.target)

    # evaluate model
    result = evaluate(
        df=df,
        target_col=args.target,
        group_col=args.group,
        dataset_name=args.dataset_name,
    )

    save_results(result, args.output)

    print(result)
    print("done")


if __name__ == "__main__":
    main()