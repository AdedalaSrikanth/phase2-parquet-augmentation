# model comparison between original and augmented datasets
# evaluates performance using logistic regression

from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings

from parquet_augment import augment_parquet_file


# current script directory
script_dir = Path(__file__).resolve().parent

# input and output files
input_file = "input.parquet"
normal_file = "augmented_normal.parquet"
group_file = "augmented_group_based.parquet"
results_file = "model_comparison_results.csv"

# column names
target_column = "target"
group_column = "group_id"
source_row_id_column = "source_row_id"
is_augmented_column = "is_augmented"
augmentation_round_column = "augmentation_round"
randomization_mode_column = "randomization_mode"


def get_file_path(filename: str) -> Path:
    # build file path and check existence
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"file not found: '{filename}'. place the file in the same directory: {script_dir}"
        )
    return file_path


def get_output_path(filename: str) -> Path:
    # build output path in script directory
    return script_dir / filename


def load_parquet_file(filename: str) -> pd.DataFrame:
    # load parquet dataset
    return pd.read_parquet(get_file_path(filename))


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    # remove non-feature columns
    excluded_columns = {
        target_col,
        group_col,
        source_row_id_column,
        is_augmented_column,
        augmentation_round_column,
        randomization_mode_column,
    }

    x = df.drop(columns=list(excluded_columns), errors="ignore").copy()
    y = df[target_col].copy()

    # separate numeric and categorical columns
    numeric_columns = x.select_dtypes(include=["number"]).columns.tolist()
    text_columns = x.select_dtypes(exclude=["number"]).columns.tolist()

    # fill missing values
    for column in numeric_columns:
        x[column] = x[column].fillna(x[column].mean())

    for column in text_columns:
        x[column] = x[column].fillna("missing")

    # convert categorical variables to numeric
    x = pd.get_dummies(x, columns=text_columns, drop_first=False)

    return x, y


def align_train_test_columns(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ensure train and test have same columns
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, x_test


def evaluate_original(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    dataset_name: str,
) -> dict:
    # evaluate model on original dataset
    x, y = prepare_features(df, target_col, group_col)

    # standard train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    x_train, x_test = align_train_test_columns(x_train, x_test)

    # train model
    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train)

    # predict and evaluate
    predictions = model.predict(x_test)

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def evaluate_normal(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    dataset_name: str,
) -> dict:
    # evaluate model on normally augmented dataset
    x, y = prepare_features(df, target_col, group_col)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    x_train, x_test = align_train_test_columns(x_train, x_test)

    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def evaluate_group_based(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    dataset_name: str,
) -> dict:
    # evaluate model using group-aware split
    x, y = prepare_features(df, target_col, group_col)
    groups = df[group_col]

    # ensure same group does not appear in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(x, y, groups))

    x_train = x.iloc[train_index].copy()
    x_test = x.iloc[test_index].copy()
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    x_train, x_test = align_train_test_columns(x_train, x_test)

    model = LogisticRegression(max_iter=3000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }


def save_results(results: list[dict], filename: str) -> None:
    # save comparison results to csv
    output_path = get_output_path(filename)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"saved: {output_path.name}")


def parse_arguments():
    # read command line switches from the user
    parser = argparse.ArgumentParser(
        description="model comparison for original, normal augmented, and group-based augmented parquet datasets"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input parquet file name",
    )

    parser.add_argument(
        "--normal-output",
        type=str,
        required=True,
        help="output parquet file name for normal augmentation",
    )

    parser.add_argument(
        "--group-output",
        type=str,
        required=True,
        help="output parquet file name for group-based augmentation",
    )

    parser.add_argument(
        "--results-output",
        type=str,
        required=True,
        help="output csv file name for model comparison results",
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
        default=2,
        help="number of augmented copies to create per row",
    )

    parser.add_argument(
        "--original-name",
        type=str,
        default="original",
        help="dataset name to store for original dataset results",
    )

    parser.add_argument(
        "--normal-name",
        type=str,
        default="normal_augmented",
        help="dataset name to store for normal augmented dataset results",
    )

    parser.add_argument(
        "--group-name",
        type=str,
        default="group_based_augmented",
        help="dataset name to store for group-based augmented dataset results",
    )

    return parser.parse_args()


def main() -> None:
    # suppress convergence warnings for cleaner output
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    args = parse_arguments()

    print("starting model comparison")
    print(f"script folder: {script_dir}")
    print(f"input file: {args.input}")
    print(f"normal augmented output file: {args.normal_output}")
    print(f"group-based augmented output file: {args.group_output}")
    print(f"results output file: {args.results_output}")
    print(f"target column: {args.target}")
    print(f"group column: {args.group}")
    print(f"copies: {args.copies}")

    # generate augmented datasets
    augment_parquet_file(
        input_filename=args.input,
        output_filename=args.normal_output,
        copies=args.copies,
        mode="normal",
    )

    augment_parquet_file(
        input_filename=args.input,
        output_filename=args.group_output,
        copies=args.copies,
        mode="group_based",
    )

    # load datasets
    original_df = load_parquet_file(args.input)
    normal_df = load_parquet_file(args.normal_output)
    group_df = load_parquet_file(args.group_output)

    # evaluate all datasets
    results = [
        evaluate_original(
            df=original_df,
            target_col=args.target,
            group_col=args.group,
            dataset_name=args.original_name,
        ),
        evaluate_normal(
            df=normal_df,
            target_col=args.target,
            group_col=args.group,
            dataset_name=args.normal_name,
        ),
        evaluate_group_based(
            df=group_df,
            target_col=args.target,
            group_col=args.group,
            dataset_name=args.group_name,
        ),
    ]

    # save results
    save_results(results, args.results_output)

    print("model comparison completed successfully")


if __name__ == "__main__":
    main()