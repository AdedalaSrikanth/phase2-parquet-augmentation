from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

script_dir = Path(__file__).resolve().parent

input_file = "augmented_minority_only.parquet"
results_file = "minority_model_results.csv"

target_column = "target"
group_column = "group_id"


def get_file_path(filename: str):
    file_path = script_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {filename}")
    return file_path


def load_data(filename: str):
    return pd.read_parquet(get_file_path(filename))


def get_features(df):
    excluded = {
        target_column,
        group_column,
        "source_row_id",
        "is_augmented",
        "augmentation_round",
        "randomization_mode",
    }

    numeric = df.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric if col not in excluded]


def evaluate(df):
    features = get_features(df)

    x = df[features]
    y = df[target_column]
    groups = df[group_column]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups))

    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    return {
        "dataset": "minority_balanced",
        "rows": len(df),
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }


def main():
    print("testing minority balanced dataset")

    df = load_data(input_file)
    result = evaluate(df)

    result_df = pd.DataFrame([result])
    result_df.to_csv(script_dir / results_file, index=False)

    print(result)
    print("saved:", results_file)


if __name__ == "__main__":
    main()