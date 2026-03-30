from pathlib import Path
import pandas as pd


script_dir = Path(__file__).resolve().parent

csv_file = "ai4i2020.csv"
output_file = "input.parquet"


def main() -> None:
    csv_path = script_dir / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(
            f"file not found: '{csv_file}'. place the file in the same directory as this software: {script_dir}"
        )

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "Air temperature [K]": "air_temperature_k",
            "Process temperature [K]": "process_temperature_k",
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Machine failure": "target",
        }
    )

    df = df[
        [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "target",
        ]
    ].copy()

    df["group_id"] = range(len(df))

    output_path = script_dir / output_file
    df.to_parquet(output_path, index=False)

    print(f"saved: {output_file}")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print("real parquet input created successfully")


if __name__ == "__main__":
    main()