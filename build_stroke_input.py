import pandas as pd

df = pd.read_csv("stroke.csv")

# rename target column 
df = df.rename(columns={"stroke": "target"}) 

# convert categorical columns
for col in df.columns:
    if df[col].dtype == "object" and col != "target":
        df[col] = pd.factorize(df[col])[0]

# convert target if needed
if df["target"].dtype == "object":
    df["target"] = pd.factorize(df["target"])[0]

df.to_parquet("input.parquet", index=False)

print("saved: input.parquet")
print(df.head())