import os
import pandas as pd
from pathlib import Path

Dataset_Dir = Path(__file__).resolve().parent / "DatasetWork"
save_dir = Path(__file__).resolve().parent / "EDA_Results"
save_dir.mkdir(exist_ok=True)

for file in os.listdir(Dataset_Dir):
    if file.endswith(".csv"):
        file_path = Dataset_Dir / file
        df = pd.read_csv(file_path)
        with open (save_dir / f"{file_path.stem}.txt", "w") as f:
            f.write(f"DataFrame for {file}:\n")
            f.write("\nInformation: \n")
            df.info(buf=f)
            f.write("\n Statistics: \n")
            f.write(str(df.describe()))
            f.write("\n Number of Null Values: \n")
            f.write(str(df.isna().sum()))
            f.write("\nShape: \n")
            f.write(str(df.shape))
            f.write("\nColumns: \n")
            f.write(str(df.columns.tolist()))
            f.write(str(df["label"].value_counts()))
            f.write("First 5 rows: \n")
            f.write(str(df.head(5)))
            f.write("\nLast 5 rows: \n")
            f.write(str(df.tail(5)))
            f.write("\n")