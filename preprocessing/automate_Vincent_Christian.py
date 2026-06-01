import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

RAW_DATA_PATTERN = "../Global Superstore.csv"
OUTPUT_DIR       = "namadataset_preprocessing"
TARGET_COL       = "Profit"
TARGET_BINARY    = "Profitable"
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
COLS_TO_CLIP     = ["Sales", "Discount", "Quantity"]
ID_KEYWORDS      = ["id", "name", "postal", "date", "city", "state",
                    "country", "region"]

def log(msg: str) -> None:
    print(f"[automate] {msg}")


def load_dataset(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file CSV ditemukan dengan pola: {pattern}\n"
            "Pastikan folder namadataset_raw/ berisi file CSV hasil unduhan."
        )
    path = files[0]
    log(f"Memuat dataset dari: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    log(f"Shape awal: {df.shape}")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ID_KEYWORDS)
    ]
    log(f"Menghapus kolom tidak relevan: {drop_cols}")
    return df.drop(columns=drop_cols, errors="ignore")


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_BINARY] = (df[TARGET_COL] >= 0).astype(int)
    pos = df[TARGET_BINARY].mean() * 100
    log(f"Target biner '{TARGET_BINARY}' dibuat  — positif: {pos:.1f}%")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    log(f"Duplikat dihapus: {before - len(df)} baris")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    remaining = df.isnull().sum().sum()
    log(f"Imputasi selesai — missing values tersisa: {remaining}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    log(f"Label encoding diterapkan pada {len(cat_cols)} kolom: {cat_cols}")
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    targets = [c for c in COLS_TO_CLIP if c in df.columns]
    for col in targets:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        log(f"Outlier '{col}': {n_out} nilai di-clip ke [{lo:.2f}, {hi:.2f}]")
    return df


def split_and_scale(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, TARGET_BINARY]]
    X = df[feature_cols]
    y = df[TARGET_BINARY]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log(f"Split selesai — train: {len(X_train)}, test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_outputs(X_train, X_test, y_train, y_test, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv",  index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv",  index=False)
    log(f"Output disimpan di folder: {out_dir}/")
    log("  X_train.csv  X_test.csv  y_train.csv  y_test.csv")


def run(raw_pattern: str, output_dir: str) -> None:
    log("=== Memulai pipeline preprocessing ===")

    df = load_dataset(raw_pattern)
    df = drop_irrelevant_columns(df)
    df = create_binary_target(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = encode_categoricals(df)
    df = clip_outliers(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)
    save_outputs(X_train, X_test, y_train, y_test, output_dir)

    log("=== Pipeline selesai ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automate preprocessing — Enhanced Superstore Sales Dataset"
    )
    parser.add_argument(
        "--data", default=RAW_DATA_PATTERN,
        help="Glob pattern menuju file CSV raw (default: ../namadataset_raw/**/*.csv)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help="Folder output hasil preprocessing (default: namadataset_preprocessing)"
    )
    args = parser.parse_args()
    run(args.data, args.output)