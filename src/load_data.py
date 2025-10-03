# src/prepare_data.py
import os
import sqlite3
import pandas as pd

# --- Paths / names ---
RAW_FILE   = "data/raw/creditCardDefault.csv"
OUT_DIR    = "data/processed"
OUT_CSV    = f"{OUT_DIR}/credit_clean.csv"
DB_FILE    = "data/credit.db"
RAW_TBL    = "credit_raw"
CLEAN_TBL  = "credit_clean"
TARGET_COL = "default payment next month"   # keep exact header (with spaces)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep original CAPS headers. Add a few informative features:
      - UTIL_RATIO = BILL_AMT1 / LIMIT_BAL (protect against div-by-zero)
      - BILL_SUM = sum(BILL_AMT1..BILL_AMT6)
      - PAY_SUM  = sum(PAY_AMT1..PAY_AMT6)
      - PAY_STAT_AVG = mean(PAY_0, PAY_2..PAY_6)
      - DEFAULT = duplicate of target column (clean, no spaces) for modeling later
    """
    # Utilization ratio (clip just for sanity viewing)
    util = (df["BILL_AMT1"] / df["LIMIT_BAL"]).where(df["LIMIT_BAL"] != 0)
    df["UTIL_RATIO"] = util.clip(lower=0, upper=3)

    # Aggregates for bills & payments
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols  = [f"PAY_AMT{i}"  for i in range(1, 7)]
    df["BILL_SUM"] = df[bill_cols].sum(axis=1)
    df["PAY_SUM"]  = df[pay_cols].sum(axis=1)

    # Average repayment status over recent months
    pay_stat_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["PAY_STAT_AVG"] = df[pay_stat_cols].mean(axis=1)

    # Clean duplicate of the target (keep original too)
    df["DEFAULT"] = df[TARGET_COL].astype("int64")

    return df

if __name__ == "__main__":
    # 1) Load raw CSV (CAPS headers kept)
    df_raw = pd.read_csv(RAW_FILE)

    # 2) Ensure output dir exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3) Write RAW table to SQLite (nice to keep for reference)
    with sqlite3.connect(DB_FILE) as conn:
        df_raw.to_sql(RAW_TBL, conn, if_exists="replace", index=False)

    # 4) Build features
    df_clean = build_features(df_raw.copy())

    # 5) Save processed CSV
    df_clean.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved processed CSV -> {OUT_CSV}")

    # 6) Write CLEAN table to SQLite
    with sqlite3.connect(DB_FILE) as conn:
        df_clean.to_sql(CLEAN_TBL, conn, if_exists="replace", index=False)
    print(f"✅ Wrote tables '{RAW_TBL}' and '{CLEAN_TBL}' to {DB_FILE}")

    # Minimal confirmation
    print("Rows:", len(df_clean), "| Columns:", len(df_clean.columns))

    