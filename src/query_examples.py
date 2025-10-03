# src/query_examples.py
import pandas as pd
import sqlite3

# Paths
CLEAN_CSV = "data/processed/credit_clean.csv"
DB_FILE   = "data/credit.db"
TABLE     = "credit_clean"

if __name__ == "__main__":
    # --- Pandas queries ---
    print("=== Pandas Examples ===")
    df = pd.read_csv(CLEAN_CSV)

    # 1. Average credit limit by education
    print("\nAvg credit limit by education:")
    print(df.groupby("EDUCATION")["LIMIT_BAL"].mean())

    # 2. Default rate by marriage status
    print("\nDefault rate by marriage status:")
    print(df.groupby("MARRIAGE")["DEFAULT"].mean())

    # 3. Correlation matrix for key vars
    print("\nCorrelation matrix:")
    print(df[["LIMIT_BAL", "AGE", "UTIL_RATIO", "PAY_STAT_AVG", "DEFAULT"]].corr())

    # 4. Top 5 customers by utilization ratio
    print("\nTop 5 customers by UTIL_RATIO:")
    print(df.nlargest(5, "UTIL_RATIO")[["ID", "LIMIT_BAL", "BILL_AMT1", "UTIL_RATIO", "DEFAULT"]])

    # --- SQL queries ---
    print("\n=== SQL Examples ===")
    with sqlite3.connect(DB_FILE) as conn:
        # 1. Average credit limit by education
        q1 = f"""
        SELECT EDUCATION, AVG(LIMIT_BAL) AS avg_limit
        FROM {TABLE}
        GROUP BY EDUCATION;
        """
        print("\nAvg credit limit by education:")
        print(pd.read_sql(q1, conn))

        # 2. Default rate by marriage status
        q2 = f"""
        SELECT MARRIAGE, AVG("DEFAULT") AS default_rate
        FROM {TABLE}
        GROUP BY MARRIAGE;
        """
        print("\nDefault rate by marriage status:")
        print(pd.read_sql(q2, conn))


        # 3. Age distribution (min, max, avg)
        q3 = f"""
        SELECT MIN(AGE) AS min_age,
               MAX(AGE) AS max_age,
               AVG(AGE) AS avg_age
        FROM {TABLE};
        """
        print("\nAge distribution:")
        print(pd.read_sql(q3, conn))

        # 4. Top 5 customers by UTIL_RATIO
        q4 = f"""
        SELECT ID, LIMIT_BAL, BILL_AMT1, UTIL_RATIO, "DEFAULT" AS DEFAULT_FLAG
        FROM {TABLE}
        ORDER BY UTIL_RATIO DESC
        LIMIT 5;
        """
        print("\nTop 5 customers by UTIL_RATIO:")
        print(pd.read_sql(q4, conn))
