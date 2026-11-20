"""
Pelatihan & Evaluasi Model Regresi Gaji Profesional Data, AI, dan Teknologi
IT6043 - Machine Learning | Kelompok GodBless (FTI UKDW)

Membandingkan 3 algoritma regresi:
1. Ridge Regression
2. Random Forest Regressor
3. HistGradientBoosting Regressor (Gradient Boosting)

Menyimpan pipeline model terbaik ke `model.pkl`.
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = [
    os.path.join(BASE_DIR, "Proposal", "ai_jobs_salaries_clean.csv"),
    os.path.join(BASE_DIR, "ai_jobs_salaries_clean.csv"),
]


def load_dataset() -> pd.DataFrame:
    for path in DATA_PATHS:
        if os.path.exists(path):
            print(f"[INFO] Memuat dataset dari: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("File ai_jobs_salaries_clean.csv tidak ditemukan.")


def main():
    df = load_dataset()
    print(f"[INFO] Dimensi dataset: {df.shape[0]:,} baris x {df.shape[1]} kolom")

    # Feature engineering sesuai spesifikasi aplikasi client (app.py / prediksiGaji.py)
    df["is_us_company"] = (df["company_location"] == "US").astype(int)
    df["same_country"] = (df["company_location"] == df["employee_residence"]).astype(int)

    feature_cols = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "remote_ratio",
        "company_location",
        "company_size",
        "is_us_company",
        "same_country",
    ]

    X = df[feature_cols]
    # Transformasi log1p pada variabel target untuk menangani right-skewness (skew = 1.48)
    y = np.log1p(df["salary_in_usd"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ord_exp",
                OrdinalEncoder(
                    categories=[["EN", "MI", "SE", "EX"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["experience_level"],
            ),
            (
                "ord_size",
                OrdinalEncoder(
                    categories=[["S", "M", "L"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["company_size"],
            ),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=False,
                ),
                ["employment_type", "job_title", "employee_residence", "company_location"],
            ),
            (
                "num",
                StandardScaler(),
                ["work_year", "remote_ratio", "is_us_company", "same_country"],
            ),
        ]
    )

    models = {
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting (HistGBM)": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_depth=10,
            random_state=42,
        ),
    }

    best_name = None
    best_pipe = None
    best_r2 = -np.inf

    print("\n" + "=" * 82)
    print(f"{'Model Regresi':30s} | {'R² (Log)':>10s} | {'MAE (USD)':>14s} | {'RMSE (USD)':>14s}")
    print("-" * 82)

    for name, reg in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", reg)])
        pipe.fit(X_train, y_train)

        pred_log = pipe.predict(X_test)
        pred_usd = np.expm1(pred_log)
        true_usd = np.expm1(y_test)

        r2_log = r2_score(y_test, pred_log)
        mae_usd = mean_absolute_error(true_usd, pred_usd)
        rmse_usd = np.sqrt(mean_squared_error(true_usd, pred_usd))

        print(f"{name:30s} | {r2_log:10.4f} | ${mae_usd:13,.2f} | ${rmse_usd:13,.2f}")

        if r2_log > best_r2:
            best_r2 = r2_log
            best_name = name
            best_pipe = pipe

    print("=" * 82)
    print(f"[BEST MODEL] {best_name} (R² Log = {best_r2:.4f})")

    out_root = os.path.join(BASE_DIR, "model.pkl")
    joblib.dump(best_pipe, out_root)
    print(f"[SAVED] Model disimpan ke: {out_root}")


if __name__ == "__main__":
    main()
