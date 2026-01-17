import os
from typing import Tuple
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_FILE_DIR, "..", "dataset", "diabetes.csv")
OUTPUT_DIR = os.path.join(CURRENT_FILE_DIR, "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Path untuk menyimpan artifacts
IMPUTER_PATH = os.path.join(OUTPUT_DIR, "imputer_median.joblib")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_standard.joblib")
BP_PARAMS_PATH = os.path.join(OUTPUT_DIR, "bloodpressure_outlier_params.joblib")
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "diabetes_train.csv")
TEST_DATA_PATH = os.path.join(OUTPUT_DIR, "diabetes_test.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from the given path."""
    print(f"[INFO] Loading dataset from: {path}")

    try:
        df: pd.DataFrame = pd.read_csv(path)
        print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found at {path}")
        raise


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    print("[INFO] Splitting data into Train and Test (80:20)...")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    print(f"[INFO] Shape X_train: {X_train.shape}")
    print(f"[INFO] Shape X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


def handle_missing_values(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle hidden missing values (zeros) in specific columns.
    Uses SimpleImputer with Median strategy.
    """
    print("[INFO] Starting Missing Values Handling (Imputation)...")

    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    zeros_train = (X_train[zero_columns] == 0).sum().sum()
    zeros_test = (X_test[zero_columns] == 0).sum().sum()
    print(f"[LOG] Total zero values in medical columns (Train): {zeros_train}")
    print(f"[LOG] Total zero values in medical columns (Test) : {zeros_test}")

    imputer = SimpleImputer(missing_values=0, strategy="median")

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train[zero_columns]),
        columns=zero_columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test[zero_columns]),
        columns=zero_columns,
        index=X_test.index,
    )

    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    X_train_final[zero_columns] = X_train_imputed
    X_test_final[zero_columns] = X_test_imputed

    joblib.dump(imputer, IMPUTER_PATH)
    print(f"[INFO] Imputer saved successfully to {IMPUTER_PATH}")

    return X_train_final, X_test_final


def handle_outliers(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle specific outliers in the BloodPressure column.
    Values below the lower bound are replaced with the Train median.
    """
    print("[INFO] Starting Outlier Handling (BloodPressure)...")

    Q1 = X_train["BloodPressure"].quantile(0.25)
    Q3 = X_train["BloodPressure"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    median_val = X_train["BloodPressure"].median()

    print(f"[LOG] BloodPressure Lower Bound: {lower_bound}")
    print(f"[LOG] Replacement Median: {median_val}")

    def replace_bp_outliers(df, dataset_name):
        count = (df["BloodPressure"] < lower_bound).sum()
        df.loc[df["BloodPressure"] < lower_bound, "BloodPressure"] = median_val
        print(f"[LOG] {count} outlier BloodPressure ditangani pada {dataset_name}.")
        return df

    X_train = replace_bp_outliers(X_train, "Train Set")
    X_test = replace_bp_outliers(X_test, "Test Set")

    bp_params = {"lower_bound": lower_bound, "replacement_value": median_val}
    joblib.dump(bp_params, BP_PARAMS_PATH)
    print(f"[INFO] Outlier parameters saved to {BP_PARAMS_PATH}")

    return X_train, X_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize features using StandardScaler.
    """
    print("[INFO] Starting Feature Standardization (Scaling)...")

    scaler = StandardScaler()

    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_arr, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_arr, columns=X_test.columns, index=X_test.index
    )

    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved successfully to {SCALER_PATH}")

    return X_train_scaled, X_test_scaled


def save_processed_data(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Combine features and target, then save to CSV.
    """
    print("[INFO] Saving preprocessed data...")

    # Reset index for clean concatenation
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    train_final = pd.concat([X_train, y_train], axis=1)
    test_final = pd.concat([X_test, y_test], axis=1)

    train_final.to_csv(TRAIN_DATA_PATH, index=False)
    test_final.to_csv(TEST_DATA_PATH, index=False)

    print(f"[SUCCESS] Train data saved: {TRAIN_DATA_PATH} {train_final.shape}")
    print(f"[SUCCESS] Test data saved : {TEST_DATA_PATH} {test_final.shape}")


def main() -> None:
    print("=" * 50)
    print("START PREPROCESSING PIPELINE")
    print("=" * 50)

    # 1. Load Data
    df = load_data(DATASET_PATH)

    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Handle Missing Values (Imputation)
    X_train, X_test = handle_missing_values(X_train, X_test)

    # 4. Handle Outliers
    X_train, X_test = handle_outliers(X_train, X_test)

    # 5. Scaling
    X_train, X_test = scale_features(X_train, X_test)

    # 6. Save Data
    save_processed_data(X_train, y_train, X_test, y_test)

    print("=" * 50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()
