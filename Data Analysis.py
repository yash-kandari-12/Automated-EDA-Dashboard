import argparse
 
import numpy as np
import pandas as pd
 
from EDA_files import build_report
 
# In the Pima Indians dataset a 0 in these columns is physically impossible,
# so it really means "not measured".
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
 
 
def clean(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")
    cols = [c for c in ZERO_MEANS_MISSING if c in df.columns]
    zeros = (df[cols] == 0).sum()
    print("Hidden missing values (zeros) found:")
    print(zeros[zeros > 0].to_string() or "  none")
    df[cols] = df[cols].replace(0, np.nan)
    return df
 
 
def baseline_model(df, target):
    """Cross-validated random forest. Imputation lives inside the pipeline
    so the median is learned only from training folds (no data leakage)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
 
    X, y = df.drop(columns=[target]), df[target]
    X = X.select_dtypes("number")
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"\nBaseline Random Forest ROC-AUC (5-fold): {auc.mean():.3f} +/- {auc.std():.3f}")
    pipe.fit(X, y)
    imp = pd.Series(pipe[-1].feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.rename("importance").to_csv("feature_importance.csv")
    print("\nFeature importance:")
    print(imp.round(3).to_string())
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", default="Outcome")
    a = ap.parse_args()
 
    raw = pd.read_csv(a.path)
    if a.target not in raw.columns:
        raise SystemExit(f"Target '{a.target}' not found. Columns: {list(raw.columns)}")
 
    print("=== Raw report ===")
    build_report(a.path, a.target, "report_raw.html")
 
    print("\n=== Cleaning ===")
    cleaned = clean(raw)
    cleaned.to_csv("diabetes_clean.csv", index=False)
 
    print("\n=== Clean report ===")
    build_report("diabetes_clean.csv", a.target, "report_clean.html")
 
    if cleaned[a.target].nunique() == 2:
        baseline_model(cleaned, a.target)