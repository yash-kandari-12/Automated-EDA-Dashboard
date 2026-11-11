import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path

# ------------------------------------------------
# STEP 2: DATASET LOADING MODULE
# ------------------------------------------------

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset directory
DATASET_DIR = BASE_DIR / "datasets"

# Dataset filename
DATASET_PATH = DATASET_DIR / "diabetes.csv"


def load_dataset(file_path):

    print("\n========== DATASET LOADING ==========\n")

    # Check whether file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load CSV or Excel
    extension = file_path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file format. Use CSV or Excel."
        )

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty!")

    print("Dataset loaded successfully!")

    return df


# Load dataset
df = load_dataset(DATASET_PATH)


# ------------------------------------------------
# BASIC DATASET INFORMATION
# ------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\n========== LOADING COMPLETED ==========")
# ------------------------------------------------
# STEP 3: DATASET OVERVIEW AND STATISTICAL SUMMARY
# ------------------------------------------------

def dataset_overview(df):
    print("\n========== DATASET OVERVIEW ==========\n")
    rows, columns = df.shape
    print("Total Rows:", rows)
    print("Total Columns:", columns)
    print("\nCOLUMN INFORMATION:")
    print(df.dtypes)
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()
    print("\nNUMERICAL COLUMNS:")
    print(numerical_columns)
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    print("\nCATEGORICAL COLUMNS:")
    print(categorical_columns)
    print("\nSTATISTICAL SUMMARY:")
    print(df.describe(include="all").T)
    print("\nUNIQUE VALUES:")
    print(df.nunique())
    memory = df.memory_usage(deep=True).sum()
    print("\nMEMORY USAGE:")
    print(f"{memory / 1024:.2f} KB")
    print("\n========== OVERVIEW COMPLETED ==========")
dataset_overview(df)
# ------------------------------------------------
# STEP 4: MISSING VALUE ANALYSIS
# ------------------------------------------------

def missing_value_analysis(df):
    print("\n========== MISSING VALUE ANALYSIS ==========\n")
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells) * 100
    print("Total Missing Values:", total_missing)
    print("Overall Missing Percentage:",
          round(missing_percentage, 2), "%")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Values": missing_count,
        "Missing Percentage": missing_percent
    })

    missing_summary = missing_summary[
        missing_summary["Missing Values"] > 0
    ]
    print("\nMISSING VALUE SUMMARY:")
    if missing_summary.empty:
        print("No missing values found!")
    else:
        print(missing_summary.sort_values(
            by="Missing Percentage",
            ascending=False
        ))
    print("\nMISSING VALUE SEVERITY:")
    for column in df.columns:
        percentage = missing_percent[column]
        if percentage == 0:
            continue
        elif percentage < 5:
            severity = "Low"
        elif percentage < 30:
            severity = "Medium"
        else:
            severity = "High"
        print(f"{column}: {severity} "
              f"({percentage:.2f}%)")
    print("\n========== MISSING VALUE ANALYSIS COMPLETED ==========")
missing_value_analysis(df)
# ------------------------------------------------
# STEP 5: DUPLICATE RECORD ANALYSIS
# ------------------------------------------------

def duplicate_analysis(df):

    print("\n========== DUPLICATE ANALYSIS ==========\n")

    # Total number of rows
    total_rows = len(df)

    # Identify duplicate rows
    duplicate_rows = df.duplicated()

    # Count duplicates
    total_duplicates = duplicate_rows.sum()

    # Calculate duplicate percentage
    if total_rows > 0:
        duplicate_percentage = (
            total_duplicates / total_rows
        ) * 100
    else:
        duplicate_percentage = 0

    print("Total Rows:", total_rows)

    print("Total Duplicate Rows:", total_duplicates)

    print(
        "Duplicate Percentage:",
        round(duplicate_percentage, 2), "%"
    )

    # Display duplicate records
    if total_duplicates > 0:

        print("\nDUPLICATE RECORDS:")

        print(df[duplicate_rows].head(10))

    else:

        print("\nNo duplicate records found!")

    # Duplicate analysis summary
    print("\nDUPLICATE ANALYSIS SUMMARY:")

    print("Unique Rows:", total_rows - total_duplicates)

    print("Duplicate Rows:", total_duplicates)

    print("\n========== DUPLICATE ANALYSIS COMPLETED ==========")


# Execute Step 5
duplicate_analysis(df)
# ------------------------------------------------
# STEP 6: DATA TYPE ANALYSIS
# ------------------------------------------------

def data_type_analysis(df):

    print("\n========== DATA TYPE ANALYSIS ==========\n")

    numerical = []
    categorical = []
    datetime_columns = []
    boolean_columns = []
    other_columns = []

    for column in df.columns:

        dtype = df[column].dtype

        if pd.api.types.is_bool_dtype(dtype):
            boolean_columns.append(column)

        elif pd.api.types.is_numeric_dtype(dtype):
            numerical.append(column)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_columns.append(column)

        elif dtype == "object":

            # Check whether object column contains dates
            values = df[column].dropna()

            if len(values) > 0:
                parsed = pd.to_datetime(
                    values,
                    errors="coerce"
                )

                date_ratio = parsed.notna().mean()

                if date_ratio >= 0.8:
                    datetime_columns.append(column)
                else:
                    categorical.append(column)

            else:
                categorical.append(column)

        elif isinstance(dtype, pd.CategoricalDtype):
            categorical.append(column)

        else:
            other_columns.append(column)

    print("Numerical Columns:", numerical)

    print("\nCategorical Columns:", categorical)

    print("\nDatetime Columns:", datetime_columns)

    print("\nBoolean Columns:", boolean_columns)

    print("\nOther Columns:", other_columns)

    print("\n========== DATA TYPE ANALYSIS COMPLETED ==========")

    return numerical, categorical, datetime_columns
# Step 6
data_type_analysis(df)
# ------------------------------------------------
# STEP 7: OUTLIER DETECTION
# ------------------------------------------------

def outlier_detection(df):

    print("\n========== OUTLIER DETECTION ==========\n")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    # Exclude boolean and likely ID columns if necessary
    excluded_columns = []

    columns_to_analyze = [
        col for col in numerical_columns
        if col not in excluded_columns
        and df[col].nunique() >= 5
    ]

    outlier_summary = []

    # Track rows containing at least one IQR outlier
    outlier_rows = pd.Series(False, index=df.index)

    for column in columns_to_analyze:

        data = df[column].dropna()

        if len(data) < 2:
            continue

        # IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        iqr_mask = (
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        )

        iqr_count = iqr_mask.sum()

        outlier_rows |= iqr_mask

        # Z-Score Method
        mean = data.mean()
        std = data.std(ddof=0)

        if std != 0:

            z_scores = (df[column] - mean) / std

            z_mask = z_scores.abs() > 3

            z_count = z_mask.sum()

        else:
            z_count = 0

        outlier_summary.append({

            "Column": column,

            "IQR Outliers": iqr_count,

            "Z-Score Outliers": z_count,

            "Lower Limit": round(lower_limit, 2),

            "Upper Limit": round(upper_limit, 2)

        })

    # Display results
    if outlier_summary:

        result = pd.DataFrame(outlier_summary)

        print(result.to_string(index=False))

    else:

        print("No suitable numerical columns found.")

    total_outlier_rows = outlier_rows.sum()

    print("\nTotal Rows Containing IQR Outliers:",
          total_outlier_rows)

    print("\n========== OUTLIER DETECTION COMPLETED ==========")

    return outlier_summary, outlier_rows
# Step 7
outlier_summary, outlier_rows = outlier_detection(df)
# ------------------------------------------------
# STEP 8: DATA QUALITY SCORE
# ------------------------------------------------

def data_quality_score(df, outlier_rows):

    print("\n========== DATA QUALITY SCORE ==========\n")

    total_rows = len(df)
    total_cells = df.size

    if total_rows == 0 or total_cells == 0:

        print("Cannot calculate score for empty dataset.")

        return None

    # 1. Data Completeness
    missing_percentage = (
        df.isnull().sum().sum() / total_cells
    ) * 100

    completeness_score = 100 - missing_percentage

    # 2. Data Uniqueness
    duplicate_percentage = (
        df.duplicated().sum() / total_rows
    ) * 100

    uniqueness_score = 100 - duplicate_percentage

    # 3. Outlier-free Rows
    outlier_percentage = (
        outlier_rows.sum() / total_rows
    ) * 100

    outlier_free_score = 100 - outlier_percentage

    # Calculate weighted score
    final_score = (
        completeness_score * 0.40 +
        uniqueness_score * 0.30 +
        outlier_free_score * 0.30
    )

    # Keep score between 0 and 100
    final_score = max(0, min(100, final_score))

    # Grade
    if final_score >= 90:
        grade = "Excellent"

    elif final_score >= 75:
        grade = "Good"

    elif final_score >= 60:
        grade = "Average"

    else:
        grade = "Poor"

    print("Completeness Score:",
          round(completeness_score, 2))

    print("Uniqueness Score:",
          round(uniqueness_score, 2))

    print("Outlier-Free Score:",
          round(outlier_free_score, 2))

    print("\nFINAL DATA QUALITY SCORE:",
          round(final_score, 2), "/ 100")

    print("DATASET GRADE:", grade)

    print("\n========== QUALITY ANALYSIS COMPLETED ==========")

    return final_score

# Step 8
score = data_quality_score(df, outlier_rows)
# ------------------------------------------------
# STEP 9: DESCRIPTIVE STATISTICAL ANALYSIS
# ------------------------------------------------

def descriptive_statistics(df):

    print("\n========== DESCRIPTIVE STATISTICS ==========\n")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # NUMERICAL ANALYSIS

    print("\nNUMERICAL STATISTICS:")

    if len(numerical_columns) > 0:

        numerical_stats = df[numerical_columns].describe().T

        numerical_stats["Median"] = df[numerical_columns].median()

        numerical_stats["Mode"] = df[numerical_columns].mode().iloc[0]

        numerical_stats["Variance"] = df[numerical_columns].var()

        numerical_stats["Range"] = (
            df[numerical_columns].max() -
            df[numerical_columns].min()
        )

        numerical_stats["Skewness"] = df[numerical_columns].skew()

        numerical_stats["Kurtosis"] = df[numerical_columns].kurtosis()

        print(numerical_stats.round(3))

    else:

        print("No numerical columns found.")

    # CATEGORICAL ANALYSIS

    print("\nCATEGORICAL STATISTICS:")

    if len(categorical_columns) > 0:

        for column in categorical_columns:

            print(f"\nColumn: {column}")

            print("Unique Values:", df[column].nunique())

            print("Most Frequent Value:",
                  df[column].mode().iloc[0]
                  if not df[column].mode().empty
                  else "No values")

            print("Value Frequency:")

            print(df[column].value_counts().head(10))

    else:

        print("No categorical columns found.")

    print("\n========== STATISTICAL ANALYSIS COMPLETED ==========")

# Step 9
descriptive_statistics(df)
# ------------------------------------------------
# STEP 10: NUMERICAL DATA ANALYSIS
# ------------------------------------------------

def numerical_analysis(df):

    print("\n========== NUMERICAL DATA ANALYSIS ==========\n")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Exclude Boolean columns
    numerical_columns = [
        col for col in numerical_columns
        if not pd.api.types.is_bool_dtype(df[col])
    ]

    if not numerical_columns:
        print("No numerical columns found.")
        return None

    results = []

    for column in numerical_columns:

        data = df[column].dropna()

        if data.empty:
            continue

        results.append({

            "Column": column,

            "Count": data.count(),

            "Mean": data.mean(),

            "Median": data.median(),

            "Minimum": data.min(),

            "Maximum": data.max(),

            "Standard Deviation": data.std(),

            "Skewness": data.skew(),

            "Unique Values": data.nunique()

        })

    result_df = pd.DataFrame(results)

    print(result_df.round(3).to_string(index=False))

    print("\n========== NUMERICAL ANALYSIS COMPLETED ==========")

    return result_df

numerical_results = numerical_analysis(df)
# ------------------------------------------------
# STEP 11: CATEGORICAL DATA ANALYSIS
# ------------------------------------------------

def categorical_analysis(df):

    print("\n========== CATEGORICAL DATA ANALYSIS ==========\n")

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not categorical_columns:
        print("No categorical columns found.")
        return {}

    category_results = {}

    for column in categorical_columns:

        print(f"\nCOLUMN: {column}")

        data = df[column].dropna()

        if data.empty:
            print("No valid values found.")
            continue

        # Unique categories
        unique_count = data.nunique()

        # Category frequency
        frequency = data.value_counts()

        # Percentage distribution
        percentage = data.value_counts(
            normalize=True
        ) * 100

        print("Unique Categories:", unique_count)

        print("Most Frequent Category:", frequency.index[0])

        print("Least Frequent Category:", frequency.index[-1])

        # Summary table
        summary = pd.DataFrame({

            "Frequency": frequency,

            "Percentage": percentage

        })

        print("\nCategory Distribution:")

        print(summary.head(10).round(2))

        category_results[column] = summary

    print("\n========== CATEGORICAL ANALYSIS COMPLETED ==========")

    return category_results

# Step 11
categorical_results = categorical_analysis(df)

# ------------------------------------------------
# STEP 12: CORRELATION ANALYSIS
# ------------------------------------------------

def correlation_analysis(df):

    print("\n========== CORRELATION ANALYSIS ==========\n")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numerical_columns = [
        col for col in numerical_columns
        if not pd.api.types.is_bool_dtype(df[col])
    ]

    if len(numerical_columns) < 2:

        print("At least two numerical columns are required.")

        return None

    # Calculate correlation
    correlation_matrix = df[
        numerical_columns
    ].corr(method="pearson")

    print("CORRELATION MATRIX:\n")

    print(correlation_matrix.round(3))

    # Find highly correlated feature pairs
    print("\nHIGHLY CORRELATED FEATURES:\n")

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    pairs = upper_triangle.stack().reset_index()

    pairs.columns = [
        "Feature 1",
        "Feature 2",
        "Correlation"
    ]

    pairs["Absolute Correlation"] = (
        pairs["Correlation"].abs()
    )

    strong_pairs = pairs[
        pairs["Absolute Correlation"] >= 0.7
    ].sort_values(
        by="Absolute Correlation",
        ascending=False
    )

    if strong_pairs.empty:

        print("No strongly correlated pairs found.")

    else:

        print(
            strong_pairs.to_string(index=False)
        )

    print("\n========== CORRELATION ANALYSIS COMPLETED ==========")

    return correlation_matrix, strong_pairs

# Step 12
correlation_results = correlation_analysis(df)
# ------------------------------------------------
# STEP 13: FEATURE RELATIONSHIP ANALYSIS
# ------------------------------------------------

def feature_relationship_analysis(df):

    print("\n========== FEATURE RELATIONSHIP ANALYSIS ==========\n")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    numerical_columns = [
        col for col in numerical_columns
        if not pd.api.types.is_bool_dtype(df[col])
    ]

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    results = {}

    # --------------------------------------------
    # 1. NUMERICAL VS NUMERICAL
    # --------------------------------------------

    print("\n1. NUMERICAL VS NUMERICAL\n")

    if len(numerical_columns) >= 2:

        correlation = df[numerical_columns].corr()

        results["numerical_correlation"] = correlation

        print(correlation.round(3))

    else:

        print("Not enough numerical columns.")

    # --------------------------------------------
    # 2. CATEGORICAL VS NUMERICAL
    # --------------------------------------------

    print("\n2. CATEGORICAL VS NUMERICAL\n")

    for cat_col in categorical_columns:

        # Avoid columns with too many categories
        if df[cat_col].nunique() > 20:
            continue

        for num_col in numerical_columns:

            grouped = df.groupby(
                cat_col,
                observed=True
            )[num_col].agg(
                ["count", "mean", "median"]
            )

            if grouped.empty:
                continue

            print(f"\n{cat_col} vs {num_col}")

            print(grouped.round(2))

            results[
                f"{cat_col}_vs_{num_col}"
            ] = grouped

    # --------------------------------------------
    # 3. CATEGORICAL VS CATEGORICAL
    # --------------------------------------------

    print("\n3. CATEGORICAL VS CATEGORICAL\n")

    selected_categories = [
        col for col in categorical_columns
        if df[col].nunique() <= 10
    ]

    if len(selected_categories) >= 2:

        for i in range(len(selected_categories)):

            for j in range(i + 1, len(selected_categories)):

                col1 = selected_categories[i]

                col2 = selected_categories[j]

                cross_table = pd.crosstab(
                    df[col1],
                    df[col2]
                )

                print(f"\n{col1} vs {col2}")

                print(cross_table)

                results[
                    f"{col1}_vs_{col2}"
                ] = cross_table

    else:

        print("Not enough suitable categorical columns.")

    print("\n========== FEATURE RELATIONSHIP ANALYSIS COMPLETED ==========")

    return results
# Step 13
relationship_results = feature_relationship_analysis(df)

from pathlib import Path
import re

CHART_DIR = Path(__file__).resolve().parent / "charts"

CHART_DIR.mkdir(exist_ok=True)
# ============================================================
# STEP 14: DISTRIBUTION PLOTS
# ============================================================

def distribution_plots(df):

    print("\n========== STEP 14: DISTRIBUTION PLOTS ==========")

    # Select numerical columns
    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numerical_cols) == 0:
        print("No numerical columns found!")
        return

    print("Numerical columns:", numerical_cols)

    # Create output folder
    os.makedirs("EDA_Results", exist_ok=True)

    # ------------------------------------------------
    # 1. HISTOGRAM
    # ------------------------------------------------

    df[numerical_cols].hist(
        figsize=(15, 12),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig("EDA_Results/histograms.png")

    plt.show()

    # ------------------------------------------------
    # 2. KDE PLOTS
    # ------------------------------------------------

    for col in numerical_cols:

        plt.figure(figsize=(8, 5))

        sns.kdeplot(
            data=df,
            x=col,
            fill=True
        )

        plt.title(f"KDE Distribution of {col}")

        plt.tight_layout()

        plt.savefig(
            f"EDA_Results/KDE_{col}.png"
        )

        plt.show()

    # ------------------------------------------------
    # 3. BOX PLOTS
    # ------------------------------------------------

    for col in numerical_cols:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            data=df,
            x=col
        )

        plt.title(f"Box Plot of {col}")

        plt.tight_layout()

        plt.savefig(
            f"EDA_Results/Boxplot_{col}.png"
        )

        plt.show()

    print("\nStep 14 completed successfully!")
    print("All plots saved inside EDA_Results folder.")
distribution_plots(df)
# ============================================================
# STEP 15: OUTLIER VISUALIZATION
# ============================================================

def outlier_visualization(df):

    print("\n========== STEP 15: OUTLIER VISUALIZATION ==========")

    # Select numerical columns
    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numerical_cols) == 0:
        print("No numerical columns found!")
        return

    # Create folder
    os.makedirs("EDA_Results", exist_ok=True)

    # Generate box plots
    for col in numerical_cols:

        if df[col].nunique() < 2:
            continue

        plt.figure(figsize=(10, 5))

        sns.boxplot(
            x=df[col],
            color="skyblue"
        )

        plt.title(f"Outlier Visualization - {col}")
        plt.xlabel(col)

        plt.tight_layout()

        plt.savefig(
            f"EDA_Results/Outlier_{col}.png"
        )

        plt.show()

    print("\nStep 15 completed successfully!")
    print("Outlier plots saved inside EDA_Results folder.")
outlier_visualization(df)    
# ============================================================
# STEP 16: CATEGORICAL VISUALIZATION
# ============================================================

def categorical_visualization(df):

    print("\n========== STEP 16: CATEGORICAL VISUALIZATION ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Include binary numerical columns such as Outcome
    binary_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if df[col].nunique() == 2
    ]

    categorical_cols = list(set(categorical_cols + binary_cols))

    if len(categorical_cols) == 0:
        print("No suitable categorical columns found.")
        return

    print("Categorical columns:", categorical_cols)

    # Generate count plots
    for col in categorical_cols:

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=col
        )

        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            f"EDA_Results/Categorical_{col}.png"
        )

        plt.show()

    print("\nStep 16 completed successfully!")
    print("Categorical plots saved in EDA_Results.")
categorical_visualization(df)
# ============================================================
# STEP 17: CORRELATION HEATMAP ANALYSIS
# ============================================================

def correlation_heatmap(df):

    print("\n========== STEP 17: CORRELATION ANALYSIS ==========")

    # Select numerical columns
    numerical_df = df.select_dtypes(
        include=np.number
    )

    if numerical_df.shape[1] < 2:
        print("Not enough numerical columns.")
        return

    # Calculate correlation
    correlation_matrix = numerical_df.corr()

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(2))

    # Create output folder
    os.makedirs("EDA_Results", exist_ok=True)

    # Generate heatmap
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        square=True
    )

    plt.title("Correlation Heatmap", fontsize=16)

    plt.tight_layout()

    plt.savefig(
        "EDA_Results/Correlation_Heatmap.png"
    )

    plt.show()

    # Correlation with target variable
    if "Outcome" in correlation_matrix.columns:

        print("\nCorrelation with Outcome:")

        print(
            correlation_matrix["Outcome"]
            .sort_values(ascending=False)
        )

    print("\nStep 17 completed successfully!")
    print("Correlation heatmap saved in EDA_Results.")
correlation_heatmap(df)    
# ============================================================
# STEP 18: TARGET VARIABLE ANALYSIS
# ============================================================

def target_variable_analysis(df):

    print("\n========== STEP 18: TARGET VARIABLE ANALYSIS ==========")

    target = "Outcome"

    if target not in df.columns:
        print("Target column not found!")
        return

    os.makedirs("EDA_Results", exist_ok=True)

    # Count target values
    target_counts = df[target].value_counts().sort_index()

    print("\nTarget Distribution:")
    print(target_counts)

    # Calculate percentages
    percentages = df[target].value_counts(
        normalize=True
    ).sort_index() * 100

    print("\nTarget Percentages:")
    print(percentages.round(2))

    # Visualization
    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=target_counts.index,
        y=target_counts.values
    )

    plt.title("Diabetes Outcome Distribution")
    plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    plt.ylabel("Number of Patients")

    plt.xticks([0, 1], ["No Diabetes", "Diabetes"])

    plt.tight_layout()

    plt.savefig("EDA_Results/Target_Analysis.png")

    plt.show()

    # Check class imbalance
    difference = percentages.max() - percentages.min()

    print("\nClass Distribution Difference:",
          round(difference, 2), "%")

    print("\nStep 18 completed successfully!")
target_variable_analysis(df)
# ============================================================
# STEP 19: FEATURE COMPARISON BASED ON OUTCOME
# ============================================================

def feature_comparison(df):

    print("\n========== STEP 19: FEATURE COMPARISON ==========")

    if "Outcome" not in df.columns:
        print("Outcome column not found!")
        return

    os.makedirs("EDA_Results", exist_ok=True)

    # Select numerical features except target
    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    features = [
        col for col in numerical_cols
        if col != "Outcome" and df[col].nunique() > 1
    ]

    if len(features) == 0:
        print("No suitable numerical features found!")
        return

    # Group statistics
    print("\nFeature Comparison by Outcome:")

    comparison = df.groupby("Outcome")[features].mean()

    print(comparison.round(2))

    # Generate box plots
    for col in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Outcome",
            y=col
        )

        plt.title(f"{col} Comparison by Diabetes Outcome")

        plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
        plt.ylabel(col)

        plt.tight_layout()

        plt.savefig(
            f"EDA_Results/Comparison_{col}.png"
        )

        plt.show()

    print("\nStep 19 completed successfully!")
    print("Feature comparison plots saved.")
feature_comparison(df)
# ============================================================
# STEP 20: ADVANCED STATISTICAL ANALYSIS
# ============================================================

def advanced_statistical_analysis(df):

    print("\n========== STEP 20: STATISTICAL ANALYSIS ==========")

    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.empty:
        print("No numerical columns found!")
        return

    statistics = pd.DataFrame({
        "Mean": numerical_df.mean(),
        "Median": numerical_df.median(),
        "Mode": numerical_df.mode().iloc[0],
        "Standard Deviation": numerical_df.std(),
        "Variance": numerical_df.var(),
        "Minimum": numerical_df.min(),
        "Maximum": numerical_df.max(),
        "Skewness": numerical_df.skew(),
        "Kurtosis": numerical_df.kurtosis()
    })

    print("\nAdvanced Statistical Summary:")
    print(statistics.round(2))

    os.makedirs("EDA_Results", exist_ok=True)

    statistics.to_csv(
        "EDA_Results/Advanced_Statistics.csv"
    )

    print("\nStep 20 completed successfully!")
# ============================================================
# STEP 21: FEATURE IMPORTANCE ANALYSIS
# ============================================================

def feature_importance_analysis(df):

    print("\n========== STEP 21: FEATURE IMPORTANCE ==========")

    target = "Outcome"

    if target not in df.columns:
        print("Target column not found!")
        return

    numerical_df = df.select_dtypes(include=np.number)

    correlations = numerical_df.corr()[target].drop(
        labels=[target]
    )

    importance = pd.DataFrame({
        "Feature": correlations.index,
        "Correlation": correlations.values,
        "Absolute Correlation": correlations.abs().values
    })

    importance = importance.sort_values(
        by="Absolute Correlation",
        ascending=False
    )

    print("\nFeature Ranking:")
    print(importance.round(3))

    os.makedirs("EDA_Results", exist_ok=True)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance,
        x="Absolute Correlation",
        y="Feature"
    )

    plt.title("Feature Relationship with Diabetes Outcome")

    plt.tight_layout()

    plt.savefig(
        "EDA_Results/Feature_Importance.png"
    )

    plt.show()

    importance.to_csv(
        "EDA_Results/Feature_Importance.csv",
        index=False
    )

    print("\nStep 21 completed successfully!")
# ============================================================
# STEP 22: DATA QUALITY ANALYSIS
# ============================================================

def data_quality_analysis(df):

    print("\n========== STEP 22: DATA QUALITY ANALYSIS ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    # Missing values
    missing = df.isnull().sum()

    # Duplicate records
    duplicates = df.duplicated().sum()

    print("\nMissing Values:")
    print(missing)

    print("\nDuplicate Records:", duplicates)

    # Potential invalid zero values
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    available_columns = [
        col for col in zero_columns if col in df.columns
    ]

    zero_counts = {}

    for col in available_columns:

        zero_counts[col] = (
            df[col].eq(0).sum()
        )

    print("\nPotential Invalid Zero Values:")
    print(pd.Series(zero_counts))

    # Create quality report
    quality_report = pd.DataFrame({
        "Missing Values": missing,
        "Data Type": df.dtypes.astype(str),
        "Unique Values": df.nunique()
    })

    quality_report.to_csv(
        "EDA_Results/Data_Quality_Report.csv"
    )

    print("\nStep 22 completed successfully!")
    print("Data quality report saved.")
# ============================================================
# STEP 23: AUTOMATED INSIGHTS GENERATION
# ============================================================

def automated_insights(df):

    print("\n========== STEP 23: AUTOMATED INSIGHTS ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    insights = []

    # 1. Dataset information
    rows, columns = df.shape

    insights.append(
        f"Dataset contains {rows} records and {columns} columns."
    )

    # 2. Target distribution
    if "Outcome" in df.columns:

        counts = df["Outcome"].value_counts()

        if 0 in counts.index and 1 in counts.index:

            diabetic_percentage = (
                counts[1] / counts.sum()
            ) * 100

            insights.append(
                f"{diabetic_percentage:.2f}% of records have Outcome = 1."
            )

    # 3. Highest mean feature
    numerical_df = df.select_dtypes(include=np.number)

    if not numerical_df.empty:

        feature_means = numerical_df.drop(
            columns=["Outcome"],
            errors="ignore"
        ).mean()

        if not feature_means.empty:

            highest_mean = feature_means.idxmax()

            insights.append(
                f"{highest_mean} has the highest numerical mean."
            )

    # 4. Strongest correlation
    if "Outcome" in numerical_df.columns:

        correlations = numerical_df.corr()["Outcome"].drop(
            labels=["Outcome"]
        ).dropna()

        if not correlations.empty:

            strongest = correlations.abs().idxmax()

            value = correlations[strongest]

            insights.append(
                f"{strongest} has the strongest absolute "
                f"correlation with Outcome ({value:.3f})."
            )

    # 5. Potential outlier detection using IQR
    outlier_features = []

    for col in numerical_df.columns:

        if col == "Outcome":
            continue

        q1 = numerical_df[col].quantile(0.25)
        q3 = numerical_df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = (
            (numerical_df[col] < lower) |
            (numerical_df[col] > upper)
        ).sum()

        if outliers > 0:

            outlier_features.append(
                f"{col}: {outliers} potential outliers"
            )

    insights.append(
        "Potential outliers: " +
        (", ".join(outlier_features)
         if outlier_features else "None detected.")
    )

    # Display insights
    print("\nAUTOMATED DATASET INSIGHTS\n")

    for number, insight in enumerate(insights, start=1):

        print(f"{number}. {insight}")

    # Save insights
    with open(
        "EDA_Results/Automated_Insights.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for number, insight in enumerate(insights, start=1):

            file.write(f"{number}. {insight}\n")

    print("\nStep 23 completed successfully!")
    print("Automated insights saved.")
# STEP 20
advanced_statistical_analysis(df)

# STEP 21
feature_importance_analysis(df)

# STEP 22
data_quality_analysis(df)

# STEP 23
automated_insights(df)
# ============================================================
# STEP 24: COMPREHENSIVE EDA REPORT
# ============================================================

def generate_eda_report(df):

    print("\n========== STEP 24: EDA REPORT ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    report_path = "EDA_Results/Complete_EDA_Report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("       COMPREHENSIVE EXPLORATORY DATA ANALYSIS\n")
        file.write("=" * 60 + "\n\n")

        # Dataset information
        file.write("1. DATASET INFORMATION\n")
        file.write("-" * 40 + "\n")

        file.write(f"Total Rows: {df.shape[0]}\n")
        file.write(f"Total Columns: {df.shape[1]}\n")
        file.write(f"Total Missing Values: {df.isnull().sum().sum()}\n")
        file.write(f"Duplicate Records: {df.duplicated().sum()}\n\n")

        # Data types
        file.write("2. DATA TYPES\n")
        file.write("-" * 40 + "\n")
        file.write(df.dtypes.to_string())
        file.write("\n\n")

        # Missing values
        file.write("3. MISSING VALUE ANALYSIS\n")
        file.write("-" * 40 + "\n")
        file.write(df.isnull().sum().to_string())
        file.write("\n\n")

        # Descriptive statistics
        file.write("4. DESCRIPTIVE STATISTICS\n")
        file.write("-" * 40 + "\n")
        file.write(df.describe().round(2).to_string())
        file.write("\n\n")

        # Correlation
        numerical_df = df.select_dtypes(include=np.number)

        file.write("5. CORRELATION ANALYSIS\n")
        file.write("-" * 40 + "\n")

        file.write(
            numerical_df.corr().round(2).to_string()
        )

        file.write("\n\n")

        # Target analysis
        if "Outcome" in df.columns:

            file.write("6. TARGET DISTRIBUTION\n")
            file.write("-" * 40 + "\n")

            file.write(
                df["Outcome"].value_counts().to_string()
            )

            file.write("\n\n")

        file.write("=" * 60 + "\n")
        file.write("END OF EDA REPORT\n")

    print("EDA report generated successfully!")
    print("Saved:", report_path)
# ============================================================
# STEP 25: EXPORT RESULTS TO EXCEL AND PDF
# ============================================================

from matplotlib.backends.backend_pdf import PdfPages

def export_eda_results(df):

    print("\n========== STEP 25: EXPORT RESULTS ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    numerical_df = df.select_dtypes(include=np.number)

    # ------------------------------------------------
    # 1. EXPORT TO EXCEL
    # ------------------------------------------------

    excel_path = "EDA_Results/EDA_Complete_Report.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Dataset overview
        overview = pd.DataFrame({
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Missing Values",
                "Duplicate Records"
            ],
            "Value": [
                df.shape[0],
                df.shape[1],
                df.isnull().sum().sum(),
                df.duplicated().sum()
            ]
        })

        overview.to_excel(
            writer,
            sheet_name="Overview",
            index=False
        )

        # Descriptive statistics
        df.describe().to_excel(
            writer,
            sheet_name="Statistics"
        )

        # Missing values
        df.isnull().sum().to_frame(
            "Missing Values"
        ).to_excel(
            writer,
            sheet_name="Missing Values"
        )

        # Correlation
        numerical_df.corr().to_excel(
            writer,
            sheet_name="Correlation"
        )

        # Target distribution
        if "Outcome" in df.columns:

            df["Outcome"].value_counts().to_frame(
                "Count"
            ).to_excel(
                writer,
                sheet_name="Target Distribution"
            )

    print("Excel report generated!")

    # ------------------------------------------------
    # 2. EXPORT TO PDF
    # ------------------------------------------------

    pdf_path = "EDA_Results/EDA_Complete_Report.pdf"

    with PdfPages(pdf_path) as pdf:

        # Page 1: Dataset Overview
        fig, ax = plt.subplots(figsize=(11, 8))

        ax.axis("off")

        overview_text = f"""
        COMPREHENSIVE EDA REPORT

        Dataset: Diabetes Dataset

        Total Records: {df.shape[0]}

        Total Features: {df.shape[1]}

        Missing Values: {df.isnull().sum().sum()}

        Duplicate Records: {df.duplicated().sum()}

        Numerical Features: {len(numerical_df.columns)}

        Analysis Status: Completed
        """

        ax.text(
            0.1, 0.85,
            overview_text,
            fontsize=15,
            verticalalignment="top"
        )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Statistical Summary
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.axis("off")

        stats = df.describe().round(2)

        table = ax.table(
            cellText=stats.values,
            rowLabels=stats.index,
            colLabels=stats.columns,
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        ax.set_title(
            "Descriptive Statistical Analysis",
            fontsize=15
        )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Correlation Heatmap
        if numerical_df.shape[1] >= 2:

            fig, ax = plt.subplots(figsize=(10, 8))

            sns.heatmap(
                numerical_df.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax
            )

            ax.set_title("Correlation Analysis")

            pdf.savefig(fig)
            plt.close(fig)

    print("PDF report generated!")

    print("\nStep 25 completed successfully!")

    print("Excel:", excel_path)
    print("PDF:", pdf_path)
# ============================================================
# STEP 26: VISUALIZATION SUMMARY
# ============================================================

def visualization_summary(df):

    print("\n========== STEP 26: VISUALIZATION SUMMARY ==========")

    os.makedirs("EDA_Results", exist_ok=True)

    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 18)
    )

    fig.suptitle(
        "DIABETES DATASET - EDA VISUALIZATION SUMMARY",
        fontsize=18
    )

    # 1. Outcome Distribution
    if "Outcome" in df.columns:

        sns.countplot(
            data=df,
            x="Outcome",
            ax=axes[0, 0]
        )

        axes[0, 0].set_title("Outcome Distribution")

    # 2. Glucose Distribution
    if "Glucose" in df.columns:

        sns.histplot(
            data=df,
            x="Glucose",
            kde=True,
            ax=axes[0, 1]
        )

        axes[0, 1].set_title("Glucose Distribution")

    # 3. BMI Distribution
    if "BMI" in df.columns:

        sns.histplot(
            data=df,
            x="BMI",
            kde=True,
            ax=axes[1, 0]
        )

        axes[1, 0].set_title("BMI Distribution")

    # 4. Age Distribution
    if "Age" in df.columns:

        sns.histplot(
            data=df,
            x="Age",
            kde=True,
            ax=axes[1, 1]
        )

        axes[1, 1].set_title("Age Distribution")

    # 5. Correlation Heatmap
    numerical_df = df.select_dtypes(include=np.number)

    if numerical_df.shape[1] >= 2:

        sns.heatmap(
            numerical_df.corr(),
            cmap="coolwarm",
            annot=True,
            fmt=".1f",
            ax=axes[2, 0]
        )

        axes[2, 0].set_title("Correlation Heatmap")

    # 6. Glucose vs Outcome
    if "Glucose" in df.columns and "Outcome" in df.columns:

        sns.boxplot(
            data=df,
            x="Outcome",
            y="Glucose",
            ax=axes[2, 1]
        )

        axes[2, 1].set_title("Glucose vs Outcome")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(
        "EDA_Results/Visualization_Summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nStep 26 completed successfully!")

    print(
        "Visualization saved: "
        "EDA_Results/Visualization_Summary.png"
    )
# STEP 24
generate_eda_report(df)

# STEP 25
export_eda_results(df)

# STEP 26
visualization_summary(df)