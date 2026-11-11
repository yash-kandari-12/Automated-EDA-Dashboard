import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="EDA Automation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("📊 Automated EDA Dashboard")

st.markdown("""
### Exploratory Data Analysis & Visualization Platform

Upload your dataset and automatically explore
its structure, statistics, missing values and patterns.
""")

st.divider()

# ------------------------------------------------
# DATASET UPLOAD
# ------------------------------------------------

st.sidebar.header("Dataset Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset loaded successfully!")

    # ------------------------------------------------
    # DATASET OVERVIEW
    # ------------------------------------------------

    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.divider()

    # ------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------

    st.header("2. Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    # ------------------------------------------------
    # DATA TYPES
    # ------------------------------------------------

    st.header("3. Column Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(info_df, use_container_width=True)

    # ------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # ------------------------------------------------

    st.header("4. Descriptive Statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------

    st.header("5. Missing Value Analysis")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) > 0:

        missing_df = missing.reset_index()

        missing_df.columns = [
            "Column",
            "Missing Values"
        ]

        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.success("No missing values found!")

    st.divider()

    # ------------------------------------------------
    # NUMERICAL ANALYSIS
    # ------------------------------------------------

    st.header("6. Numerical Analysis")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numerical_columns:

        selected_column = st.selectbox(
            "Select Numerical Column",
            numerical_columns
        )

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            marginal="box"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.warning("No numerical columns available.")

    st.divider()

    # ------------------------------------------------
    # CORRELATION ANALYSIS
    # ------------------------------------------------

    st.header("7. Correlation Analysis")

    if len(numerical_columns) >= 2:

        correlation = df[numerical_columns].corr()

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    else:

        st.warning(
            "At least two numerical columns are required."
        )

    st.divider()

    # ------------------------------------------------
    # DOWNLOAD DATASET
    # ------------------------------------------------

    st.header("8. Download Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name="analyzed_dataset.csv",
        mime="text/csv"
    )

else:

    st.info("Please upload a CSV or Excel dataset to begin.")

    st.markdown("""
    ### Dashboard Features

    - Dataset Overview
    - Dataset Preview
    - Column Information
    - Descriptive Statistics
    - Missing Value Analysis
    - Numerical Analysis
    - Correlation Heatmap
    - Dataset Download
    """)