# 📊 Automated EDA Dashboard

An interactive Exploratory Data Analysis (EDA) tool built using Python and Streamlit. This project automates the initial stages of data analysis and helps users understand datasets through statistical summaries, data quality checks, and visualizations.

## 🚀 Project Overview

Exploratory Data Analysis is an important part of the Data Science workflow. It helps identify patterns, relationships, missing values, and trends within datasets.

This project provides an automated approach to EDA, allowing users to upload datasets and explore their characteristics through an interactive dashboard.

## ✨ Features

- Dataset Upload (CSV and Excel)
- Dataset Overview and Preview
- Data Type Analysis
- Missing Value Detection
- Duplicate Record Detection
- Descriptive Statistical Analysis
- Numerical Data Visualization
- Correlation Heatmap
- Interactive Dashboard using Streamlit
- Dataset Download Functionality

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Plotly | Interactive Charts |
| Streamlit | Dashboard Development |

## 📂 Project Structure

```text
Automated-EDA-Dashboard/
│
├── datasets/
│   └── diabetes.csv
│
├── visualizations/
│   └── EDA charts (.png)
│
├── reports/
│
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Automated-EDA-Dashboard.git
```

### 2. Navigate to the project directory

```bash
cd Automated-EDA-Dashboard
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
python -m pip install pandas numpy matplotlib seaborn plotly streamlit openpyxl
```

### 4. Run the dashboard

```bash
python -m streamlit run dashboard.py
```

The dashboard will open in your web browser.

## 📊 Dataset

This project currently uses the Diabetes Dataset (`diabetes.csv`) for exploratory data analysis.

The dashboard also supports uploading other CSV and Excel datasets.

## 🎯 Project Objectives

- Automate repetitive EDA tasks.
- Understand dataset structure and quality.
- Identify missing values and duplicate records.
- Generate meaningful statistical summaries.
- Visualize numerical distributions and feature correlations.
- Provide an easy-to-use interactive analysis interface.

## 📸 Project Screenshots

Project visualizations and dashboard screenshots are available in the `visualizations/` folder.

## 🔮 Future Improvements

- Advanced interactive visualizations.
- Automated insight generation.
- PDF report generation.
- Additional statistical analysis.

## 👨‍💻 Author

**Yash**

Computer Science Engineering Student  
Interested in Data Science, Machine Learning, and Data Analytics.

## ⭐ Acknowledgements

Built as a learning project to explore Python-based data analysis, visualization, and dashboard development.

If you find this project useful, consider giving the repository a star!
