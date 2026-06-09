import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Import scikit-learn pipeline elements
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Parser for command line inputs
parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
parser.add_argument("--input", type=str, required=True, help="Input CSV file")
parser.add_argument("--output-csv", type=str, default="cleaned_data.csv", help="Output CSV path")
parser.add_argument("--output-pdf", type=str, default="analysis_report.pdf", help="Output PDF report path")
parser.add_argument("--scaler", type=str, choices=["minmax", "standard"], default="minmax", help="Scaling type")
args = parser.parse_args()

# Custom transformer to cap outliers using IQR method
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[col], upper=self.upper_bounds_[col])
        return X_df.values

# 1. Load dataset
df = pd.read_csv(args.input)
initial_shape = df.shape
print(f"Loaded dataset: {initial_shape}")

# 2. Drop duplicates
df = df.drop_duplicates()
duplicates_removed = initial_shape[0] - len(df)
print(f"Removed {duplicates_removed} duplicate rows")

# Separate columns by data type
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Log stats for PDF summary report
imputed_list = []
outlier_list = []

for col in df.columns:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        imputed_list.append(f"{col}: filled {nulls} missing values")

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_list.append(f"{col}: capped {outliers} outliers")

# 3. Define the Scikit-learn Pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('capper', OutlierCapper()),
    ('scaler', MinMaxScaler() if args.scaler == "minmax" else StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# 4. Run preprocessing pipeline
processed_array = preprocessor.fit_transform(df)

# Rebuild DataFrame from the pipeline output
processed_cols = numeric_cols + categorical_cols
processed_df = pd.DataFrame(processed_array, columns=processed_cols)

# Save the final cleaned CSV file
processed_df.to_csv(args.output_csv, index=False)
print(f"Cleaned dataset saved to {args.output_csv}")

# 5. Generate PDF report
with PdfPages(args.output_pdf) as pdf:
    # Page 1: Logging Text
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    report_text = (
        "Dataset Preprocessing Report\n"
        "============================\n\n"
        f"Original shape: {initial_shape}\n"
        f"Cleaned shape: {df.shape}\n"
        f"Duplicates removed: {duplicates_removed}\n\n"
        "Imputed values:\n"
    )
    if imputed_list:
        report_text += "\n".join([f"- {item}" for item in imputed_list])
    else:
        report_text += "- No missing values found.\n"
        
    report_text += "\n\nOutlier capping summary:\n"
    if outlier_list:
        report_text += "\n".join([f"- {item}" for item in outlier_list])
    else:
        report_text += "- No numeric columns to cap outliers.\n"
        
    plt.text(0.1, 0.9, report_text, transform=ax.transAxes, fontsize=11, family='monospace', va='top')
    pdf.savefig(fig)
    plt.close(fig)
    
    # Page 2: Summary Stats Table
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    plt.text(0.5, 0.92, "Summary Statistics Table", fontsize=14, fontweight='bold', ha='center')
    
    if numeric_cols:
        # Generate stats on original values (before scaling)
        stats_df = df[numeric_cols].describe().round(4).reset_index()
        stats_df.rename(columns={'index': 'metric'}, inplace=True)
        
        table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.4)
    else:
        plt.text(0.5, 0.5, "No numeric columns in this dataset", ha='center')
        
    pdf.savefig(fig)
    plt.close(fig)
    
    # Page 3: Distribution Plots (Max 4 columns)
    plot_cols = numeric_cols[:4]
    if plot_cols:
        fig, axes = plt.subplots(len(plot_cols), 2, figsize=(8.5, 11))
        if len(plot_cols) == 1:
            axes = np.array([axes])
            
        for i, col in enumerate(plot_cols):
            # Hist / KDE
            sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')
            axes[i, 0].set_title(f"{col} Distribution")
            
            # Boxplot
            sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')
            axes[i, 1].set_title(f"{col} Boxplot")
            
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    # Page 4: Correlation Matrix Heatmap (Max 10 columns)
    corr_cols = numeric_cols[:10]
    if len(corr_cols) > 1:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Generated PDF report: {args.output_pdf}")
