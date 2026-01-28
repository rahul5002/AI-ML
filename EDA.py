import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.ioff()

try:
    file_name = r"C:\Users\dell\OneDrive\Desktop\Student Data.xlsx"
    df_full = pd.read_excel(file_name) 
    
    if df_full.shape[1] < 5:
        df = df_full.copy()
        print(f"Warning: The file only has {df_full.shape[1]} columns. Analyzing all columns.")
    else:
        df = df_full.iloc[:, :5].copy() # Select the first 5 columns

    print("--- Analysis on Selected Columns: ", df.columns.tolist(), " ---")
    print("\n" + "="*50)


    # 1. Load the dataset and show the first 10 rows.
    print("1. First 10 Rows:")
    print(df.head(10))
    print("\n" + "="*50)

    # 2. Display the number of rows and columns.
    rows, cols = df.shape
    print(f"2. Number of Rows: {rows}, Number of Columns: {cols}")
    print("\n" + "="*50)

    # 3. Check for missing values in each column.
    print("3. Missing Values per Column:")
    print(df.isnull().sum())
    print("\n" + "="*50)

    # 4. Show the summary statistics (mean, median, mode, standard deviation).
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    print("4. Summary Statistics:")
    if not numeric_cols.empty:
        print("\n   A. Numerical Columns (Mean, Median (50%), St. Dev (std)):")
        print(df[numeric_cols].describe().loc[['mean', '50%', 'std']].T)
        print("\n   B. Numerical Columns Mode:")
        print(df[numeric_cols].mode().iloc[0])
    if not categorical_cols.empty:
        print("\n   C. Categorical/Other Columns (Count, Top Value, Frequency):")
        print(df[categorical_cols].describe().loc[['count', 'top', 'freq']].T)
    print("\n" + "="*50)

    # 5. Find the column with the maximum and minimum values.
    if not numeric_cols.empty:
        numeric_df = df[numeric_cols]
        max_val = numeric_df.max().max()
        max_col = numeric_df.max().idxmax()
        min_val = numeric_df.min().min()
        min_col = numeric_df.min().idxmin()

        print(f"5. Max Value Column: {max_col} ({max_val:.2f})")
        print(f"   Min Value Column: {min_col} ({min_val:.2f})")
    print("\n" + "="*50)

    # 6. Count how many unique values each categorical column has.
    if not categorical_cols.empty:
        print("6. Unique Value Count for Categorical Columns:")
        print(df[categorical_cols].nunique().sort_values(ascending=False))
    print("\n" + "="*50)

    # --- PLOTTING SECTION (Tasks 7-10) ---
    print("--- Plotting Results (Tasks 7-10) ---")

    numeric_cols_list = numeric_cols.tolist()
    categorical_cols_list = categorical_cols.tolist()
    
    # 7. Histogram (uses first numeric column)
    if len(numeric_cols_list) >= 1:
        col_hist = numeric_cols_list[0]
        plt.figure(figsize=(7, 4))
        df[col_hist] = pd.to_numeric(df[col_hist], errors='coerce')
        df[col_hist].dropna().hist(bins=15, edgecolor='black', color='teal', alpha=0.8)
        plt.title(f'7. Histogram of {col_hist}')
        plt.xlabel(col_hist)
        plt.ylabel('Frequency')
        plt.savefig('histogram.png')
        print(f"7. Histogram plotted for column: {col_hist}")
    

    # 8. Bar Plot (uses first categorical column)
    if len(categorical_cols_list) >= 1:
        col_bar = categorical_cols_list[0]
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col_bar, data=df, palette='viridis',
                      order=df[col_bar].value_counts().index)
        plt.title(f'8. Frequency of {col_bar} Categories')
        plt.xlabel(col_bar)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('barplot.png')
        print(f"8. Bar plot plotted for column: {col_bar}")
    

    # 9. Scatter Plot (uses first two numeric columns)
    if df_full.shape[1] >= 7:
        col_3_name = df_full.columns[2] # 3rd column
        col_7_name = df_full.columns[6] # 7th column

        # Attempt to convert to numeric and handle missing values
        x_data = pd.to_numeric(df_full[col_3_name], errors='coerce')
        y_data = pd.to_numeric(df_full[col_7_name], errors='coerce')
        
        scatter_data = pd.DataFrame({col_3_name: x_data, col_7_name: y_data}).dropna()
        
        if not scatter_data.empty:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=col_3_name, y=col_7_name, data=scatter_data, alpha=0.7)
            plt.title(f'9. Scatter Plot: {col_3_name} vs. {col_7_name}')
            plt.xlabel(col_3_name)
            plt.ylabel(col_7_name)
            plt.savefig('scatterplot.png')
            print(f"9. Scatter plot plotted for columns: {col_3_name} (3rd) and {col_7_name} (7th)")
        else:
            print(f"9. Scatter plot skipped: Data for {col_3_name} and {col_7_name} is not numerical or is empty after cleaning.")

    else:
        print("9. Scatter plot skipped: Dataset has fewer than 7 columns to use for the plot.")

    

    # 10. Boxplot (uses first numeric column)
    if len(numeric_cols_list) >= 1:
        col_box = numeric_cols_list[0]
        plt.figure(figsize=(5, 6))
        df[col_box] = pd.to_numeric(df[col_box], errors='coerce')
        sns.boxplot(y=df[col_box].dropna(), color='lightcoral')
        plt.title(f'10. Boxplot for {col_box} (Outlier Detection)')
        plt.ylabel(col_box)
        plt.tight_layout()
        plt.savefig('boxplot.png')
        print(f"10. Boxplot plotted for column: {col_box}")

    plt.close('all')

except FileNotFoundError:
    print("\n--- ERROR ---")
    print(f"The file '{file_name}' was not found. Please re-upload the file or ensure it is named exactly '{file_name}'.")
except Exception as e:
    print(f"\nAn unexpected error occurred during execution: {type(e).__name__}: {e}")