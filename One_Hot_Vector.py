import pandas as pd
import numpy as np

try:
    file_name = r"C:\Users\dell\OneDrive\Desktop\student_dream.xlsx"
    df = pd.read_excel(file_name)

    print("--- Original Data ---")
    print(df)
    print("\n")

    columns_to_encode = df.select_dtypes(include=['object']).columns

    if columns_to_encode.empty:
        print("No text-based columns found to encode.")
    else:
        print(f"Automatically identified columns for encoding: {list(columns_to_encode)}")

        one_hot_encoded_df = pd.get_dummies(df, columns=columns_to_encode, dtype=int)

        for original_col in columns_to_encode:
            one_hot_cols = [col for col in one_hot_encoded_df.columns if col.startswith(original_col + '_')]
            if one_hot_cols:
                new_col_name = f'{original_col}_vector'
                one_hot_encoded_df[new_col_name] = one_hot_encoded_df[one_hot_cols].apply(lambda row: list(row), axis=1).astype(str)
        
        original_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        vector_cols = [col for col in one_hot_encoded_df.columns if col.endswith('_vector')]
        
        final_df = one_hot_encoded_df[original_numeric_cols + vector_cols]

        print("\n--- Final DataFrame to be Saved (Preview) ---")
        print(final_df.head())
        print("\n")

        output_file_name = "one_hot_encoded_auto_output.xlsx"
        final_df.to_excel(output_file_name, index=False)

        print(f"Successfully created the one-hot encoded file: '{output_file_name}'")

except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {file_name}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")