import pandas as pd
import numpy as np

# Load the dataset from a CSV file
file_path = "data_processed.csv"  
df = pd.read_csv(file_path)

# Inspect the data to understand its distribution
print(df['Actual Density'].describe())

# Removing Outliers Using IQR Method
Q1 = df['Actual Density'].quantile(0.25)  # First quartile
Q3 = df['Actual Density'].quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataframe to exclude outliers
df_no_outliers_iqr = df[(df['Actual Density'] >= lower_bound) & (df['Actual Density'] <= upper_bound)]

# Save the cleaned dataset if needed
df_no_outliers_iqr.to_csv("data_processed_no_outlier.csv", index=False)

print("Original dataset size:", df.shape[0])
print("Dataset size after IQR filtering:", df_no_outliers_iqr.shape[0])