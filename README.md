# DATA 605 Zespri 1 Project
**Authored:** Jeremy Chen

## Overview
This project was completed as part of a data science internship with Zespri. It explores data extraction, exploratory analysis, quadrat sampling strategies, and predictive modeling based on spatial fruit location data.

## Raw data
- `fruit_location_data.zip`: Contains the raw spatial data.

## Project workflow
1. **Raw Data Extraction**
   - `00_raw_data_extraction.py`: Extracts and samples raw data into Parquet format for local storage.

2. **Data Overview**
   - `01_parquet_data_extraction.py`: Provides a high-level summary of the preprocessed dataset.

3. **Exploratory Data Analysis**
   - `01_eda_analysis.py`: Conducts exploratory analysis on rows.
   - `01_xyz_coordinate_analysis.py`: Analyzes spatial distribution of data points.

4. **Quadrat Sampling Experiments**
   - `02_circle_quadrat_sampling.py`
   - `02_rectangle_quadrat_sampling.py`
   - `02_square_quadrat_sampling.py`: Perform quadrat sampling with different shapes, then evaluate bias, variance, and accuracy.

5. **Sampling Examples**
   - `02_quadrat_sampling_plot_example.py`: Visualizes a sample random quadrat placement.

6. **Row Centre Sampling**
   - `03_row_centre_only_sampling.py`: Assesses bias from center-only sampling.
   - `03_row_centre_only_example.py`: Visual example of row center definition.

7. **Data Processing and Feature Engineering**
   - `04_data_processing.py`: Prepares data and creates feature sets.
   - `04_outlier_removal.py`: Cleans data by removing outliers.

8. **Machine Learning Models**
   - `04_LR_model.py`: Linear Regression model with feature selection and evaluation.
   - `04_RF_model.py`: Random Forest model with hyperparameter tuning and feature importance plots.


## Note 
- Each step beyond Step 1 relies on the Parquet files generated from the initial extraction step.
- Scripts require configuration of input/output directories based on your local file structure.

---

