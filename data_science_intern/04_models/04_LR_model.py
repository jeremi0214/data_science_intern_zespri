import os
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------
# Helper Functions for Model Evaluation
# --------------------------------------

# Function to calculate estimation accuracy
def estimation_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculate the percentage of predictions within a certain threshold of the actual values.
    """
    errors = abs(y_true - y_pred)
    accuracy = (errors <= threshold * abs(y_true)).mean() * 100  # Percentage
    return accuracy

# Evaluation function with estimation accuracy
def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate regression model performance using various metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(root_mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    acc_10 = estimation_accuracy(y_true, y_pred, threshold=0.1)  # Within 10%
    acc_5 = estimation_accuracy(y_true, y_pred, threshold=0.05)  # Within 5%
    
    print(f"{model_name} Performance:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  Estimation Accuracy (±10%): {acc_10:.2f}%")
    print(f"  Estimation Accuracy (±5%): {acc_5:.2f}%\n")

# ------------------------------
# Load and Preprocess Data
# ------------------------------

# Load Dataset
data = pd.read_csv('data_processed.csv')

# Feature Engineering: Compute additional features
data["Row Centre Ratio"] = data["Row Centre Area"] / data["Row Area"]
data["Quadrat Size to Row Width"] = 1 / data["Row Width"]

# Drop non-numeric or identifier columns
data = data.drop(columns=['KPIN', 'Block', 'Row'])

# # Features and Target
# features = ['Row Width', 'Row Length', 'Row Centre Ratio', 'Row Centre Density', 
#             'Quadrat Size to Row Width', 'Average Quadrat Density', 'Mean of X Coordinates',
#             'Standard Deviation of X Coordinates', 'Average Distance to Row Centre']

# Define target variable
target = 'Actual Density'

# Split into features (X) and target (y)
X = data.drop(columns=target)
y = data[target]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------
# OLS Regression (Statsmodels)
# --------------------------------

# Adding a constant for the intercept (required by statsmodels)
X_train_sm = sm.add_constant(X_train)

# Fit the model using statsmodels for p-values
model = sm.OLS(y_train, X_train_sm).fit()

# Get the coefficients and p-values
summary_df = pd.DataFrame({
    'Feature': ['Intercept'] + X.columns.tolist(),
    'Coefficient': model.params.values,
    'p-Value': model.pvalues.values
})

# Add a column to indicate statistical significance based on a threshold (e.g., 0.05)
summary_df['Significant'] = summary_df['p-Value'] < 0.05

# Sort by absolute value of coefficients
summary_df = summary_df.reindex(summary_df['Coefficient'].abs().sort_values(ascending=False).index)


############### Standardisation ###############
############### Not needed ###############
# from sklearn.preprocessing import StandardScaler

# # Standardize the features
# scaler = StandardScaler()
# X_standardized = scaler.fit_transform(X)


# ------------------------------------
# (Optional) Feature Correlation Heatmap
# ------------------------------------

# # Compute correlation matrix
# correlation_matrix = X.corr()

# # Plot correlation heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Matrix")
# plt.show()

# ------------------------------
# Train Linear Regression Model
# ------------------------------

# Initialize and fit the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# Evaluate model performance
evaluate_model(y_test, y_pred_lr, "Linear Regression")


# ------------------------------------
# (Optional) Cross Validation
# ------------------------------------
# from sklearn.model_selection import cross_val_score, cross_val_predict

# # Cross-validation setup
# cv = 5  # Number of folds

# # Evaluate R² scores across folds
# r2_scores = cross_val_score(lr, X, y, cv=cv, scoring='r2')
# print(f"Cross-validated R² Scores: {r2_scores}")
# print(f"Mean R² Score: {r2_scores.mean():.3f}\n")

# # Evaluate MAE across folds
# mae_scores = cross_val_score(lr, X, y, cv=cv, scoring='neg_mean_absolute_error')
# print(f"Cross-validated MAE Scores (negative): {mae_scores}")
# print(f"Mean MAE: {-mae_scores.mean():.3f}\n")

# # Evaluate RMSE across folds
# rmse_scores = cross_val_score(lr, X, y, cv=cv, scoring='neg_mean_squared_error')
# print(f"Cross-validated RMSE Scores (negative): {rmse_scores}")
# print(f"Mean RMSE: {(-rmse_scores.mean()) ** 0.5:.3f}\n")

# # Generate cross-validated predictions
# y_pred_cv = cross_val_predict(lr, X, y, cv=cv)

# # Evaluate predictions from cross-validation
# mae_cv = mean_absolute_error(y, y_pred_cv)
# rmse_cv = root_mean_squared_error(y, y_pred_cv)
# r2_cv = r2_score(y, y_pred_cv)

# print("Evaluation Based on Cross-Validation Predictions:")
# print(f"  MAE: {mae_cv:.3f}")
# print(f"  RMSE: {rmse_cv:.3f}")
# print(f"  R² Score: {r2_cv:.3f}")


# ------------------------------------
# (Optional) Coefficient Bar Chart
# ------------------------------------

# # Determine x-axis limits
# x_min = summary_df['Coefficient'].min() * 1.7
# x_max = summary_df['Coefficient'].max() * 1.4

# # Set style for a professional look
# sns.set(style="whitegrid", palette="pastel")

# # Determine color palette: green for significant, gray for insignificant
# colors = ['steelblue' if sig else 'lightgray' for sig in summary_df['Significant']]

# # Plot the bar chart
# plt.figure(figsize=(16, 10))
# sns.barplot(
#     x='Coefficient', y='Feature', 
#     data=summary_df, 
#     palette=colors, 
#     orient='h',
#     hue='Feature',
#     legend=False
# )

# # Highlight the first three bars and their labels
# highlight_color = 'red'
# for i in range(3):
#     plt.gca().patches[i].set_edgecolor(highlight_color)
#     plt.gca().patches[i].set_linewidth(2.5)

# # Add coefficient values and p-values as text annotations
# for i, (coef, pval, sig) in enumerate(zip(summary_df['Coefficient'], summary_df['p-Value'], summary_df['Significant'])):
#     color = 'black' if sig else 'gray'
#     if i < 3:  # Highlight for the first three features
#         color = highlight_color
#         fontsize = 18
#         fontweight = "bold"
#     else:
#         fontsize = 12
#         fontweight = "normal"
#     offset = 0.5
#     plt.text(coef + (offset if coef > 0 else -offset), i, f'{coef:.2f}, p={pval:.3f}', 
#              ha='left' if coef > 0 else 'right', 
#              va='center', fontsize=fontsize, 
#              fontweight=fontweight, color=color)

# # Add vertical line at 0 for reference
# plt.axvline(0, color='black', linewidth=1, linestyle='--')

# # Highlight the y-axis tick labels for the first three features
# ax = plt.gca()
# yticks = ax.get_yticklabels()
# for i in range(3):
#     yticks[i].set_color(highlight_color)  
#     yticks[i].set_fontweight("bold")
#     yticks[i].set_fontsize(18)


# legend_labels = [Patch(color='steelblue', label='Significant (p < 0.05)'), 
#                  Patch(color='lightgray', label='Not Significant (p ≥ 0.05)')]
# plt.legend(handles=legend_labels, title="Feature Significance", loc='lower right')

# # Set x-axis limits
# plt.xlim(x_min, x_max)

# # Add a title and labels
# plt.title('Linear Regression Coefficients and Significance', fontsize=24, fontweight='bold')
# plt.xlabel('Coefficient Value', fontsize=18)
# plt.ylabel('Features', fontsize=18)

# # Show the plot
# plt.tight_layout()
# plt.show()


# -----------------------------------
# (Optional) Residual Analysis Plot
# -----------------------------------

# # Compute residuals (differences between actual and predicted values)
# residuals = y_test - y_pred_lr

# # Plot residuals vs. predicted values
# plt.scatter(y_pred_lr, residuals)
# plt.axhline(0, color='red', linestyle='--')
# plt.title("Residuals vs. Predicted Values")
# plt.xlabel("Predicted Values")
# plt.ylabel("Residuals")
# plt.show()

