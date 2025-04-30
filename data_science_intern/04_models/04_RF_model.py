import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm


# Function to calculate estimation accuracy
def estimation_accuracy(y_true, y_pred, threshold=0.1):
    """
    Calculate the percentage of predictions within a certain threshold of the actual values.
    
    Parameters:
        y_true: array-like, true values.
        y_pred: array-like, predicted values.
        threshold: float, acceptable deviation as a proportion of actual value (e.g., 0.1 for 10%).

    Returns:
        accuracy: float, percentage of predictions within the threshold.
    """
    errors = abs(y_true - y_pred)
    accuracy = (errors <= threshold * abs(y_true)).mean() * 100  # Percentage
    return accuracy

# Evaluation function with estimation accuracy
def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate regression model performance with common metrics.
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

# Load Data
data = pd.read_csv('data_processed.csv')

data["Row Centre Ratio"] = data["Row Centre Area"] / data["Row Area"]
data["Quadrat Size to Row Width"] = 3 / data["Row Width"]

data = data.drop(columns=['Block', 'Row', 'Row Area', 'Row Centre Area', 'KPIN'])
# Features and Target
# features = ['Row Width', 'Row Length', 'Row Centre Ratio', 'Row Centre Density', 
#             'Quadrat Size to Row Width', 'Average Quadrat Density', 'Mean of X Coordinates',
#             'Standard Deviation of X Coordinates', 'Average Distance to Row Centre']

target = 'Actual Density'

# Train-Test Split
X = data.drop(columns=target)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############### Grid search part ###############
# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Grid Search
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
# grid_search.fit(X_train, y_train)

# # Best Parameters and Model
# print("Best Parameters:", grid_search.best_params_)
# best_rf = grid_search.best_estimator_

# # Predictions
# y_pred_tuned = best_rf.predict(X_test)

# # Evaluation
# print("MAE:", mean_absolute_error(y_test, y_pred_tuned))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
# print("R² Score:", r2_score(y_test, y_pred_tuned))


# --------------------------------------------
# Train a Random Forest Regressor
# --------------------------------------------

# Random Forest Regression
rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=None, 
    min_samples_leaf=1, 
    min_samples_split=2, 
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")


# --------------------------------------------
# Partial Dependence Plots
# --------------------------------------------

# # Create facet-like partial dependence plots
# features_to_plot = ['Row Centre Density', 'Average Quadrat Density', 'Standard Deviation of X Coordinates']
# num_features = len(features_to_plot)

# fig, ax = plt.subplots(1, num_features, figsize=(18, 6), sharey=True)

# # Flatten axes to pass into the plot function
# ax = ax.flatten()

# PartialDependenceDisplay.from_estimator(
#     rf, 
#     X, 
#     features=features_to_plot, 
#     kind="average",
#     ax=ax[:num_features]
# )

# # Customize each subplot
# for i, axis in enumerate(ax[:num_features]):
#     axis.grid(True, linestyle="--", alpha=0.7)  # Add grid lines
#     axis.set_title(features_to_plot[i], fontsize=14, fontweight="bold")  # Title
#     axis.set_xlabel("Feature Value", fontsize=12)  # X-axis label
#     axis.set_ylabel("Partial Dependence", fontsize=12)  # Y-axis label

# # Adjust font sizes for tick labels
#     axis.tick_params(axis='both', which='major', labelsize=10)

# # Hide any unused subplots if the grid is larger than the number of features
# for unused_ax in ax[num_features:]:
#     unused_ax.set_visible(False)

# plt.suptitle("Partial Dependence Plot for Interaction", fontsize=18, fontweight='bold')
# plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout for the title
# plt.show()


# --------------------------------------------
# Cross-validation
# --------------------------------------------

from sklearn.model_selection import cross_val_score, cross_val_predict

# Cross-validation setup
cv = 5  # Number of folds

# Evaluate R² scores across folds
r2_scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
print(f"Cross-validated R² Scores: {r2_scores}")
print(f"Mean R² Score: {r2_scores.mean():.3f}\n")

# Evaluate MAE across folds
mae_scores = cross_val_score(rf, X, y, cv=cv, scoring='neg_mean_absolute_error')
print(f"Cross-validated MAE Scores (negative): {mae_scores}")
print(f"Mean MAE: {-mae_scores.mean():.3f}\n")

# Evaluate RMSE across folds
rmse_scores = cross_val_score(rf, X, y, cv=cv, scoring='neg_mean_squared_error')
print(f"Cross-validated RMSE Scores (negative): {rmse_scores}")
print(f"Mean RMSE: {(-rmse_scores.mean()) ** 0.5:.3f}\n")

# Generate cross-validated predictions
y_pred_cv = cross_val_predict(rf, X, y, cv=cv)

# Evaluate predictions from cross-validation
mae_cv = mean_absolute_error(y, y_pred_cv)
rmse_cv = root_mean_squared_error(y, y_pred_cv)
r2_cv = r2_score(y, y_pred_cv)

print("Evaluation Based on Cross-Validation Predictions:")
print(f"  MAE: {mae_cv:.3f}")
print(f"  RMSE: {rmse_cv:.3f}")
print(f"  R² Score: {r2_cv:.3f}")


# --------------------------------------------
# Feature Importance Analysis
# --------------------------------------------

# # Feature importance from the trained Random Forest model
# importance = rf.feature_importances_
# feature_names = X.columns

# # Sort features by importance
# sorted_idx = np.argsort(importance)[::-1]  # Descending order

# # Create color map
# colors = cm.viridis(importance[sorted_idx] / max(importance[sorted_idx]))

# # Plot feature importance
# plt.figure(figsize=(20, 10))
# plt.bar(range(len(importance)), importance[sorted_idx], align="center", color=colors)
# plt.xticks(range(len(importance)), feature_names[sorted_idx], rotation=45, ha='right', rotation_mode='anchor')
# plt.title("Feature Importance Analysis (Random Forest)", fontsize=20, fontweight="bold")
# plt.xlabel("")  # Remove default x-label
# plt.figtext(0.5, 0.05, "Features", ha="center", fontsize=16)
# plt.ylabel("Importance", fontsize=16)

# # Add cumulative importance
# cumulative_importance = np.cumsum(importance[sorted_idx])
# plt.plot(range(len(importance)), cumulative_importance, color="red", marker="o", label="Cumulative Importance")
# plt.axhline(y=0.1, color="orange", linestyle="--", label="Threshold (0.1)")
# plt.legend(fontsize=12)

# # Annotate bars dynamically
# for i, v in enumerate(importance[sorted_idx]):
#     if i < 2:  # Adjust first two labels to avoid overlap
#         plt.text(i, v - 0.05, f"{v:.2f}", ha='center', fontsize=20, color="red", fontweight="bold") 
#     else:  # Regular placement above the bar
#         plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=11)

# # Highlight the first two tick labels
# ax = plt.gca()
# tick_labels = ax.get_xticklabels()
# for i, label in enumerate(tick_labels):
#     if i < 2:  # Highlight first two
#         label.set_color("red")  # Change color to red
#         label.set_fontweight("bold")  # Make bold
#         label.set_fontsize(16)  # Increase font size

# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Print the feature importance values
# for i, idx in enumerate(sorted_idx):
#     print(f"{feature_names[idx]}: {importance[idx]:.3f}")