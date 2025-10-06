import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# Function to evaluate model
def evaluate_model(model, X_train, y_train):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    mae = mean_absolute_error(y_train, predictions)
    r2 = r2_score(y_train, predictions)

    return rmse, mae, r2, training_time

# Load datasets
datasets = ['preprocessed_household_power.csv', 
            'preprocessed_appliances_energy.csv', 
            'preprocessed_smart_home_energy.csv']

results = {}

for dataset in datasets:
    data = load_data(dataset)
    if data is not None:
        X = data.drop('target', axis=1)  # assuming 'target' is the column to predict
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'Support Vector Regression': SVR(),
            'Neural Network': MLPRegressor(max_iter=1000)
        }

        # Evaluate models
        for name, model in models.items():
            rmse, mae, r2, training_time = evaluate_model(model, X_train, y_train)
            results[name] = results.get(name, []) + [(dataset, rmse, mae, r2, training_time)]

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results, columns=['Dataset', 'RMSE', 'MAE', 'R2', 'Training Time'])

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars='Dataset'), x='variable', y='value', hue='Dataset')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.legend(title='Dataset')
plt.tight_layout()
plt.show()

# Output the best performing model
best_model = min(results_df['RMSE'])  # example for RMSE; can change based on criteria
print(f"The best model based on RMSE is: {best_model}")
