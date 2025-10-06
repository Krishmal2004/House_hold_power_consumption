import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# Function to load data
def load_data(file_paths):
    data = {}
    for file in file_paths:
        data[file] = pd.read_csv(file)
    return data

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return rmse, mae, r2, mape, training_time

# Function to run comparisons
def run_comparisons(data):
    results = {}
    for file, df in data.items():
        X = df.drop('target', axis=1)  # Replace 'target' with actual target column
        y = df['target']  # Replace 'target' with actual target column

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'AdaBoost': AdaBoostRegressor(),
            'KNN': KNeighborsRegressor(),
            'SVR': SVR(),
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'CatBoost': CatBoostRegressor(silent=True),
        }

        for name, model in models.items():
            rmse, mae, r2, mape, training_time = evaluate_model(model, X_train, y_train, X_test, y_test)
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape,
                'Training Time': training_time,
            }

    return results

# Function to visualize results
def visualize_results(results):
    df_results = pd.DataFrame(results).T
    df_results.sort_values('RMSE', inplace=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=df_results.index, y='RMSE', data=df_results)
    plt.xticks(rotation=45)
    plt.title('Model Comparison - RMSE')
    plt.ylabel('RMSE')
    plt.show()

    # More visualizations can be added

# Main function
if __name__ == "__main__":
    file_paths = [
        'preprocessed_household_power.csv',
        'preprocessed_appliances_energy.csv',
        'preprocessed_smart_home_energy.csv'
    ]
    
    data = load_data(file_paths)
    results = run_comparisons(data)
    visualize_results(results)

    # Generate detailed report
    print("Detailed Report:")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")
