"""
Random Forest Implementation with Hyperparameter Tuning
For Energy Consumption Prediction using all three preprocessed datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
import warnings
import pickle
warnings.filterwarnings('ignore')

print("="*80)
print("RANDOM FOREST - HYPERPARAMETER TUNING FOR ENERGY CONSUMPTION")
print("="*80)

class RandomForestTuner:
    """Class to handle Random Forest training and tuning"""
    
    def __init__(self):
        self.best_models = {}
        self.results = []
    
    def load_dataset(self, file_path, target_column):
        """Load and prepare dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"\n✅ Loaded: {file_path}")
            print(f"   Shape: {df.shape}")
            
            if target_column not in df.columns:
                print(f"   ⚠️  Target '{target_column}' not found")
                print(f"   Available: {df.columns.tolist()[:10]}...")
                return None, None, None
            
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Keep numeric only
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]
            X = X.fillna(X.median())
            
            print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None, None, None, None
    
    def baseline_model(self, X_train, X_test, y_train, y_test, dataset_name):
        """Train baseline Random Forest model"""
        print(f"\n{'='*80}")
        print(f"BASELINE MODEL: {dataset_name}")
        print('='*80)
        
        # Default Random Forest
        rf_baseline = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        print("\n🌲 Training baseline Random Forest...")
        start_time = time.time()
        rf_baseline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        y_pred_train = rf_baseline.predict(X_train)
        y_pred_test = rf_baseline.predict(X_test)
        
        # Metrics
        results = {
            'Dataset': dataset_name,
            'Model': 'Baseline RF',
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Test MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
            'Training Time (s)': training_time,
            'n_estimators': 100,
            'max_depth': 'None',
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        
        print(f"\n📊 Baseline Results:")
        print(f"   Train R²: {results['Train R²']:.4f}")
        print(f"   Test R²: {results['Test R²']:.4f}")
        print(f"   Test RMSE: {results['Test RMSE']:.4f}")
        print(f"   Test MAE: {results['Test MAE']:.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        
        self.results.append(results)
        return rf_baseline, results
    
    def grid_search_tuning(self, X_train, X_test, y_train, y_test, dataset_name):
        """Perform Grid Search for hyperparameter tuning"""
        print(f"\n{'='*80}")
        print(f"GRID SEARCH TUNING: {dataset_name}")
        print('='*80)
        
        # Define parameter grid (smaller for speed)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        print(f"\n🔍 Testing {np.prod([len(v) for v in param_grid.values()])} combinations...")
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_rf.predict(X_train)
        y_pred_test = best_rf.predict(X_test)
        
        # Metrics
        results = {
            'Dataset': dataset_name,
            'Model': 'Grid Search RF',
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Test MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
            'Training Time (s)': tuning_time,
            **grid_search.best_params_
        }
        
        print(f"\n🏆 Best Parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Tuned Model Results:")
        print(f"   Train R²: {results['Train R²']:.4f}")
        print(f"   Test R²: {results['Test R²']:.4f}")
        print(f"   Test RMSE: {results['Test RMSE']:.4f}")
        print(f"   Test MAE: {results['Test MAE']:.4f}")
        print(f"   Tuning Time: {tuning_time:.2f}s")
        
        self.results.append(results)
        self.best_models[dataset_name] = best_rf;
        
        return best_rf, results;
    
    def random_search_tuning(self, X_train, X_test, y_train, y_test, dataset_name):
        """Perform Random Search for hyperparameter tuning (faster)"""
        print(f"\n{'='*80}")
        print(f"RANDOM SEARCH TUNING: {dataset_name}")
        print('='*80)
        
        # Define parameter distributions
        param_dist = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        print(f"\n🎲 Testing 20 random combinations...")
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='r2',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        # Best model
        best_rf = random_search.best_estimator_
        
        # Predictions
        y_pred_train = best_rf.predict(X_train)
        y_pred_test = best_rf.predict(X_test)
        
        # Metrics
        results = {
            'Dataset': dataset_name,
            'Model': 'Random Search RF',
            'Train R²': r2_score(y_train, y_pred_train),
            'Test R²': r2_score(y_test, y_pred_test),
            'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Test MAE': mean_absolute_error(y_test, y_pred_test),
            'Test MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
            'Training Time (s)': tuning_time,
            **random_search.best_params_
        }
        
        print(f"\n🏆 Best Parameters:")
        for param, value in random_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Tuned Model Results:")
        print(f"   Train R²: {results['Train R²']:.4f}")
        print(f"   Test R²: {results['Test R²']:.4f}")
        print(f"   Test RMSE: {results['Test RMSE']:.4f}")
        print(f"   Test MAE: {results['Test MAE']:.4f}")
        print(f"   Tuning Time: {tuning_time:.2f}s")
        
        self.results.append(results)
        
        return best_rf, results;
    
    def feature_importance_analysis(self, model, X_train, dataset_name):
        """Analyze feature importance"""
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE: {dataset_name}")
        print('='*80)
        
        # Get feature importance
        importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 6))
        top_15 = importance.head(15)
        plt.barh(range(len(top_15)), top_15['Importance'])
        plt.yticks(range(len(top_15)), top_15['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Feature Importances - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{dataset_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance
    
    def visualize_comparison(self):
        """Compare all models"""
        if not self.results:
            print("No results to visualize")
            return
        
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON VISUALIZATIONS")
        print('='*80)
        
        df = pd.DataFrame(self.results)
        
        # 1. R² Comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Test R²
        pivot_r2 = df.pivot_table(index='Model', columns='Dataset', values='Test R²')
        pivot_r2.plot(kind='barh', ax=axes[0])
        axes[0].set_title('Test R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('R² Score')
        axes[0].legend(title='Dataset')
        
        # RMSE
        pivot_rmse = df.pivot_table(index='Model', columns='Dataset', values='Test RMSE')
        pivot_rmse.plot(kind='barh', ax=axes[1])
        axes[1].set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('RMSE')
        axes[1].legend(title='Dataset')
        
        plt.tight_layout()
        plt.savefig('rf_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: rf_model_comparison.png")
        
        # 2. Training Time Comparison
        plt.figure(figsize=(12, 6))
        pivot_time = df.pivot_table(index='Model', columns='Dataset', values='Training Time (s)')
        pivot_time.plot(kind='bar')
        plt.title('Training/Tuning Time Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.legend(title='Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('rf_training_time.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: rf_training_time.png")
    
    def save_models(self):
        """Save best models"""
        print(f"\n{'='*80}")
        print("SAVING BEST MODELS")
        print('='*80)
        
        for dataset_name, model in self.best_models.items():
            filename = f"best_rf_{dataset_name.replace(' ', '_').lower()}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Saved: {filename}")
    
    def generate_report(self):
        """Generate detailed report"""
        if not self.results:
            print("No results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*80}")
        print("FINAL REPORT - RANDOM FOREST TUNING")
        print('='*80)
        
        # Best model per dataset
        print("\n🏆 BEST MODELS BY DATASET:")
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            best = dataset_df.loc[dataset_df['Test R²'].idxmax()]
            
            print(f"\n📊 {dataset}:")
            print(f"   Best Approach: {best['Model']}")
            print(f"   Test R²: {best['Test R²']:.4f}")
            print(f"   Test RMSE: {best['Test RMSE']:.4f}")
            print(f"   Test MAE: {best['Test MAE']:.4f}")
            print(f"   Test MAPE: {best['Test MAPE']:.2f}%")
            
            # Show improvement
            baseline = dataset_df[dataset_df['Model'] == 'Baseline RF'].iloc[0]
            improvement = ((best['Test R²'] - baseline['Test R²']) / baseline['Test R²']) * 100
            print(f"   Improvement over baseline: {improvement:.2f}%")
        
        # Save to CSV
        df.to_csv('random_forest_tuning_results.csv', index=False)
        print(f"\n✅ Results saved to: random_forest_tuning_results.csv")
        
        return df


def main():
    """Main execution"""    
    # Initialize tuner
    tuner = RandomForestTuner()
    
    # Dataset configurations
    datasets = [
        ('preprocessed_household_power.csv', 'Global_active_power', 'Household Power'),
        ('preprocessed_appliances_energy.csv', 'Appliances', 'Appliances Energy'),
        ('preprocessed_smart_home_energy.csv', 'Energy Consumption (kWh)', 'Smart Home Energy')
    ]
    
    # Process each dataset
    for file_path, target_col, display_name in datasets:
        print(f"\n\n{'#'*80}")
        print(f"PROCESSING: {display_name}")
        print('#'*80)
        
        # Load data
        X_train, X_test, y_train, y_test = tuner.load_dataset(file_path, target_col)
        
        if X_train is not None:
            # 1. Baseline model
            baseline_model, baseline_results = tuner.baseline_model(
                X_train, X_test, y_train, y_test, display_name
            )
            
            # 2. Grid Search (choose one based on dataset size)
            if X_train.shape[0] < 100000:  # For smaller datasets
                best_model, grid_results = tuner.grid_search_tuning(
                    X_train, X_test, y_train, y_test, display_name
                )
            else:
                # 3. Random Search (faster for large datasets)
                best_model, random_results = tuner.random_search_tuning(
                    X_train, X_test, y_train, y_test, display_name
                )
            
            # 4. Feature importance
            importance = tuner.feature_importance_analysis(best_model, X_train, display_name)
    
    # Generate comparisons
    tuner.visualize_comparison()
    
    # Save models
    tuner.save_models()
    
    # Final report
    results_df = tuner.generate_report()
    
    print(f"\n{'='*80}")
    print("✅ RANDOM FOREST TUNING COMPLETE!")
    print('='*80)
    print("\n📁 Generated Files:")
    print("   - random_forest_tuning_results.csv")
    print("   - rf_model_comparison.png")
    print("   - rf_training_time.png")
    print("   - feature_importance_*.png (for each dataset)")
    print("   - best_rf_*.pkl (saved models)")
    print('='*80 + "\n")


if __name__ == "__main__":
    main()