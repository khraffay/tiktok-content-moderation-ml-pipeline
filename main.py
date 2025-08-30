from data_loader import load_data
from preprocessing import preprocess_data
from train import train_random_forest, train_xgboost
from evaluate import evaluate_model
from utils import plot_text_length_distribution
import numpy as np
import os
import time

def main():
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print("All plots will be saved to this directory")
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Basic EDA
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Claim status distribution:")
    print(df["claim_status"].value_counts())
    
    # Plot text length distribution FIRST (before preprocessing)
    print("Creating text length distribution plot...")
    plot_text_length_distribution(df)
    print("✓ Text length plot created!")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)
    print("✓ Data preprocessing completed!")
    
    # Debug: Check target variable types
    print(f"y_train type: {type(y_train)}")
    print(f"y_train dtype: {y_train.dtype}")
    print(f"y_train unique values: {np.unique(y_train)}")
    
    # Train models with timing
    print("Training Random Forest...")
    start_time = time.time()
    rf_model = train_random_forest(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"✓ Random Forest training completed in {rf_time:.2f} seconds!")
    print("Best RF params:", rf_model.best_params_)
    
    # Evaluate Random Forest immediately
    print("Evaluating Random Forest...")
    rf_metrics = evaluate_model(rf_model, X_val, y_val, "Random_Forest")
    print("✓ Random Forest evaluation completed!")
    
    print("Training XGBoost...")
    start_time = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    xgb_time = time.time() - start_time
    print(f"✓ XGBoost training completed in {xgb_time:.2f} seconds!")
    print("Best XGB params:", xgb_model.best_params_)
    
    # Evaluate XGBoost immediately
    print("Evaluating XGBoost...")
    xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "XGBoost")
    print("✓ XGBoost evaluation completed!")
    
    # Compare models
    print("\n=== Model Comparison ===")
    print("Random Forest - Recall:", rf_metrics['recall'])
    print("XGBoost - Recall:", xgb_metrics['recall'])
    
    # Select best model based on recall
    if rf_metrics['recall'] > xgb_metrics['recall']:
        best_model = rf_model
        print("Best model: Random Forest")
    else:
        best_model = xgb_model
        print("Best model: XGBoost")
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    final_metrics = evaluate_model(best_model, X_test, y_test, "Best_Model")
    
    print("Project completed successfully!")
    print("All plots saved to current directory")
    
    # List all PNG files in current directory
    png_files = [f for f in os.listdir(current_dir) if f.endswith('.png')]
    print("Generated plot files:", png_files)

if __name__ == "__main__":
    main()