#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BigMart Sales Prediction Solution

A comprehensive script for data preprocessing, feature engineering, 
model building, and prediction for the BigMart sales prediction challenge.

The goal is to predict the sales of different products across various 
BigMart outlets based on product attributes and store information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """
    Main function to execute the entire BigMart sales prediction pipeline.
    """
    print("Starting BigMart Sales Prediction...")
    
    # Step 1: Load the datasets
    try:
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure 'train.csv' and 'test.csv' are in the current directory.")
        return
    
    # Step 2: Data Preprocessing and Feature Engineering
    print("\nPerforming data preprocessing and feature engineering...")
    X_train, y_train, X_test = preprocess_data(train_data, test_data)
    
    # Step 3: Model Building and Evaluation
    print("\nTraining and evaluating models...")
    best_model, predictions = build_and_evaluate_models(X_train, y_train, X_test)
    
    # Step 4: Create Submission File
    print("\nCreating submission file...")
    create_submission(test_data, predictions)
    
    print("\nBigMart Sales Prediction completed successfully!")

def preprocess_data(train_data, test_data):
    """
    Preprocess the raw data and engineer new features.
    
    Args:
        train_data (pd.DataFrame): The raw training data
        test_data (pd.DataFrame): The raw test data
        
    Returns:
        tuple: A tuple containing X_train, y_train, and X_test
    """
    # Create a flag to identify the datasets
    train_data['source'] = 'train'
    test_data['source'] = 'test'
    
    # Save the target variable separately
    target = train_data['Item_Outlet_Sales'].copy() if 'Item_Outlet_Sales' in train_data.columns else None
    
    # Combine datasets for consistent preprocessing
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # 1. HANDLING MISSING VALUES
    
    # 1.1 Item_Weight missing values
    # Group by Item_Identifier and fill missing values with the mean weight of each product
    print("Handling missing values in Item_Weight...")
    item_avg_weight = data.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    data['Item_Weight'].fillna(item_avg_weight, inplace=True)
    
    # If there are still missing values, fill with the overall mean
    if data['Item_Weight'].isnull().sum() > 0:
        global_avg_weight = data['Item_Weight'].mean()
        data['Item_Weight'].fillna(global_avg_weight, inplace=True)
    
    # 1.2 Outlet_Size missing values
    print("Handling missing values in Outlet_Size...")
    # Create mapping based on Outlet_Type to fill missing Outlet_Size values
    outlet_size_mapping = {
        'Grocery Store': 'Small',
        'Supermarket Type1': 'Small',
        'Supermarket Type2': 'Medium',
        'Supermarket Type3': 'Medium'
    }
    
    # Fill missing values based on Outlet_Type
    for outlet_type, size in outlet_size_mapping.items():
        data.loc[(data['Outlet_Size'].isnull()) & (data['Outlet_Type'] == outlet_type), 'Outlet_Size'] = size
    
    # If any Outlet_Size values are still missing, fill with 'Small' (most common)
    data['Outlet_Size'].fillna('Small', inplace=True)
    
    # 2. DATA CLEANING
    
    # 2.1 Standardize Fat Content
    print("Standardizing Item_Fat_Content values...")
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    })
    
    # 2.2 Fix zero visibility values
    print("Fixing zero visibility values...")
    # Calculate the mean visibility by Item_Type
    visibility_means = data.groupby('Item_Type')['Item_Visibility'].transform('mean')
    # Replace 0 visibility values with the mean of their respective Item_Type
    zero_indices = data['Item_Visibility'] == 0
    data.loc[zero_indices, 'Item_Visibility'] = visibility_means[zero_indices]
    
    # 3. FEATURE ENGINEERING
    
    # 3.1 Create Item_Category from Item_Identifier prefix
    print("Engineering new features...")
    data['Item_Category'] = data['Item_Identifier'].apply(lambda x: x[:2])
    # Map to meaningful categories
    data['Item_Category'] = data['Item_Category'].map({
        'FD': 'Food',
        'DR': 'Drinks',
        'NC': 'Non-Consumable'
    })
    
    # 3.2 Create Outlet_Age feature
    data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
    
    # 3.3 Create MRP bins
    data['Item_MRP_Bin'] = pd.cut(
        data['Item_MRP'], 
        bins=[0, 50, 100, 150, 200, 300], 
        labels=['0-50', '50-100', '100-150', '150-200', '200+']
    )
    
    # 3.4 Create visibility bins
    data['Item_Visibility_Bin'] = pd.qcut(
        data['Item_Visibility'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # 3.5 Create Outlet_Age bins
    data['Outlet_Age_Bin'] = pd.cut(
        data['Outlet_Age'], 
        bins=[0, 5, 10, 15, 20, 30], 
        labels=['0-5', '6-10', '11-15', '16-20', '20+']
    )
    
    # 3.6 Create interaction features
    data['Item_MRP_By_Outlet_Type'] = data['Item_MRP'] * data['Outlet_Type'].map({
        'Grocery Store': 1,
        'Supermarket Type1': 2,
        'Supermarket Type2': 3,
        'Supermarket Type3': 4
    })
    
    # 4. ENCODING CATEGORICAL VARIABLES
    
    # Convert categorical variables to numeric for modeling
    print("Encoding categorical variables...")
    
    # 4.1 Identify categorical columns
    categorical_cols = [
        'Item_Fat_Content', 
        'Item_Type', 
        'Outlet_Identifier',
        'Outlet_Size', 
        'Outlet_Location_Type', 
        'Outlet_Type',
        'Item_Category',
        'Item_MRP_Bin',
        'Item_Visibility_Bin',
        'Outlet_Age_Bin'
    ]
    
    # 4.2 Apply one-hot encoding - handle each column separately to avoid errors
    for col in categorical_cols:
        try:
            # Create dummy variables for each categorical column
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            # Add the dummy variables to the dataframe
            data = pd.concat([data, dummies], axis=1)
            # Drop the original categorical column
            data = data.drop(col, axis=1)
        except Exception as e:
            print(f"Error encoding column {col}: {e}")
    
    # 5. FEATURE SELECTION
    
    # 5.1 Remove unnecessary columns
    print("Performing feature selection...")
    cols_to_drop = ['Item_Identifier', 'Outlet_Establishment_Year', 'source']
    
    # If Item_Outlet_Sales exists in combined data (it might not in test data), drop it
    if 'Item_Outlet_Sales' in data.columns:
        cols_to_drop.append('Item_Outlet_Sales')
        
    data_processed = data.drop(columns=cols_to_drop, errors='ignore')
    
    # 6. CONVERT BOOLEAN COLUMNS TO INTEGERS
    
    # Convert boolean columns to integers for better compatibility with models
    bool_cols = [col for col in data_processed.columns if data_processed[col].dtype == bool]
    for col in bool_cols:
        data_processed[col] = data_processed[col].astype(int)
    
    # 7. HANDLE REMAINING CATEGORICAL COLUMNS
    
    # Identify any remaining object/string columns
    object_cols = data_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        try:
            # Try to convert to numeric if possible
            data_processed[col] = pd.to_numeric(data_processed[col])
        except:
            # Otherwise, use label encoding
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col])
    
    # 8. SPLIT BACK TO TRAIN AND TEST
    
    # Get the indices for train and test sets
    train_idx = data[data['source'] == 'train'].index
    test_idx = data[data['source'] == 'test'].index
    
    # Split the processed data
    X_train = data_processed.iloc[train_idx]
    X_test = data_processed.iloc[test_idx]
    
    # Make sure column types are consistent
    for col in X_train.columns:
        if col in X_test.columns and X_train[col].dtype != X_test[col].dtype:
            try:
                # Try to convert to a common datatype
                common_type = np.find_common_type([X_train[col].dtype, X_test[col].dtype], [])
                X_train[col] = X_train[col].astype(common_type)
                X_test[col] = X_test[col].astype(common_type)
            except:
                print(f"Warning: Column {col} has different types in train and test sets.")
    
    # 9. CONFIRM NO MISSING VALUES
    print(f"Missing values in processed train data: {X_train.isnull().sum().sum()}")
    print(f"Missing values in processed test data: {X_test.isnull().sum().sum()}")
    
    # 10. Ensure identical column sets between train and test
    # This is critical to avoid the feature mismatch error during prediction
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    
    cols_to_add_to_test = train_cols - test_cols
    for col in cols_to_add_to_test:
        X_test[col] = 0  # Add missing columns to test set with default value 0
        
    cols_to_add_to_train = test_cols - train_cols
    for col in cols_to_add_to_train:
        X_train[col] = 0  # Add missing columns to train set with default value 0
    
    # Ensure column order is identical
    X_test = X_test[X_train.columns]
    
    return X_train, target, X_test

def build_and_evaluate_models(X_train, y_train, X_test):
    """
    Build and evaluate multiple regression models to predict sales.
    
    Args:
        X_train (pd.DataFrame): The processed training features
        y_train (pd.Series): The target variable (sales)
        X_test (pd.DataFrame): The processed test features
        
    Returns:
        tuple: A tuple containing the best model and predictions for the test set
    """
    if y_train is None:
        print("Error: Target variable is None")
        return None, None
    
    # Split data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train_split.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Function to evaluate models
    def evaluate_model(model, X_train, X_val, y_train, y_val):
        """
        Train and evaluate a model using RMSE and R² metrics.
        
        Args:
            model: The machine learning model to evaluate
            X_train, X_val: Training and validation features
            y_train, y_val: Training and validation targets
            
        Returns:
            tuple: A tuple containing the validation RMSE and the trained model
        """
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        train_r2 = r2_score(y_train, train_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Validation RMSE: {val_rmse:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        return val_rmse, model
    
    # Train and evaluate multiple models
    models_results = {}
    
    print("\n--- Linear Regression ---")
    try:
        lr = LinearRegression()
        lr_rmse, lr_model = evaluate_model(lr, X_train_split, X_val, y_train_split, y_val)
        models_results['Linear Regression'] = (lr_rmse, lr_model)
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
    
    print("\n--- Decision Tree ---")
    try:
        dt = DecisionTreeRegressor(random_state=42)
        dt_rmse, dt_model = evaluate_model(dt, X_train_split, X_val, y_train_split, y_val)
        models_results['Decision Tree'] = (dt_rmse, dt_model)
    except Exception as e:
        print(f"Error training Decision Tree: {e}")
    
    print("\n--- Random Forest ---")
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_rmse, rf_model = evaluate_model(rf, X_train_split, X_val, y_train_split, y_val)
        models_results['Random Forest'] = (rf_rmse, rf_model)
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    print("\n--- XGBoost ---")
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_rmse, xgb_model = evaluate_model(xgb_model, X_train_split, X_val, y_train_split, y_val)
        models_results['XGBoost'] = (xgb_rmse, xgb_model)
    except Exception as e:
        print(f"Error training XGBoost: {e}")
    
    # Compare model performances
    print("\n--- Model Comparison ---")
    if not models_results:
        print("No models were successfully trained.")
        return None, None
        
    for model_name, (rmse, _) in sorted(models_results.items(), key=lambda x: x[1][0]):
        print(f"{model_name}: RMSE = {rmse:.2f}")
    
    # Find the best model
    best_model_name = min(models_results.keys(), key=lambda k: models_results[k][0])
    best_rmse, best_model = models_results[best_model_name]
    print(f"\nBest model: {best_model_name} with RMSE = {best_rmse:.2f}")
    
    # If the best model is tree-based, show feature importance
    if best_model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        print("\n--- Feature Importance ---")
        try:
            importances = best_model.feature_importances_
            feat_names = X_train.columns
            
            # Create a DataFrame for better visualization
            feature_importance = pd.DataFrame({
                'Feature': feat_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top 15 features
            print(feature_importance.head(15))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Top 15 Feature Importance - {best_model_name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
    
    # Retrain the best model on the full training set before prediction
    try:
        print("\nRetraining best model on full training set...")
        best_model.fit(X_train, y_train)
        
        # Make predictions on the test set
        predictions = best_model.predict(X_test)
        print(f"Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"Error making predictions: {e}")
        predictions = None
    
    return best_model, predictions

def create_submission(test_data, predictions):
    """
    Create a submission file with predictions.
    
    Args:
        test_data (pd.DataFrame): The original test data
        predictions (np.array): The predicted sales values
    """
    if predictions is None:
        print("No predictions to save.")
        return
    
    try:
        # Flatten predictions if they are multi-dimensional
        if hasattr(predictions, 'ndim') and predictions.ndim > 1:
            predictions = predictions.flatten()
        
        # Create submission DataFrame
        # Create submission DataFrame
        submission = pd.DataFrame({
            'Item_Identifier': test_data['Item_Identifier'],
            'Outlet_Identifier': test_data['Outlet_Identifier'],
            'Item_Outlet_Sales': np.round(np.maximum(predictions, 0)).astype(int)  # Make sure values are positive then round to integers
        })
        
        # Save to CSV
        submission_path = 'submissions.csv'
        submission.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
        print(f"Prediction sample (first 5 rows):")
        print(submission.head())
    except Exception as e:
        print(f"Error creating submission file: {e}")

if __name__ == "__main__":
    main()