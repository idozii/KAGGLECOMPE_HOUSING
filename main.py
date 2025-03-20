import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import os

# Start timing
start_time = time.time()

# Ensure GPU is visible
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load data
print("Loading data...")
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

def preprocess_data(train_data, test_data):
    test_ids = test_data['Id'].copy()
    
    # Fill missing values
    for c in train_data.columns:
        if train_data[c].dtype == 'object':
            train_data[c] = train_data[c].fillna(train_data[c].mode()[0])
        else:
            train_data[c] = train_data[c].fillna(train_data[c].median())

    for c in test_data.columns:
        if test_data[c].dtype == 'object':
            test_data[c] = test_data[c].fillna(test_data[c].mode()[0])
        else:
            test_data[c] = test_data[c].fillna(test_data[c].median())

    # Extract target variable
    y = train_data['SalePrice'].copy()
    
    # Drop target from features
    train_data = train_data.drop('SalePrice', axis=1)
    
    # Combine for consistent encoding
    all_data = pd.concat([train_data, test_data], sort=False)
    
    # One-hot encode categorical features
    all_data_encoded = pd.get_dummies(all_data)
    
    # Split back to train and test
    X = all_data_encoded.iloc[:len(train_data)]
    test_encoded = all_data_encoded.iloc[len(train_data):]
    
    return X, y, test_encoded, test_ids

# Process data
X, y, test_encoded, test_ids = preprocess_data(train_data, test_data)

# Split for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Convert to DMatrix for better GPU performance
print("Converting to DMatrix format for GPU optimization...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(test_encoded)

# XGBoost parameters optimized for GPU
params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'seed': 42
}

# Set up evaluation metrics
evallist = [(dtrain, 'train'), (dvalid, 'valid')]

# Train model using native XGBoost API
print("Training model with GPU acceleration...")
num_rounds = 1000
early_stopping = 50

model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evallist,
    early_stopping_rounds=early_stopping,
    verbose_eval=25  # Print progress every 25 iterations
)

# Save best iteration number
best_iteration = model.best_iteration
print(f"Best iteration: {best_iteration}")

# Make predictions
print("Making predictions...")
predictions = model.predict(dvalid)
mae = mean_absolute_error(y_valid, predictions)
print(f"Validation MAE: {mae}")

# Generate test predictions
preds = model.predict(dtest)

# Create submission file
print("Creating submission file...")
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)

# Report performance
elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")