import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

X, y, test_encoded, test_ids = preprocess_data(train_data, test_data)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(test_encoded)

xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',     # Updated from 'gpu_hist'
    'device': 'cuda',         # Explicitly set device to CUDA
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'seed': 42
}

gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    n_iter_no_change=25
)
gb_model.fit(X_train, y_train)

gb_preds_valid = gb_model.predict(X_valid)
gb_preds_test = gb_model.predict(test_encoded)
print(f"GradientBoosting validation MAE: {mean_absolute_error(y_valid, gb_preds_valid)}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_preds_test = np.zeros(test_encoded.shape[0])
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold+1}...")
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Convert to DMatrix for this fold
    dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    # Train model with updated parameters
    evallist_fold = [(dtrain_fold, 'train'), (dval_fold, 'valid')]
    xgb_fold = xgb.train(
        xgb_params,
        dtrain_fold,
        500,
        evallist_fold,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Validate fold model
    fold_preds = xgb_fold.predict(dval_fold)
    fold_mae = mean_absolute_error(y_fold_val, fold_preds)
    fold_scores.append(fold_mae)
    print(f"Fold {fold+1} MAE: {fold_mae}")
    
    # Add predictions to ensemble
    xgb_cv_preds_test += xgb_fold.predict(dtest) / kf.n_splits

print(f"Average K-fold MAE: {np.mean(fold_scores)}")

# Combine predictions
final_preds_test = 0.25 * gb_preds_test + 0.75 * xgb_cv_preds_test

# Save submission
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds_test})
submission.to_csv('submission.csv', index=False)