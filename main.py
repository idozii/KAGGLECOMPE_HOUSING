import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

def engineer_features(data):
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
    data['TotalBathrooms'] = data['FullBath'] + (0.5 * data['HalfBath']) + \
                             data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']
    data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
    data['IsRemodeled'] = (data['YearRemodAdd'] != data['YearBuilt']).astype(int)
    
    skewed_cols = ['LotArea', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']
    for col in skewed_cols:
        if col in data.columns:
            data[col] = np.log1p(data[col])
    
    return data

def preprocess_data(train_data, test_data):
    test_ids = test_data['Id'].copy()
    
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

    y = train_data['SalePrice'].copy()
    train_data = train_data.drop('SalePrice', axis=1)
    
    all_data = pd.concat([train_data, test_data], sort=False)
    
    all_data = engineer_features(all_data)

    all_data_encoded = pd.get_dummies(all_data)
    
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
    'tree_method': 'hist',     
    'device': 'cuda',         
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
    
    dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval_fold = xgb.DMatrix(X_fold_val, label=y_fold_val)
    
    evallist_fold = [(dtrain_fold, 'train'), (dval_fold, 'valid')]
    xgb_fold = xgb.train(
        xgb_params,
        dtrain_fold,
        500,
        evallist_fold,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    fold_preds = xgb_fold.predict(dval_fold)
    fold_mae = mean_absolute_error(y_fold_val, fold_preds)
    fold_scores.append(fold_mae)
    print(f"Fold {fold+1} MAE: {fold_mae}")
    
    xgb_cv_preds_test += xgb_fold.predict(dtest) / kf.n_splits

print(f"Average K-fold MAE: {np.mean(fold_scores)}")

final_preds_test = 0.25 * gb_preds_test + 0.75 * xgb_cv_preds_test

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds_test})
submission.to_csv('submission.csv', index=False)