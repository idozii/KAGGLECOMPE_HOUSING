import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

def preprocess_data(train_data, test_data):
    # Fill NaN values
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

    # Save target variable before encoding
    y = train_data['SalePrice'].copy()
    
    # Drop target variable from train data
    train_data = train_data.drop('SalePrice', axis=1)
    
    # Combine train and test for consistent encoding
    all_data = pd.concat([train_data, test_data], sort=False)
    
    # Apply one-hot encoding to all data
    all_data_encoded = pd.get_dummies(all_data)
    
    # Split back into train and test
    X = all_data_encoded.iloc[:len(train_data)]
    test_encoded = all_data_encoded.iloc[len(train_data):]
    
    return X, y, test_encoded

X, y, test_data = preprocess_data(train_data, test_data)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print(f"Validation MAE: {mean_absolute_error(y_valid, predictions)}")

preds = model.predict(test_data)

submission = pd.DataFrame({'Id': test_data.index, 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)