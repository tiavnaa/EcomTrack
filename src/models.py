import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

class ChurnPredictor:
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        return results
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Modele non entraine. Appelez train() d'abord.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path='outputs/models/churn_model.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Modele sauvegarde: {path}")
    
    def load_model(self, path='outputs/models/churn_model.pkl'):
        self.model = joblib.load(path)
        print(f"Modele charge: {path}")


class SalesPredictor:
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        results = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
        
        return results
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Modele non entraine. Appelez train() d'abord.")
        
        return self.model.predict(X)
    
    def predict_future(self, days, last_data):
        predictions = []
        current_data = last_data.copy()
        
        for i in range(days):
            pred = self.predict(current_data[self.feature_names].iloc[[-1]])[0]
            predictions.append(pred)
            
            new_row = current_data.iloc[-1:].copy()
            new_row['Revenue'] = pred
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return np.array(predictions)
    
    def save_model(self, path='outputs/models/sales_model.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Modele sauvegarde: {path}")
    
    def load_model(self, path='outputs/models/sales_model.pkl'):
        self.model = joblib.load(path)
        print(f"Modele charge: {path}")