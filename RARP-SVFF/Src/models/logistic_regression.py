from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_logistic(X_train, y_train, C: float = 1.0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler

def predict_logistic(model, scaler, X):
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]
