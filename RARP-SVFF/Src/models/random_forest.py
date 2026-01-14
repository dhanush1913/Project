from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators: int = 100, max_depth: int = 5):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X):
    return model.predict_proba(X)[:, 1]
