from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, max_depth: int = 5):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_decision_tree(model, X):
    return model.predict_proba(X)[:, 1]
