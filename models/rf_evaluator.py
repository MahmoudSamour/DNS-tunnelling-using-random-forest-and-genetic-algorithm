from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

class RFEvaluator:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def get_fitness(self, indices):
        """Standard F1 evaluation without penalty"""
        if len(indices) == 0: 
            return 0
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        clf.fit(self.X_train[:, indices], self.y_train)
        return f1_score(self.y_val, clf.predict(self.X_val[:, indices]), average='weighted')

    def evaluate_with_penalty(self, individual, total_features):
        """Evaluation with feature size penalty used in proposed GA"""
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected: 
            return 0,
        f1 = self.get_fitness(selected)
        penalty = sum(individual) / total_features
        return f1 - (0.01 * penalty),
